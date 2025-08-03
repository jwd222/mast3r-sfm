# --- Imports needed from the original mast3r codebase or common libraries ---
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import json
import cv2
from torchvision import transforms


# You will need these helper functions from the original mast3r repository's utils.
# Make sure they are available in your Python path.
from dust3r.utils.geometry import xy_grid
from mast3r.fast_nn import bruteforce_reciprocal_nns
from kapture.converter.colmap.database_extra import kapture_to_colmap, get_colmap_image_ids_from_db
from dust3r.inference import inference

# --- FOR PERFECT REUSE, IMPORT ImgNorm DIRECTLY FROM THE SOURCE ---
# Make sure the dust3r library is in your Python environment.
try:
    from dust3r.datasets.utils.transforms import ImgNorm
    print("Successfully imported ImgNorm from dust3r library.")
except ImportError:
    print("Warning: Could not import ImgNorm from dust3r. Defining it manually.")
    print("For best results, ensure the dust3r library is installed and accessible.")
    ImgNorm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def extract_matches_from_tile_prediction(pred1, pred2, conf_thr=3.0, device='cuda'):
    """
    ADAPTED from the core logic of `get_im_matches`.
    This function takes the raw output of the MASt3R model for a SINGLE tile pair
    and extracts the 2D-2D point correspondences in their local tile coordinates.

    Args:
        pred1 (dict): The model's prediction for the first tile.
        pred2 (dict): The model's prediction for the second tile.
        conf_thr (float): The confidence threshold to filter keypoints.
        device (str): The device to use for calculations.

    Returns:
        tuple: A tuple containing (matches_im0, matches_im1), where each is a
               numpy array of shape (N, 2) with keypoint coordinates local to their tile.
               Returns (None, None) if no matches are found.
    """
    # This logic is for dense matching using descriptors, copied from get_im_matches
    descs = [pred1['desc'], pred2['desc']]
    confidences = [pred1['desc_conf'], pred2['desc_conf']]
    desc_dim = descs[0].shape[-1]
    
    confidence_masks = [confidences[0] >= conf_thr, confidences[1] >= conf_thr]
    
    pts2d_list, desc_list = [], []
    for j in range(2):
        # Get shape directly from the descriptor tensor: (1, H, W, C) -> (H, W)
        true_shape_j = descs[j].shape[1:3]
        
        conf_j = confidence_masks[j].cpu().numpy().flatten()
        pts2d_j = xy_grid(true_shape_j[1], true_shape_j[0]).reshape(-1, 2)[conf_j]
        desc_j = descs[j].detach().cpu().numpy().reshape(-1, desc_dim)[conf_j]
        
        pts2d_list.append(pts2d_j)
        desc_list.append(desc_j)
        
    if len(desc_list[0]) == 0 or len(desc_list[1]) == 0:
        return None, None

    # Use the existing reciprocal nearest neighbor matching function
    nn0, nn1 = bruteforce_reciprocal_nns(
        desc_list[0], desc_list[1],
        device=device, dist='dot', block_size=2**13
    )
    reciprocal_in_P0 = (nn1[nn0] == np.arange(len(nn0)))

    # Get the final matched coordinates in their respective local tile spaces
    matches_im1 = pts2d_list[1][nn0][reciprocal_in_P0]
    matches_im0 = pts2d_list[0][reciprocal_in_P0]

    return matches_im0, matches_im1


def deduplicate_and_format_for_colmap(aggregated_matches, distance_threshold=2.0):
    """
    REPLACES the DisjointSet track-building logic from `export_matches`.
    Takes all raw matches, deduplicates keypoints within each image, and creates
    the final data structures needed for COLMAP export.
    
    Args:
        aggregated_matches (dict): The dictionary of raw, globally-reprojected matches.
        distance_threshold (float): Pixel distance to consider keypoints duplicates.

    Returns:
        tuple: (unique_kpts_per_image, final_matches_indexed)
    """
    # This function was well-defined in the previous step and can be used as is.
    # It correctly replaces the need for manual track building.
    
    # --- Step 1: Collect all detected keypoints for each image ---
    all_kpts_per_image = {}
    for (path_A, path_B), match_data in aggregated_matches.items():
        if path_A not in all_kpts_per_image: all_kpts_per_image[path_A] = []
        if path_B not in all_kpts_per_image: all_kpts_per_image[path_B] = []
        all_kpts_per_image[path_A].append(match_data["kpts0"])
        all_kpts_per_image[path_B].append(match_data["kpts1"])

    # --- Step 2: Deduplicate keypoints for each image ---
    unique_kpts_per_image = {}
    print("Deduplicating keypoints for each image...")
    for path, kpt_list in all_kpts_per_image.items():
        if not kpt_list:
            unique_kpts_per_image[path] = np.array([])
            continue
            
        all_kpts = np.concatenate(kpt_list, axis=0)
        # Using rounding for simple, fast deduplication.
        _, unique_indices = np.unique(np.round(all_kpts), axis=0, return_index=True)
        unique_kpts_per_image[path] = all_kpts[unique_indices]

    # --- Step 3: Re-map matches to use the new unique keypoint indices ---
    print("Re-indexing matches...")
    final_matches_indexed = {}
    for (path_A, path_B), match_data in aggregated_matches.items():
        kpts_A_unique = unique_kpts_per_image[path_A]
        kpts_B_unique = unique_kpts_per_image[path_B]

        if kpts_A_unique.shape[0] == 0 or kpts_B_unique.shape[0] == 0:
            continue

        tree_A = cKDTree(kpts_A_unique)
        tree_B = cKDTree(kpts_B_unique)

        dist_A, indices_A = tree_A.query(match_data["kpts0"], distance_upper_bound=distance_threshold)
        dist_B, indices_B = tree_B.query(match_data["keypoints1"], distance_upper_bound=distance_threshold)

        valid_mask = (indices_A < len(kpts_A_unique)) & (indices_B < len(kpts_B_unique))
        
        final_matches_indexed[(path_A, path_B)] = np.stack([indices_A[valid_mask], indices_B[valid_mask]], axis=-1)

    return unique_kpts_per_image, final_matches_indexed


def run_chunked_mast3r_matching(
    model, device, image_pairs_to_match, root_path, colmap_db,
    precomputed_transforms,
    tile_size=(1024, 1024), overlap_px=256, conf_thr=3.0):
    """
    The COMPLETE REPLACEMENT for the original `run_mast3r_matching` function.
    It orchestrates the tiling, matching, aggregation, and database export.
    """
    
    # --- Part A: Tiling and Tile-Pair Determination ---
    tile_cache = {}
    all_image_paths_abs = np.unique([p for pair in image_pairs_to_match for p in pair])
    
    for image_path in tqdm(all_image_paths_abs, total=len(all_image_paths_abs), desc="Preparing tiles"):
        if image_path not in tile_cache:
            tile_cache[image_path] = generate_image_tiles(image_path, tile_size, overlap_px)

    # --- Part B: Inference on Tile Pairs and Match Aggregation ---
    aggregated_matches = {}
    batch_size = 4 # Adjust based on GPU memory

    print("Beginning chunked feature matching...")
    for image_path1, image_path2 in tqdm(image_pairs_to_match, desc="Processing Image Pairs"):
        # Get basenames for transform lookup
        name1_base = os.path.splitext(os.path.basename(image_path1))[0]
        name2_base = os.path.splitext(os.path.basename(image_path2))[0]
        transform_1_to_2 = precomputed_transforms.get((name1_base, name2_base))
        
        if transform_1_to_2 is None: continue

        tiles1 = tile_cache[image_path1]
        tiles2 = tile_cache[image_path2]
        
        overlapping_tile_pairs = determine_overlapping_tile_pairs(tiles1, tiles2, transform_1_to_2)
        if not overlapping_tile_pairs: continue

        # Process all tile pairs for this image pair in batches
        for i in range(0, len(overlapping_tile_pairs), batch_size):
            batch_of_pairs = overlapping_tile_pairs[i:i + batch_size]
            
            # Create a LIST of TUPLES in the exact format `inference` expects.
            inference_input_batch = []
            for tile_a, tile_b in batch_of_pairs:
                #
                # --- CORRECT NORMALIZATION APPLIED HERE ---
                # We apply the official ImgNorm directly to our tile data.
                # The tile data is a numpy array (H, W, C), which is a valid
                # input for ImgNorm (via its ToTensor() component).
                # NO RESIZING IS NEEDED OR WANTED HERE.
                #
                img_a_tensor = ImgNorm(tile_a['tile_data'])
                img_b_tensor = ImgNorm(tile_b['tile_data'])
                
                #
                # --- START: THE CRUCIAL FIX FOR THE MODEL'S FORWARD PASS ---
                #
                # The model requires 'true_shape' for the patch embedder and
                # 'instance' for the is_symmetrized check. We provide them here.
                #
                shape_a = torch.tensor(img_a_tensor.shape[-2:]).unsqueeze(0) # Shape: (1, 2)
                shape_b = torch.tensor(img_b_tensor.shape[-2:]).unsqueeze(0)
                
                # The 'instance' key can be a dummy value since we are not using the
                # symmetrized batch optimization. We use the tile_id for uniqueness.
                instance_a = tile_a['tile_id']
                instance_b = tile_b['tile_id']

                # Create the dictionary with ALL required keys for the model.
                dict_a = {
                    'img': img_a_tensor.unsqueeze(0),
                    'true_shape': shape_a,
                    'instance': instance_a
                }
                dict_b = {
                    'img': img_b_tensor.unsqueeze(0),
                    'true_shape': shape_b,
                    'instance': instance_b
                }
                # --- END: THE CRUCIAL FIX ---
                
                # Append the TUPLE of dictionaries to our batch list.
                inference_input_batch.append((dict_a, dict_b))
                
            # Run inference.
            with torch.no_grad():
                output = inference(inference_input_batch, model, device, batch_size=len(inference_input_batch), verbose=False)                

            # The output predictions will be batched, so we need to access them correctly.
            # The 'output' dictionary contains tensors where the first dimension is the batch size.

            # Extract, re-project, and aggregate matches for each result in the batch
            for j, (tile_a, tile_b) in enumerate(batch_of_pairs):
                # We need to index into the batched prediction tensors
                pred1_single = {key: val[j] for key, val in output['pred1'].items()}
                pred2_single = {key: val[j] for key, val in output['pred2'].items()}
                
                kpts1_local, kpts2_local = extract_matches_from_tile_prediction(pred1_single, pred2_single, conf_thr, device)

                if kpts1_local is None: continue

                offset_a = tile_a["bounds_in_parent"][:2]
                offset_b = tile_b["bounds_in_parent"][:2]
                kpts1_global = kpts1_local + offset_a
                kpts2_global = kpts2_local + offset_b

                pair_key = (image_path1, image_path2)
                if pair_key not in aggregated_matches:
                    aggregated_matches[pair_key] = {"kpts0": [], "kpts1": []}
                
                aggregated_matches[pair_key]["kpts0"].append(kpts1_global)
                aggregated_matches[pair_key]["kpts1"].append(kpts2_global)
    
    # --- Part C: Finalize Aggregation, Deduplicate, and Export ---
    # Finalize aggregation by concatenating lists of arrays
    for pair_key, kpt_data in aggregated_matches.items():
        if not kpt_data["kpts0"]: continue
        aggregated_matches[pair_key]["keypoints0"] = np.concatenate(kpt_data["kpts0"], axis=0)
        aggregated_matches[pair_key]["keypoints1"] = np.concatenate(kpt_data["kpts1"], axis=0)
        
    unique_kpts, indexed_matches = deduplicate_and_format_for_colmap(aggregated_matches)
    
    # Logic adapted from the END of original `export_matches`
    print("Exporting final keypoints and matches to COLMAP database...")
    colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)
    
    # Export unique keypoints
    for image_path, keypoints in unique_kpts.items():
        image_rel_path = os.path.relpath(image_path, root_path).replace('\\', '/')
        if image_rel_path in colmap_image_ids:
            colmap_id = colmap_image_ids[image_rel_path]
            # Our keypoints are already in the final coordinate system.
            # Add 0.5 to center them in the pixel, as COLMAP expects.
            colmap_db.add_keypoints(colmap_id, keypoints + 0.5)

    # Export indexed matches
    for (path1, path2), matches in indexed_matches.items():
        rel_path1 = os.path.relpath(path1, root_path).replace('\\', '/')
        rel_path2 = os.path.relpath(path2, root_path).replace('\\', '/')
        
        # Ensure correct ordering for COLMAP (id1 < id2)
        id1, id2 = colmap_image_ids[rel_path1], colmap_image_ids[rel_path2]
        if id1 > id2:
            id1, id2 = id2, id1
            matches = matches[:, ::-1] # Swap columns if we swapped image order

        colmap_db.add_matches(id1, id2, matches)
    
    colmap_db.commit()

    return indexed_matches

def generate_image_tiles(image_path, tile_size=(1024, 1024), overlap_px=256):
    """
    Divides a large image into smaller, overlapping tiles.

    Args:
        image_path (str): The file path to the high-resolution image.
        tile_size (tuple): The (width, height) of the tiles to generate. This should
                           be a size that MASt3R can handle well.
        overlap_px (int): The number of pixels of overlap between adjacent tiles.
                          This is crucial to ensure features on tile borders are
                          captured in at least one full tile context.

    Returns:
        list: A list of dictionaries, where each dictionary represents a tile.
              It contains the tile's ID, its parent image, its pixel bounds
              within the parent image, and the image data as a NumPy array.
              Returns an empty list if the image cannot be opened.
    """
    try:
        # Open the high-resolution source image
        img = Image.open(image_path)
        img_w, img_h = img.size
    except Exception as e:
        print(f"ERROR: Could not open image {image_path}. Reason: {e}")
        return []

    tiles_manifest = []
    tile_w, tile_h = tile_size
    
    # The stride is the distance to move for the start of the next tile.
    # If tile_w is 1024 and overlap is 256, the next tile starts 768 pixels over.
    stride_w = tile_w - overlap_px
    stride_h = tile_h - overlap_px
    
    tile_id_counter = 0

    # Iterate over the image grid with the specified stride
    for y in range(0, img_h, stride_h):
        for x in range(0, img_w, stride_w):
            # Define the bounding box for cropping the tile from the source image.
            # The coordinates are (left, upper, right, lower).
            x_end = min(x + tile_w, img_w)
            y_end = min(y + tile_h, img_h)
            
            # Crop the tile from the main image
            tile_img = img.crop((x, y, x_end, y_end))

            # Discard tiles that are too small (e.g., thin slivers at the edges)
            # which won't be useful for feature matching.
            if tile_img.width < overlap_px or tile_img.height < overlap_px:
                continue

            # This dictionary holds all the critical info about the tile
            tile_info = {
                "tile_id": f"{os.path.basename(image_path)}_tile_{tile_id_counter:04d}",
                "parent_image_path": image_path,
                # CRITICAL: Store the tile's position relative to the original image.
                # Format is (x_min, y_min, x_max, y_max).
                "bounds_in_parent": (x, y, x_end, y_end),
                # The actual image data as a NumPy array, ready for the model.
                "tile_data": np.array(tile_img)
            }
            tiles_manifest.append(tile_info)
            tile_id_counter += 1
            
    return tiles_manifest

def _check_bbox_intersection(boxA, boxB):
    """Helper function to check if two bounding boxes intersect."""
    # box format: [x_min, y_min, x_max, y_max]
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    return x_right >= x_left and y_bottom >= y_top

def determine_overlapping_tile_pairs(tiles_A, tiles_B, transform_A_to_B):
    """
    Identifies pairs of tiles from two images that likely overlap based on a
    geometric transformation.

    Args:
        tiles_A (list): The tile manifest for the first image (the "source").
        tiles_B (list): The tile manifest for the second image (the "destination").
        transform_A_to_B (np.ndarray): A 2x3 affine transformation matrix that maps
                                      pixel coordinates from image A to image B.
                                      This must be pre-calculated from footprints.

    Returns:
        list: A list of tuples, where each tuple contains two corresponding
              tile dictionaries, e.g., [(tile_A1, tile_B3), (tile_A2, tile_B4), ...].
    """
    if transform_A_to_B is None:
        return []

    overlapping_pairs = []

    for tile_a in tiles_A:
        # Get the bounding box of tile A in its parent's coordinate system.
        x_min, y_min, x_max, y_max = tile_a["bounds_in_parent"]

        # Define the four corners of the bounding box to be transformed.
        # The shape must be (1, N, 2) for cv2.transform.
        corners_a = np.array([
            [[x_min, y_min]],
            [[x_max, y_min]],
            [[x_max, y_max]],
            [[x_min, y_max]]
        ], dtype=np.float32)

        # Project the corners of tile A's bounding box into image B's coordinate system.
        transformed_corners = cv2.transform(corners_a, transform_A_to_B)
        
        # Create a new bounding box in image B that encloses the projected shape.
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        projected_bbox_in_B = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]

        # Now, iterate through all tiles in image B and check for intersection.
        for tile_b in tiles_B:
            if _check_bbox_intersection(projected_bbox_in_B, tile_b["bounds_in_parent"]):
                overlapping_pairs.append((tile_a, tile_b))
                
    return overlapping_pairs
