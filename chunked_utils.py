# --- Imports needed from the original mast3r codebase or common libraries ---
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm
from PIL import Image
import json
import cv2
from torchvision import transforms
from collections import defaultdict
from itertools import combinations # We need this for creating pairs

# You will need these helper functions from the original mast3r repository's utils.
# Make sure they are available in your Python path.
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid, geotrf  # noqa
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

class DisjointSet:
    """
    A simple Disjoint Set Union (DSU) or Union-Find data structure.
    Used to efficiently track and merge sets of connected components,
    which in our case are the feature tracks.
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, i):
        """Finds the root representative of the set containing element i."""
        if self.parent.get(i) is None:
            # This element is new; it becomes its own parent.
            self.parent[i] = i
            self.rank[i] = 0
        
        # Path compression for efficiency
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Merges the sets containing elements i and j."""
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by rank for a more balanced tree
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1

    def get_all_subsets(self):
        """Returns all the disjoint sets as a list of lists."""
        subsets = defaultdict(list)
        for item in self.parent:
            root = self.find(item)
            subsets[root].append(item)
        return list(subsets.values())


def _cluster_keypoints(aggregated_matches):
    """
    Step 1: Takes raw matches and clusters keypoints within each image to find
    unique potential track representatives.
    
    CORRECTED to use image paths as keys.
    """
    print("Step 1: Clustering keypoints...")
    all_kpts_per_image = defaultdict(list)
    
    # Iterate through the matches and append keypoints to the correct image path list
    for (path_A, path_B), match_data in aggregated_matches.items():
        all_kpts_per_image[path_A].append(match_data["kpts0"])
        all_kpts_per_image[path_B].append(match_data["kpts1"])

    unique_kpts_per_image = {}
    for path, kpt_list in all_kpts_per_image.items():
        if not kpt_list: continue
        all_kpts = np.concatenate(kpt_list, axis=0)
        _, unique_indices = np.unique(np.round(all_kpts), axis=0, return_index=True)
        unique_kpts_per_image[path] = all_kpts[unique_indices]
        
    return unique_kpts_per_image

def _build_and_merge_tracks(aggregated_matches, unique_kpts_per_image, distance_threshold):
    """
    Step 2: Builds tracks from pairwise matches and merges them using a DisjointSet.
    """
    print("Step 2: Building and merging tracks...")
    dsu = DisjointSet()
    
    # Pre-build k-d trees for fast lookups
    trees = {path: cKDTree(kpts) for path, kpts in unique_kpts_per_image.items()}

    for (path_A, path_B), match_data in aggregated_matches.items():
        if path_A not in trees or path_B not in trees: continue
            
        # Find the unique index for each raw keypoint in the match
        _, indices_A = trees[path_A].query(match_data["kpts0"], distance_upper_bound=distance_threshold)
        _, indices_B = trees[path_B].query(match_data["kpts1"], distance_upper_bound=distance_threshold)

        # For each valid match, union the corresponding unique keypoints into a track
        for i in range(len(indices_A)):
            idx_A, idx_B = indices_A[i], indices_B[i]
            
            # Check if both points were found in the tree
            if idx_A < len(unique_kpts_per_image[path_A]) and idx_B < len(unique_kpts_per_image[path_B]):
                # A "point" is uniquely identified by its image path and its index within that image
                point_A = (path_A, idx_A)
                point_B = (path_B, idx_B)
                dsu.union(point_A, point_B)
                
    return dsu.get_all_subsets()


def _filter_and_finalize(all_tracks, unique_kpts_per_image, min_len_track):
    """
    FINAL, ROBUST, AND EFFICIENT VERSION.

    Step 3: Filters tracks by length and uses a "unique representative" strategy
    to build the final keypoints and matches, avoiding combinatorial explosion.

    Args:
        all_tracks (list): The list of merged tracks from the DisjointSet.
        unique_kpts_per_image (dict): The dictionary of all unique keypoint candidates.
        min_len_track (int): The minimum number of images a point must appear in.

    Returns:
        tuple: (final_keypoints, final_matches_indexed)
    """
    print("Step 3: Filtering tracks and finalizing with unique representative strategy...")
    
    # --- Data Structures for Final Output ---
    final_keypoints = defaultdict(list)
    final_matches_indexed = defaultdict(list)
    point_to_new_idx = {}
    
    # --- The Single, Efficient Pass Over All Tracks ---
    for track in tqdm(all_tracks, desc="Processing Tracks", total=len(all_tracks), unit="tracks"):
        #
        # --- Stage 1: Filter Tracks and Find Unique Representatives ---
        #
        # A track's semantic length is the number of unique images it spans.
        track_images = {path for path, idx in track}
        
        # If the track is too short, we discard it immediately.
        if len(track_images) < min_len_track:
            continue
            
        #
        # --- Stage 2: Create a "Clean" Representation of the Valid Track ---
        #
        # This is the core of the new logic. We create a dictionary that maps
        # each image path to just ONE representative point from that track,
        # even if the track contained hundreds of observations from that image.
        #
        representative_points = {}
        for path, old_idx in track:
            if path not in representative_points:
                # Store the first observation we see for this image as its representative.
                representative_points[path] = (path, old_idx)
        
        # `representative_points.values()` is now a clean list, e.g.,
        # [ (path_A, idx_A), (path_B, idx_B), (path_C, idx_C) ]
        # Its length is exactly the semantic track length.
        clean_track = list(representative_points.values())

        #
        # --- Stage 3: Process the Clean Track ---
        #
        
        # First, ensure all points in this clean track are assigned a new, final index.
        for path, old_idx in clean_track:
            point_id = (path, old_idx)
            
            if point_id not in point_to_new_idx:
                # This is the first time we've registered this representative point as valid.
                new_idx = len(final_keypoints[path])
                point_to_new_idx[point_id] = new_idx
                kpt_coords = unique_kpts_per_image[path][old_idx]
                final_keypoints[path].append(kpt_coords)

        # Second, create the pairwise matches from this clean, non-redundant track.
        # This `combinations` call is now very cheap, operating on a list of at most N items,
        # where N is the number of images.
        for point1, point2 in combinations(clean_track, 2):
            # We already know path1 != path2 because `representative_points` has unique path keys.
            path1, _ = point1
            path2, _ = point2

            # Look up the new, final indices that we just assigned.
            new_idx1 = point_to_new_idx[point1]
            new_idx2 = point_to_new_idx[point2]
            
            # Use a canonical pair ordering (path1 < path2) to group all matches correctly.
            if path1 > path2:
                path1, path2 = path2, path1
                new_idx1, new_idx2 = new_idx2, new_idx1
            
            final_matches_indexed[(path1, path2)].append([new_idx1, new_idx2])
    
    # --- Final Conversion to NumPy Arrays ---
    final_keypoints_np = {path: np.array(kpts) for path, kpts in final_keypoints.items()}
    final_matches_indexed_np = {pair: np.array(matches) for pair, matches in final_matches_indexed.items()}

    return final_keypoints_np, final_matches_indexed_np

  
def extract_matches_from_tile_prediction(pred1, pred2, shape1, shape2, conf_thr=3.0, device='cuda'):
    """
    CORRECTED VERSION.
    This function takes the raw output of the DUSt3R model for a SINGLE tile pair
    and extracts the 2D-2D point correspondences in their local tile coordinates.
    It uses the 'pts3d' and 'conf' keys.

    Args:
        pred1 (dict): The model's prediction for the first tile.
        pred2 (dict): The model's prediction for the second tile.
        shape1 (torch.Size): The (H, W) shape of the first tile tensor.
        shape2 (torch.Size): The (H, W) shape of the second tile tensor.
        conf_thr (float): The confidence threshold to filter keypoints.
        device (str): The device to use for calculations.

    Returns:
        tuple: A tuple containing (matches_im0, matches_im1), numpy arrays of
               shape (N, 2) with keypoint coordinates local to their tile.
    """
    # This logic is for dense matching using 3D point clouds, adapted from get_im_matches
    pts3d_list_raw = [pred1['pts3d'], pred2['pts3d_in_other_view']]
    confidences = [pred1['conf'], pred2['conf']]
    shapes = [shape1, shape2]

    # Create confidence masks
    confidence_masks = [conf >= conf_thr for conf in confidences]

    # Find 2D-2D matches between the two images by matching their 3D point clouds
    pts2d_list, pts3d_list = [], []
    for j in range(2):
        # Flatten the confidence mask
        conf_j = confidence_masks[j].cpu().numpy().flatten()
        
        # Get the H, W shape for the current view
        true_shape_j = shapes[j]
        
        # Create a grid of all possible 2D coordinates
        pts2d_j = xy_grid(true_shape_j[1], true_shape_j[0]).reshape(-1, 2)
        
        # Filter the 2D points and 3D points based on confidence
        pts2d_j_filtered = pts2d_j[conf_j]
        pts3d_j_filtered = pts3d_list_raw[j].detach().cpu().numpy().reshape(-1, 3)[conf_j]
        
        pts2d_list.append(pts2d_j_filtered)
        pts3d_list.append(pts3d_j_filtered)

    # Use the filtered 3D points to find matches
    PQ, PM = pts3d_list[0], pts3d_list[1]
    if len(PQ) == 0 or len(PM) == 0:
        return None, None
        
    # Find reciprocal nearest neighbors in the 3D point clouds
    reciprocal_in_PM, nnM_in_PQ, num_matches = find_reciprocal_matches(PQ, PM)

    if num_matches == 0:
        return None, None

    # Use the indices from the 3D match to get the final 2D coordinates
    matches_im1 = pts2d_list[1][reciprocal_in_PM]
    matches_im0 = pts2d_list[0][nnM_in_PQ][reciprocal_in_PM]

    return matches_im0, matches_im1


def deduplicate_and_format_for_colmap(
    aggregated_matches,
    min_len_track=3,
    distance_threshold=2.0
):
    """
    The main orchestration function. It uses helper functions to perform a
    robust, multi-step process of track building, filtering, and finalization.
    """
    if not aggregated_matches:
        return {}, {}

    # Step 1: Find all unique keypoint candidates by clustering raw points.
    unique_kpts_per_image = _cluster_keypoints(aggregated_matches)

    # Step 2: Build pairwise connections and merge them into complete tracks.
    all_tracks = _build_and_merge_tracks(aggregated_matches, unique_kpts_per_image, distance_threshold)

    # Step 3: Filter out short tracks and re-index the surviving keypoints and matches.
    final_keypoints, final_matches = _filter_and_finalize(
        all_tracks, unique_kpts_per_image, min_len_track
    )

    print(f"Finalized matching with {len(final_keypoints)} images containing valid tracks.")
    return final_keypoints, final_matches


def run_chunked_mast3r_matching(
    model, device, image_pairs_to_match, root_path, colmap_db,
    precomputed_transforms,
    tile_size=(512, 512), overlap_px=128, conf_thr=3.0):
    """
    The COMPLETE REPLACEMENT for the original `run_mast3r_matching` function.
    It orchestrates the tiling, matching, aggregation, and database export.
    """
    
    # --- Part A: Tiling and Tile-Pair Determination ---
    tile_cache = {}
    all_image_paths_abs = np.unique([p for pair in image_pairs_to_match for p in pair])
    
    for image_path in tqdm(all_image_paths_abs, total=len(all_image_paths_abs), desc="Preparing tiles"):
        if image_path not in tile_cache:
            tile_cache[image_path] = generate_image_tiles(image_path, tile_size, overlap_px, save_tile_data=True)

    # --- Part B: Inference on Tile Pairs and Match Aggregation ---
    aggregated_matches_temp = {} # Use a temporary dictionary
    batch_size = 2 # Adjust based on GPU memory

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
        for i in tqdm(range(0, len(overlapping_tile_pairs), batch_size), desc="Matching Tile Pairs", total=len(overlapping_tile_pairs)):
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
                
                #
                # --- THIS IS THE UPDATED FUNCTION CALL ---
                # We now pass the tensor shapes to the extraction function.
                #
                # We also need the tensor that was created *before* the unsqueeze(0)
                img_a_tensor = ImgNorm(tile_a['tile_data'])
                img_b_tensor = ImgNorm(tile_b['tile_data'])

                kpts1_local, kpts2_local = extract_matches_from_tile_prediction(
                    pred1_single,
                    pred2_single,
                    img_a_tensor.shape[-2:],  # Pass the (H, W) shape
                    img_b_tensor.shape[-2:],  # Pass the (H, W) shape
                    conf_thr,
                    device
                )

                if kpts1_local is None: continue

                # The rest of the re-projection logic remains the same
                offset_a = tile_a["bounds_in_parent"][:2]
                offset_b = tile_b["bounds_in_parent"][:2]
                kpts1_global = kpts1_local + offset_a
                kpts2_global = kpts2_local + offset_b
                
                pair_key = (image_path1, image_path2)
                if pair_key not in aggregated_matches_temp:
                    aggregated_matches_temp[pair_key] = {"kpts0": [], "kpts1": []}
                
                aggregated_matches_temp[pair_key]["kpts0"].append(kpts1_global)
                aggregated_matches_temp[pair_key]["kpts1"].append(kpts2_global)
    
    # --- START: NEW FINALIZATION STEP (THE FIX) ---
    print("Finalizing match aggregation...")
    aggregated_matches_final = {}
    for pair_key, kpt_data in aggregated_matches_temp.items():
        # Check if any matches were actually found for this pair
        if not kpt_data["kpts0"]:
            continue
        
        # Concatenate the list of arrays into a single large array for each key
        final_kpts0 = np.concatenate(kpt_data["kpts0"], axis=0)
        final_kpts1 = np.concatenate(kpt_data["kpts1"], axis=0)
        
        aggregated_matches_final[pair_key] = {
            "kpts0": final_kpts0,
            "kpts1": final_kpts1
        }
        
    # save the aggregated_matches_final for debugging or further processing
    torch.save(aggregated_matches_final, os.path.join(root_path, "aggregated_matches_final.pt"))
    # --- END: NEW FINALIZATION STEP (THE FIX) ---
    
    # --- Part C: Deduplicate and Export ---
    unique_kpts, indexed_matches = deduplicate_and_format_for_colmap(aggregated_matches_final, min_len_track=2)
    
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

    #
    # --- START: THE FIX FOR UNIQUE CONSTRAINT ---
    #
    # Export indexed matches with a check to prevent duplicate pair entries.
    pairs_added_to_db = set()

    for (path1, path2), matches in indexed_matches.items():
        rel_path1 = os.path.relpath(path1, root_path).replace('\\', '/')
        rel_path2 = os.path.relpath(path2, root_path).replace('\\', '/')
        
        id1, id2 = colmap_image_ids[rel_path1], colmap_image_ids[rel_path2]

        # Create a canonical pair tuple (min_id, max_id)
        canonical_pair = tuple(sorted((id1, id2)))

        # If we have already added matches for this canonical pair, skip.
        if canonical_pair in pairs_added_to_db:
            continue

        # Ensure correct ordering for COLMAP (id1 < id2) before adding
        if id1 > id2:
            # We don't need to swap ids here because the database function handles it.
            # We only need to swap the match columns to maintain correctness.
            matches = matches[:, ::-1]

        # Add the matches to the database
        colmap_db.add_matches(id1, id2, matches)
        
        # Record that we have processed this canonical pair
        pairs_added_to_db.add(canonical_pair)
    
    # --- END: THE FIX FOR UNIQUE CONSTRAINT ---
    
    colmap_db.commit()

    return indexed_matches

# It's good practice to disable the DecompressionBomb check for large images
Image.MAX_IMAGE_PIXELS = None

def generate_image_tiles(image_path, tile_size=(512, 512), overlap_px=128, save_tile_data=False):
    """
    MODIFIED VERSION (Edge-Aligned Tiling).
    Divides a large image into smaller, overlapping tiles. Instead of padding
    edge tiles, this version ensures the last tile in each row/column is aligned
    to the image edge, taking a full-sized tile. This may result in a larger
    overlap for the last tile compared to others.

    Args:
        image_path (str): The file path to the high-resolution image.
        tile_size (tuple): The (width, height) of the tiles to generate. This
                           dimension MUST be divisible by the model's patch size.
        overlap_px (int): The *minimum* number of pixels of overlap between adjacent tiles.

    Returns:
        list: A list of dictionaries for each tile, guaranteed to have uniform size.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_w, img_h = img.size
    except Exception as e:
        print(f"ERROR: Could not open image {image_path}. Reason: {e}")
        return []

    tiles_manifest = []
    target_w, target_h = tile_size
    
    # Check if the image is smaller than a single tile
    if img_w <= target_w or img_h <= target_h:
        # If so, handle with the previous padding method as a fallback
        final_tile = Image.new('RGB', tile_size, (0, 0, 0))
        final_tile.paste(img, (0, 0))
        tiles_manifest.append({
            "tile_id": f"{os.path.basename(image_path)}_tile_0000",
            "parent_image_path": image_path,
            "bounds_in_parent": (0, 0, img_w, img_h),
            "tile_data": np.array(final_tile)
        })
        return tiles_manifest

    # The stride is the distance to move for the start of the next tile.
    stride_w = target_w - overlap_px
    stride_h = target_h - overlap_px
    
    # --- START: NEW EDGE-ALIGNED COORDINATE GENERATION ---
    
    # Generate the list of starting y-coordinates
    # We create all the "normal" starting points, and then add the final
    # starting point that ensures the tile's bottom edge aligns with the image's bottom.
    y_starts = list(range(0, img_h - target_h, stride_h)) + [img_h - target_h]
    
    # Generate the list of starting x-coordinates, using the same logic
    x_starts = list(range(0, img_w - target_w, stride_w)) + [img_w - target_w]

    # Use np.unique to prevent duplicate tiles if the image size is a perfect fit
    y_starts = np.unique(y_starts).tolist()
    x_starts = np.unique(x_starts).tolist()
    
    # --- END: NEW EDGE-ALIGNED COORDINATE GENERATION ---

    tile_id_counter = 0
    # The loops now iterate over the pre-calculated starting points
    for y in y_starts:
        for x in x_starts:
            # The crop box is now guaranteed to be the target size and within bounds.
            bbox = (x, y, x + target_w, y + target_h)
            tile_img = img.crop(bbox)

            tile_info = {
                "tile_id": f"{os.path.basename(image_path)}_tile_{tile_id_counter:04d}",
                "parent_image_path": image_path,
                # The bounds are the exact crop coordinates.
                "bounds_in_parent": bbox,
                # The tile_data is guaranteed to be the target size.
                "tile_data": np.array(tile_img)
            }
            tiles_manifest.append(tile_info)
            tile_id_counter += 1
            
    # # Save json in case we want to use it later            
    # # Prepare a lightweight version for JSON
    # if save_tile_data:
    #     tiles_manifest_json = [
    #         {
    #             k: v for k, v in tile.items() if k != "tile_data"
    #         }
    #         for tile in tiles_manifest
    #     ]

    #     # Now safe to dump
    #     json.dump(tiles_manifest_json, open(f"{image_path}_tiles_manifest.json", 'w'), indent=4)
                        
    return tiles_manifest


def _check_bbox_intersection(boxA, boxB):
    """Helper function to check if two bounding boxes intersect."""
    # box format: [x_min, y_min, x_max, y_max]
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    return x_right >= x_left and y_bottom >= y_top

def _calculate_intersection_area(boxA, boxB):
    """
    NEW HELPER FUNCTION.
    Calculates the area of intersection of two bounding boxes.

    Args:
        boxA (list): Bounding box [x_min, y_min, x_max, y_max].
        boxB (list): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
        float: The area of the intersection. Returns 0 if they do not intersect.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    # If the boxes do not overlap, the area is 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area is the product of the intersection's width and height
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area


def determine_overlapping_tile_pairs(tiles_A, tiles_B, transform_A_to_B):
    """
    MODIFIED VERSION (One-to-One Best Match).
    Identifies the single best-matching tile in Image B for each tile in Image A,
    based on the largest intersection area.

    Args:
        tiles_A (list): The tile manifest for the source image A.
        tiles_B (list): The tile manifest for the destination image B.
        transform_A_to_B (np.ndarray): The 2x3 affine transformation matrix.

    Returns:
        list: A list of one-to-one tile pair tuples, e.g., [(tile_A1, tile_B3), ...].
    """
    if transform_A_to_B is None or not tiles_A or not tiles_B:
        return []

    overlapping_pairs = []

    # Iterate through each tile in the source image A
    for tile_a in tiles_A:
        # --- Find the single best match in B for this specific tile_a ---
        best_match_tile_b = None
        max_area = 0.0

        # Project tile_a's bounding box into image B's coordinate system
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

        # Now, iterate through all candidate tiles in image B to find the best one
        for tile_b in tiles_B:
            # Calculate the intersection area between the projected box and the candidate tile
            area = _calculate_intersection_area(projected_bbox_in_B, tile_b["bounds_in_parent"])

            # If this tile is a better match, update our records
            if area > max_area:
                max_area = area
                best_match_tile_b = tile_b
        
        # After checking all tiles in B, if we found a valid match for tile_a, add it to our list.
        # We use a small threshold to ensure there's a meaningful overlap.
        if best_match_tile_b is not None and max_area > 1:
            overlapping_pairs.append((tile_a, best_match_tile_b))
                
    return overlapping_pairs
