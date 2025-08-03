#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo functions
# --------------------------------------------------------
import pycolmap
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import PIL.Image
import torch

from kapture.converter.colmap.database_extra import kapture_to_colmap, get_colmap_image_ids_from_db
from kapture.converter.colmap.database import COLMAPDatabase

from mast3r.colmap.mapping import kapture_import_image_folder_or_list, run_mast3r_matching, glomap_run_mapper
from mast3r.demo import set_scenegraph_options
from mast3r.retrieval.processor import Retriever
from mast3r.image_pairs import make_pairs

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl

# New imports for chunked processing
import os
import numpy as np
from PIL import Image
import json
from scipy.spatial import cKDTree
from tqdm import tqdm
import cv2
from dust3r.inference import inference


class GlomapRecon:
    def __init__(self, world_to_cam, intrinsics, points3d, imgs):
        self.world_to_cam = world_to_cam
        self.intrinsics = intrinsics
        self.points3d = points3d
        self.imgs = imgs


class GlomapReconState:
    def __init__(self, glomap_recon, should_delete=False, cache_dir=None, outfile_name=None):
        self.glomap_recon = glomap_recon
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')
    parser.add_argument('--glomap_bin', default='glomap', type=str, help='glomap bin')
    parser.add_argument('--retrieval_model', default=None, type=str, help="retrieval_model to be loaded")

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser

############################################################################################################
############################################################################################################
############################################################################################################
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


def run_chunked_pipeline(image_pairs_to_match, model, device, precomputed_transforms, tile_size=(1024, 1024), overlap_px=256):
    """
    The main orchestration function, now using pre-computed transforms.

    Args:
        image_pairs_to_match (list): A list of tuples, each containing the ABSOLUTE
                                     file paths for two images to be matched.
        model: The loaded MASt3R model.
        device: The PyTorch device.
        precomputed_transforms (dict): The dictionary loaded from your JSON file.
                                     Keys are tuples (basename1, basename2),
                                     values are numpy transform arrays.
        tile_size (tuple): The size of tiles to generate.
        overlap_px (int): The overlap between tiles.

    Returns:
        tuple: unique_kpts (dict), indexed_matches (dict)
    """
    
    # --- Part 1: Generate tiles for all unique images involved ---
    tile_cache = {}
    all_image_paths = np.unique([path for pair in image_pairs_to_match for path in pair])
    
    for image_path in all_image_paths:
        if image_path not in tile_cache:
            print(f"Generating tiles for {os.path.basename(image_path)}...")
            tile_cache[image_path] = generate_image_tiles(image_path, tile_size, overlap_px)

    # --- Part 2: For each image pair, find overlapping tiles and run inference ---
    all_aggregated_matches = {}
    
    for image_path1, image_path2 in image_pairs_to_match:
        print(f"\n--- Processing pair: {os.path.basename(image_path1)} <-> {os.path.basename(image_path2)} ---")
        
        # Get the basenames to use as keys for the transform dictionary
        name1_base = os.path.splitext(os.path.basename(image_path1))[0]
        name2_base = os.path.splitext(os.path.basename(image_path2))[0]

        # Get the pre-computed transform from the dictionary
        transform_1_to_2 = precomputed_transforms.get((name1_base, name2_base))
        
        if transform_1_to_2 is None:
            # Also check the reverse direction in case your JSON stores one-way transforms
            transform_2_to_1 = precomputed_transforms.get((name2_base, name1_base))
            if transform_2_to_1 is not None:
                # We need 1->2, so we must invert 2->1
                # For an affine matrix M = [A | t], the inverse is [A^-1 | -A^-1*t]
                A = transform_2_to_1[:, :2]
                t = transform_2_to_1[:, 2]
                A_inv = np.linalg.inv(A)
                t_inv = -A_inv @ t
                transform_1_to_2 = np.hstack((A_inv, t_inv.reshape(2, 1)))
            else:
                 print(f"Transform not found for pair ({name1_base}, {name2_base}). Skipping.")
                 continue

        tiles1 = tile_cache[image_path1]
        tiles2 = tile_cache[image_path2]
            
        # Determine which tile pairs to run the matcher on
        # This function is from your tile_manifest.py and can be reused directly
        overlapping_tile_pairs = determine_overlapping_tile_pairs(tiles1, tiles2, transform_1_to_2)
        print(f"Found {len(overlapping_tile_pairs)} overlapping tile pairs to match.")
        
        if not overlapping_tile_pairs:
            continue
            
        matches_for_this_pair = run_inference_on_tiles(overlapping_tile_pairs, model, device)
        all_aggregated_matches.update(matches_for_this_pair)

    if not all_aggregated_matches:
        print("No matches were found across any image pairs.")
        return {}, {}
        
    # --- Part 3: Deduplicate all keypoints and format for COLMAP ---
    print("\n--- Aggregation and Deduplication Stage ---")
    unique_kpts, indexed_matches = deduplicate_and_format_for_colmap(all_aggregated_matches)
    
    print("Chunked matching pipeline complete.")
    return unique_kpts, indexed_matches


def run_inference_on_tiles(tile_pairs, model, device, conf_threshold=0.95):
    """
    Runs MASt3R inference on tile pairs, re-projects keypoints to the full
    image coordinate system, and aggregates them.

    Args:
        tile_pairs (list): List of (tile_A, tile_B) tuples to match.
        model: The loaded MASt3R model.
        device: The device to run inference on ('cuda' or 'cpu').
        conf_threshold (float): Confidence threshold to filter weak matches.

    Returns:
        dict: A dictionary where keys are parent image path tuples (path_A, path_B)
              and values are dictionaries containing two concatenated numpy arrays:
              'keypoints0' and 'keypoints1', with all merged and re-projected matches.
    """
    # This dictionary will collect all re-projected matches, grouped by their
    # original parent image pair.
    # Structure: { ('path/A.tif', 'path/B.tif'): {'kpts0': [...], 'kpts1': [...]}, ... }
    aggregated_matches = {}

    batch_size = 4 # Adjust based on GPU memory
    
    for i in tqdm(range(0, len(tile_pairs), batch_size), desc="Matching Image Tiles"):
        batch_of_pairs = tile_pairs[i:i + batch_size]
        
        # Prepare the batch in the format MASt3R's inference function expects
        # (This part might need slight adjustment to match the exact 'inference' function signature)
        inference_input = []
        for tile_a, tile_b in batch_of_pairs:
            # Convert numpy array HxWxC to torch tensor CxHxW and send to device
            img_a_tensor = torch.from_numpy(tile_a['tile_data']).permute(2, 0, 1).float().to(device)
            img_b_tensor = torch.from_numpy(tile_b['tile_data']).permute(2, 0, 1).float().to(device)
            inference_input.append([{'img': img_a_tensor}, {'img': img_b_tensor}])

        # Run inference (This assumes an 'inference' function from the MASt3R repo)
        with torch.no_grad():
            # This is a hypothetical call; adapt to the actual function if it differs.
            output = inference(inference_input, model, device)

        # Process results for each pair in the completed batch
        for j, (tile_a, tile_b) in enumerate(batch_of_pairs):
            # Extract predictions for the j-th pair
            pred_kpts1 = output['pred_pts1'][j]
            pred_kpts2 = output['pred_pts2'][j]
            confidence = output['confidence'][j]

            # Filter matches by confidence
            confident_mask = confidence > conf_threshold
            kpts1_tile_local = pred_kpts1[confident_mask]
            kpts2_tile_local = pred_kpts2[confident_mask]

            if kpts1_tile_local.shape[0] == 0:
                continue

            # --- THE CRITICAL RE-PROJECTION STEP ---
            # Get the top-left corner (offset) of each tile within its parent image.
            offset_a_x, offset_a_y, _, _ = tile_a["bounds_in_parent"]
            offset_b_x, offset_b_y, _, _ = tile_b["bounds_in_parent"]

            # Add the offset to translate tile-local coordinates to full-image coordinates.
            offset_a = torch.tensor([offset_a_x, offset_a_y], device=device)
            offset_b = torch.tensor([offset_b_x, offset_b_y], device=device)
            kpts1_full_global = kpts1_tile_local + offset_a
            kpts2_full_global = kpts2_tile_local + offset_b

            # --- AGGREGATION ---
            parent_A = tile_a["parent_image_path"]
            parent_B = tile_b["parent_image_path"]
            pair_key = (parent_A, parent_B)

            if pair_key not in aggregated_matches:
                aggregated_matches[pair_key] = {"kpts0": [], "kpts1": []}
            
            # Append the re-projected keypoints (as numpy arrays) to the list for this pair.
            aggregated_matches[pair_key]["kpts0"].append(kpts1_full_global.cpu().numpy())
            aggregated_matches[pair_key]["kpts1"].append(kpts2_full_global.cpu().numpy())

    # --- FINAL MERGE ---
    # Concatenate all match arrays for each image pair into single numpy arrays.
    final_matches = {}
    for pair_key, kpt_data in aggregated_matches.items():
        if not kpt_data["kpts0"]: continue # Skip if no matches were found
        
        final_matches[pair_key] = {
            "keypoints0": np.concatenate(kpt_data["kpts0"], axis=0),
            "keypoints1": np.concatenate(kpt_data["kpts1"], axis=0)
        }
        
    return final_matches


def deduplicate_and_format_for_colmap(aggregated_matches, distance_threshold=2.0):
    """
    Takes the aggregated matches, deduplicates keypoints within each image,
    and formats the data for easy export to a COLMAP database.

    Args:
        aggregated_matches (dict): The output from `run_inference_on_tiles`.
        distance_threshold (float): The pixel distance in the full-resolution image
                                    below which keypoints are considered duplicates.

    Returns:
        tuple: A tuple containing:
            - unique_kpts_per_image (dict): {image_path: unique_keypoints_array}.
            - final_matches_indexed (dict): { (path_A, path_B): array_of_match_indices }.
    """
    unique_kpts_per_image = {}
    
    # --- Step 5a: Collect all detected keypoints for each image ---
    all_kpts_per_image = {}
    for (path_A, path_B), match_data in aggregated_matches.items():
        if path_A not in all_kpts_per_image: all_kpts_per_image[path_A] = []
        if path_B not in all_kpts_per_image: all_kpts_per_image[path_B] = []
        all_kpts_per_image[path_A].append(match_data["keypoints0"])
        all_kpts_per_image[path_B].append(match_data["keypoints1"])

    # --- Step 5b: Deduplicate keypoints for each image ---
    print("Deduplicating keypoints for each image...")
    for path, kpt_list in all_kpts_per_image.items():
        if not kpt_list:
            unique_kpts_per_image[path] = np.array([])
            continue
            
        all_kpts = np.concatenate(kpt_list, axis=0)
        
        # A simple and fast deduplication strategy: round coordinates to the
        # nearest pixel and find unique rows.
        # For higher accuracy, clustering algorithms could be used.
        _, unique_indices = np.unique(np.round(all_kpts), axis=0, return_index=True)
        unique_kpts_per_image[path] = all_kpts[unique_indices]

    # --- Step 5c: Re-map matches to use the new unique keypoint indices ---
    print("Re-indexing matches...")
    final_matches_indexed = {}
    for (path_A, path_B), match_data in aggregated_matches.items():
        kpts_A_unique = unique_kpts_per_image[path_A]
        kpts_B_unique = unique_kpts_per_image[path_B]

        if kpts_A_unique.shape[0] == 0 or kpts_B_unique.shape[0] == 0:
            continue

        # Use a KD-Tree for very fast nearest neighbor search. This finds which
        # 'unique' keypoint corresponds to each of our original 'raw' keypoints.
        tree_A = cKDTree(kpts_A_unique) 
        tree_B = cKDTree(kpts_B_unique)

        # Query the trees to find the index of the closest unique keypoint.
        dist_A, indices_A = tree_A.query(match_data["keypoints0"], distance_upper_bound=distance_threshold)
        dist_B, indices_B = tree_B.query(match_data["keypoints1"], distance_upper_bound=distance_threshold)

        # A match is valid only if both of its keypoints were close enough to a unique keypoint.
        # cKDTree returns an index equal to `n` (the number of points) if no neighbor is found.
        valid_mask = (indices_A < len(kpts_A_unique)) & (indices_B < len(kpts_B_unique))

        # The final result is an array of corresponding indices.
        # Shape is (NumMatches, 2), where each row is [index_in_A, index_in_B].
        final_matches_indexed[(path_A, path_B)] = np.stack([indices_A[valid_mask], indices_B[valid_mask]], axis=-1)

    return unique_kpts_per_image, final_matches_indexed


def export_chunked_matches_to_colmap(colmap_db, unique_kpts_per_image, final_matches_indexed, colmap_image_ids):
    """
    Exports the results of the chunked pipeline to the COLMAP database.

    Args:
        colmap_db: An open COLMAP database connection object.
        unique_kpts_per_image (dict): Output from `deduplicate_and_format_for_colmap`.
        final_matches_indexed (dict): Output from `deduplicate_and_format_for_colmap`.
        colmap_image_ids (dict): A mapping from {image_path: colmap_image_id}.
    """
    print("Exporting unique keypoints to COLMAP database...")
    # First, add all the unique keypoints for each image to the database.
    for image_path, keypoints in unique_kpts_per_image.items():
        if image_path in colmap_image_ids:
            colmap_id = colmap_image_ids[image_path]
            # COLMAP expects keypoints in (x, y, scale, orientation) format.
            # We provide dummy values for scale and orientation.
            # The key is that the number of rows is the number of features.
            colmap_db.add_keypoints(colmap_id, keypoints)
        else:
            print(f"Warning: {image_path} not found in COLMAP image_ids map.")

    print("Exporting indexed matches to COLMAP database...")
    # Second, add the feature matches using the indices of the unique keypoints.
    for (path1, path2), matches in final_matches_indexed.items():
        if path1 in colmap_image_ids and path2 in colmap_image_ids:
            id1 = colmap_image_ids[path1]
            id2 = colmap_image_ids[path2]
            
            # The `matches` array is already in the (NumMatches, 2) format
            # that COLMAP's add_matches requires.
            colmap_db.add_matches(id1, id2, matches)
            
    # Commit the changes to the database file
    colmap_db.commit()
    print("Successfully exported matches to COLMAP.")



def get_reconstructed_scene(glomap_bin, outdir, gradio_delete_cache, model, retrieval_model, device, silent, image_size,
                            current_scene_state, filelist, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0]]
    #Jawad: custom pairs for aerial images where we have a swin type scenegraph
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)

    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')

    root_path = os.path.commonpath(filelist)
    filelist_relpath = [
        os.path.relpath(filename, root_path).replace('\\', '/')
        for filename in filelist
    ]
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)
    image_pairs = [
        (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
        for img1, img2 in pairs
    ]

    colmap_db_path = os.path.join(cache_dir, 'colmap.db')
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)

    os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)
    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    
    # ##########################################################################################################################################
    # # +++ NEW BLOCK +++
    # ##########################################################################################################################################

    # # 1. Define cache directory and paths first. This logic is still needed.
    # if current_scene_state is not None and \
    #     not current_scene_state.should_delete and \
    #         current_scene_state.cache_dir is not None:
    #     cache_dir = current_scene_state.cache_dir
    # elif gradio_delete_cache:
    #     cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    # else:
    #     cache_dir = os.path.join(outdir, 'cache')

    # # 2. Directly compute relative paths from the input filelist.
    # root_path = os.path.commonpath(filelist)
    # filelist_relpath = [
    #     os.path.relpath(filename, root_path).replace('\\', '/')
    #     for filename in filelist
    # ]

    # # 3. Load the pre-computed transforms, which now define our pairs.
    # TRANSFORMS_JSON_PATH = "/mnt/d/projects/wsl_projects/Projects/3_repo/mast3r/data/tobias/1/fp/image_transforms.json"
    # print(f"Loading pre-computed transforms from {TRANSFORMS_JSON_PATH} to determine matching pairs...")
    # try:
    #     with open(TRANSFORMS_JSON_PATH, 'r') as f:
    #         transform_data_loaded = json.load(f)
    #         precomputed_transforms = {
    #             tuple(key.split('__')): np.array(value)
    #             for key, value in transform_data_loaded.items()
    #         }
    #     print(f"Successfully loaded {len(precomputed_transforms)} potential pairs.")
    # except FileNotFoundError:
    #     print(f"ERROR: Transform file not found at {TRANSFORMS_JSON_PATH}. Aborting.")
    #     exit(1)

    # # 4. Generate the `image_pairs` list directly from the keys of the transforms dictionary.
    # image_pairs = []

    # # Create a quick lookup map from basename -> relative path
    # basename_to_relpath_map = {
    #     os.path.splitext(os.path.basename(p))[0]: p 
    #     for p in filelist_relpath
    # }

    # for name1_base, name2_base in precomputed_transforms.keys():
    #     # Find the corresponding relative paths for the basenames in the transform key
    #     relpath1 = basename_to_relpath_map.get(name1_base)
    #     relpath2 = basename_to_relpath_map.get(name2_base)
        
    #     # Only add the pair if both images are part of the current run (in filelist_relpath)
    #     if relpath1 and relpath2:
    #         image_pairs.append((relpath1, relpath2))

    # if not image_pairs:
    #     raise Exception("No overlapping pairs found based on the transforms JSON and the input filelist.")

    # print(f"Generated {len(image_pairs)} pairs to be matched based on pre-computed overlaps.")

    # # 5. The Kapture and COLMAP DB setup remains the same.
    # kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)

    # colmap_db_path = os.path.join(cache_dir, 'colmap.db')
    # if os.path.isfile(colmap_db_path):
    #     os.remove(colmap_db_path)

    # os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)
    # colmap_db = COLMAPDatabase.connect(colmap_db_path) 
    # ##########################################################################################################################################
    # # +++ NEW BLOCK +++
    # ##########################################################################################################################################
    
    try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        # colmap_image_pairs = run_mast3r_matching(model, image_size, 16, device,
        #                                          kdata, root_path, image_pairs, colmap_db,
        #                                          False, 5, 1.001,
        #                                          False, 3)
        
        colmap_image_pairs = run_mast3r_matching(model=model, maxdim=image_size, patch_size=16, device=device,
                                                 kdata=kdata, root_path=root_path,
                                                 image_pairs_kapture=image_pairs, colmap_db=colmap_db,
                                                 dense_matching=True, pixel_tol=5, conf_thr=3,
                                                 skip_geometric_verification=False, min_len_track=3)
        

        # ##########################################################################################################################################
        # # +++ NEW BLOCK +++
        # ##########################################################################################################################################
        # # 1. Get the mapping from image names to the IDs COLMAP has assigned them.
        # colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)

        # # 2. Prepare the list of absolute image paths for matching.
        # #    `image_pairs` is a list of relative paths from earlier in the script.
        # image_pairs_abs = [
        #     (os.path.join(root_path, p1), os.path.join(root_path, p2))
        #     for p1, p2 in image_pairs
        # ]

        # # 3. Run the entire new pipeline, passing the loaded transforms.
        # unique_kpts, indexed_matches = run_chunked_pipeline(
        #     image_pairs_to_match=image_pairs_abs,
        #     model=model,
        #     device=device,
        #     precomputed_transforms=precomputed_transforms,  # Pass the loaded dictionary here
        #     tile_size=(1024, 1024),
        #     overlap_px=256
        # )

        # # 4. Export the final, clean results to the COLMAP database.
        # export_chunked_matches_to_colmap(colmap_db, unique_kpts, indexed_matches, colmap_image_ids)

        # ##########################################################################################################################################
        # # +++ NEW BLOCK +++
        # ##########################################################################################################################################

        colmap_db.close()

    except Exception as e:
        print(f'Error {e}')
        colmap_db.close()
        exit(1)
        
    if len(colmap_image_pairs) == 0:
        raise Exception("no matches were kept")

    # if len(indexed_matches) == 0:
    #     raise Exception("Chunked matching resulted in no matches.")

    # colmap db is now full, run colmap
    colmap_world_to_cam = {}
    print("verify_matches")
    f = open(cache_dir + '/pairs.txt', "w")
    for image_path1, image_path2 in colmap_image_pairs:
    # for image_path1, image_path2 in image_pairs:
        f.write("{} {}\n".format(image_path1, image_path2))
    f.close()
    pycolmap.verify_matches(colmap_db_path, cache_dir + '/pairs.txt') #Jawad: why do we verify matches here?

    reconstruction_path = os.path.join(cache_dir, "reconstruction")
    if os.path.isdir(reconstruction_path):
        shutil.rmtree(reconstruction_path)
    os.makedirs(reconstruction_path, exist_ok=True)
    glomap_run_mapper(glomap_bin, colmap_db_path, reconstruction_path, root_path)

    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    ouput_recon = pycolmap.Reconstruction(os.path.join(reconstruction_path, '0'))
    print(ouput_recon.summary())

    colmap_world_to_cam = {}
    colmap_intrinsics = {}
    colmap_image_id_to_name = {}
    images = {}
    num_reg_images = ouput_recon.num_reg_images()
    for idx, (colmap_imgid, colmap_image) in enumerate(ouput_recon.images.items()):
        colmap_image_id_to_name[colmap_imgid] = colmap_image.name
        if callable(colmap_image.cam_from_world):
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world().matrix(
            )
        else:
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix
        camera = ouput_recon.cameras[colmap_image.camera_id]
        K = np.eye(3)
        K[0, 0] = camera.focal_length_x
        K[1, 1] = camera.focal_length_y
        K[0, 2] = camera.principal_point_x
        K[1, 2] = camera.principal_point_y
        colmap_intrinsics[colmap_imgid] = K

        with PIL.Image.open(os.path.join(root_path, colmap_image.name)) as im:
            images[colmap_imgid] = np.asarray(im)

        if idx + 1 == num_reg_images:
            break  # bug with the iterable ?
    points3D = []
    num_points3D = ouput_recon.num_points3D()
    for idx, (pt3d_id, pts3d) in enumerate(ouput_recon.points3D.items()):
        points3D.append((pts3d.xyz, pts3d.color))
        if idx + 1 == num_points3D:
            break  # bug with the iterable ?
    scene = GlomapRecon(colmap_world_to_cam, colmap_intrinsics, points3D, images)
    scene_state = GlomapReconState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, transparent_cams, cam_size)
    return scene_state, outfile


def get_3D_model_from_scene(silent, scene_state, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    recon = scene_state.glomap_recon

    scene = trimesh.Scene()
    pts = np.stack([p[0] for p in recon.points3d], axis=0)
    col = np.stack([p[1] for p in recon.points3d], axis=0)
    pct = trimesh.PointCloud(pts, colors=col)
    scene.add_geometry(pct)

    # add each camera
    cams2world = []
    for i, (id, pose_w2c_3x4) in enumerate(recon.world_to_cam.items()):
        intrinsics = recon.intrinsics[id]
        focal = (intrinsics[0, 0] + intrinsics[1, 1]) / 2.0
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        pose_w2c = np.eye(4)
        pose_w2c[:3, :] = pose_w2c_3x4
        pose_c2w = np.linalg.inv(pose_w2c)
        cams2world.append(pose_c2w)
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else recon.imgs[id], focal,
                      imsize=recon.imgs[id].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    ## Commented out for not saving .glb file
    # if not silent:
    #     print('(exporting 3D scene to', outfile, ')')
    # scene.export(file_obj=outfile) 

    return outfile


def main_demo(glomap_bin, tmpdirname, model, retrieval_model, device, image_size, server_name, server_port,
              silent=False, share=False, gradio_delete_cache=False):
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, glomap_bin, tmpdirname, gradio_delete_cache, model,
                                  retrieval_model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    available_scenegraph_type = [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")]
    if retrieval_model is not None:
        available_scenegraph_type.insert(1, ("retrieval: connect views based on similarity", "retrieval"))

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
        else:
            return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                    info="Only optimize one set of intrinsics for all views")
                scenegraph_type = gradio.Dropdown(available_scenegraph_type,
                                                  value='complete', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                with gradio.Column(visible=False) as win_col:
                    winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                            minimum=1, maximum=1, step=1)
                    win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                      minimum=0, maximum=0, step=1, visible=False)
            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.01, minimum=0.001, maximum=1.0, step=0.001)
            with gradio.Row():
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[win_col, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[scene, inputfiles, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, shared_intrinsics],
                          outputs=[scene, outmodel])
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, transparent_cams, cam_size],
                            outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, transparent_cams, cam_size],
                                    outputs=outmodel)
    demo.launch(share=share, server_name=server_name, server_port=server_port)
