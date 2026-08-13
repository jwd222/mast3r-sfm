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
from itertools import combinations  # needed for creating pairwise matches within a track

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
    An enhanced Disjoint Set Union (DSU) data structure.
    It tracks not only the sets of points but also the set of unique image paths
    within each track, preventing invalid merges.
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}
        # This new dictionary is the key: maps a track's root to the set of image paths it contains.
        self.image_sets = {}

    def find(self, i):
        """Finds the root representative of the set containing element i."""
        if i not in self.parent:
            # Initialize a new element. It is its own parent.
            self.parent[i] = i
            self.rank[i] = 0
            # A new track initially contains only the image of its first point.
            # i[0] is the path from the tuple (path, index).
            self.image_sets[i] = {i[0]}
        
        # Path compression for efficiency
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """
        Merges the sets containing i and j, ONLY IF the merge is valid
        (i.e., they do not share any common image paths).

        Known trade-off (audit M6): enforcing "one keypoint per image per track"
        transitively can reject a merge that would otherwise have been fine, which
        fragments tracks. The shorter resulting tracks may then fall below
        ``min_len_track`` and be dropped, thinning the SfM graph. A nearest-
        representative selection strategy could recover some of these merges, but
        that requires per-dataset benchmarking; the conservative reject behavior is
        kept here for correctness.
        """
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # --- This is the GUARD CONDITION ---
            # Check if the sets of images are disjoint before merging.
            if self.image_sets[root_i].isdisjoint(self.image_sets[root_j]):
                # The merge is valid. Proceed with standard union by rank.
                if self.rank[root_i] > self.rank[root_j]:
                    self.parent[root_j] = root_i
                    # Merge image sets
                    self.image_sets[root_i].update(self.image_sets[root_j])
                else:
                    self.parent[root_i] = root_j
                    # Merge image sets
                    self.image_sets[root_j].update(self.image_sets[root_i])
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
    for path, kpt_list in tqdm(all_kpts_per_image.items(), total=len(all_kpts_per_image), desc="Generating unique keypoints", leave=False):
        if not kpt_list: continue
        all_kpts = np.concatenate(kpt_list, axis=0)
        _, unique_indices = np.unique(all_kpts, axis=0, return_index=True)
        unique_kpts_per_image[path] = all_kpts[unique_indices]
        
    return unique_kpts_per_image


def _build_and_merge_tracks(aggregated_matches, unique_kpts_per_image, distance_threshold):
    """
    Step 2: Builds tracks from pairwise matches.
    This version uses the enhanced DisjointSet to intelligently and correctly
    merge tracks, preventing tracks from containing multiple points from the same image.
    """
    print("Step 2: Building and merging tracks with invariant enforcement...")
    dsu = DisjointSet()
    trees = {path: cKDTree(kpts) for path, kpts in unique_kpts_per_image.items()}

    # --- Pass 1: Collect all valid connections (same as before) ---
    all_connections = set()
    print("  - Pass 1/2: Collecting all pairwise connections...")
    for (path_A, path_B), match_data in tqdm(aggregated_matches.items(), desc="Collecting Edges", leave=False):
        if path_A not in trees or path_B not in trees: continue
        _, indices_A = trees[path_A].query(match_data["kpts0"], distance_upper_bound=distance_threshold)
        _, indices_B = trees[path_B].query(match_data["kpts1"], distance_upper_bound=distance_threshold)

        for i in range(len(indices_A)):
            idx_A, idx_B = indices_A[i], indices_B[i]
            if idx_A < len(unique_kpts_per_image[path_A]) and idx_B < len(unique_kpts_per_image[path_B]):
                point_A = (path_A, idx_A)
                point_B = (path_B, idx_B)
                connection = tuple(sorted((point_A, point_B)))
                all_connections.add(connection)

    # --- Pass 2: Build tracks using the smarter DSU ---
    print(f"  - Pass 2/2: Merging {len(all_connections)} connections into tracks...")
    for point_A, point_B in tqdm(all_connections, desc="Building Tracks", leave=False):
        # The new `union` method now contains the critical safety check.
        dsu.union(point_A, point_B)

    return dsu.get_all_subsets()


def _build_and_merge_tracks_tmp(aggregated_matches, unique_kpts_per_image, distance_threshold=3):
    """
    Step 2: Builds tracks from pairwise matches and merges them using a DisjointSet.
    """
    print("Step 2: Building and merging tracks...")
    dsu = DisjointSet()
    
    # Pre-build k-d trees for fast lookups
    trees = {path: cKDTree(kpts) for path, kpts in unique_kpts_per_image.items()}

    for (path_A, path_B), match_data in tqdm(aggregated_matches.items(), total=len(aggregated_matches), desc="Building and merging tracks", leave=False):
        if path_A not in trees or path_B not in trees: continue
            
        # Find the unique index for each raw keypoint in the match
        _, indices_A = trees[path_A].query(match_data["kpts0"], distance_upper_bound=distance_threshold)
        _, indices_B = trees[path_B].query(match_data["kpts1"], distance_upper_bound=distance_threshold)

        # For each valid match, union the corresponding unique keypoints into a track
        for i in tqdm(range(len(indices_A)), desc="Unioning tracks", leave=False):
            idx_A, idx_B = indices_A[i], indices_B[i]
            
            # Check if both points were found in the tree
            if idx_A < len(unique_kpts_per_image[path_A]) and idx_B < len(unique_kpts_per_image[path_B]):
                # A "point" is uniquely identified by its image path and its index within that image
                point_A = (path_A, idx_A)
                point_B = (path_B, idx_B)
                dsu.union(point_A, point_B)
                
    return dsu.get_all_subsets()


def _filter_and_finalize_tmp(all_tracks, unique_kpts_per_image, min_len_track):
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
    for track in tqdm(all_tracks, desc="Finalizing Tracks", total=len(all_tracks), unit="tracks", leave=False):
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


def _filter_and_finalize(all_tracks, unique_kpts_per_image, min_len_track):
    """
    SIMPLIFIED AND FASTER VERSION.
    Since tracks from the new builder are guaranteed to be clean (one point per image),
    this function just needs to filter by length and format the output.
    """
    print("Step 3: Filtering clean tracks and finalizing...")
    
    final_keypoints = defaultdict(list)
    final_matches_indexed = defaultdict(list)
    point_to_new_idx = {}
    
    for track in tqdm(all_tracks, desc="Finalizing Tracks", leave=False):
        # The track length is now simply the number of points in the list.
        if len(track) < min_len_track:
            continue
            
        # Since the track is clean, every point in it is valid and needs a new index.
        for path, old_idx in track:
            point_id = (path, old_idx)
            if point_id not in point_to_new_idx:
                new_idx = len(final_keypoints[path])
                point_to_new_idx[point_id] = new_idx
                kpt_coords = unique_kpts_per_image[path][old_idx]
                final_keypoints[path].append(kpt_coords)

        # The combinations call is now guaranteed to be cheap and correct.
        for point1, point2 in combinations(track, 2):
            path1, _ = point1
            path2, _ = point2
            new_idx1 = point_to_new_idx[point1]
            new_idx2 = point_to_new_idx[point2]
            
            if path1 > path2:
                path1, path2 = path2, path1
                new_idx1, new_idx2 = new_idx2, new_idx1
            
            final_matches_indexed[(path1, path2)].append([new_idx1, new_idx2])
    
    final_keypoints_np = {path: np.array(kpts) for path, kpts in final_keypoints.items()}
    final_matches_indexed_np = {pair: np.array(matches) for pair, matches in final_matches_indexed.items()}

    return final_keypoints_np, final_matches_indexed_np


def extract_matches_from_tile_prediction(pred1, pred2, shape1, shape2, conf_thr=1.5, device='cuda',
                                         kpt_stride=8):
    """
    Extracts 2D-2D point correspondences (in local tile coordinates) from the raw
    DUSt3R/MASt3R prediction for a SINGLE tile pair, using the dense 3D-point-map
    path (``pts3d`` + ``conf``).

    Design note (audit H3/H4): the chunked matcher intentionally uses the dense
    3D-point matching path rather than MASt3R's learned descriptor head
    (``desc``/``desc_conf``). ``conf_thr`` is therefore applied to the raw pointmap
    ``conf`` (whose scale differs from descriptor confidence), NOT to ``desc_conf``.
    The previous default of 3.0 was tuned for ``desc_conf`` and was far too strict
    for raw ``conf``; 1.5 is a more appropriate default for the 3D-point path but
    should be validated per dataset (exposed as a CLI flag by the orchestrator).

    Args:
        pred1 (dict): The model's prediction for the first tile.
        pred2 (dict): The model's prediction for the second tile.
        shape1 (torch.Size): The (H, W) shape of the first tile tensor.
        shape2 (torch.Size): The (H, W) shape of the second tile tensor.
        conf_thr (float): Confidence threshold applied to raw pointmap ``conf``.
        device (str): The device to use for calculations.

    Returns:
        tuple: A tuple containing (matches_im0, matches_im1), numpy arrays of
               shape (N, 2) with keypoint coordinates local to their tile, or
               (None, None) if no confident reciprocal matches are found.
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

    # Spatially sparsify dense matches for SfM tie points (audit H3 / perf):
    # keep one correspondence per kpt_stride x kpt_stride pixel cell (keyed on the
    # image-0 position). SfM only needs well-distributed tie points, so this
    # preserves coverage and exact positions while cutting the dense match count
    # ~stride**2 x, which prevents the multi-million-connection blowup in track
    # building. Set kpt_stride<=1 to keep all dense matches.
    if kpt_stride and kpt_stride > 1 and len(matches_im0) > 0:
        cells = np.floor(np.asarray(matches_im0) / float(kpt_stride)).astype(np.int64)
        _, keep_idx = np.unique(cells, axis=0, return_index=True)
        keep_idx.sort()
        matches_im0 = matches_im0[keep_idx]
        matches_im1 = matches_im1[keep_idx]

    return matches_im0, matches_im1


def deduplicate_and_format_for_colmap(
    aggregated_matches,
    min_len_track=2,
    distance_threshold=2.0
):
    """
    The main orchestration function. It uses helper functions to perform a
    robust, multi-step process of track building, filtering, and finalization.

    ``min_len_track`` defaults to 2 so two-view correspondences survive for
    narrow aerial strips; this must stay consistent with the mapper's
    ``ignore_two_view_tracks=False`` (audit H2). Exposed as a CLI flag by the
    orchestrator.
    """
    if not aggregated_matches:
        return {}, {}

    # Step 1: Find all unique keypoint candidates by clustering raw points.
    unique_kpts_per_image = _cluster_keypoints(aggregated_matches)

    # Step 2: Build pairwise connections and merge them into complete tracks.
    all_tracks = _build_and_merge_tracks(aggregated_matches, unique_kpts_per_image, distance_threshold=distance_threshold)

    # Step 3: Filter out short tracks and re-index the surviving keypoints and matches.
    final_keypoints, final_matches = _filter_and_finalize(
        all_tracks, unique_kpts_per_image, min_len_track
    )

    print(f"Finalized matching with {len(final_keypoints)} images containing valid tracks.")
    return final_keypoints, final_matches


def _get_tile_data(tile, image_handles):
    """
    Return the pixel array for a tile, cropping it on demand from a lazily-opened
    source image (audit M4). Falls back to a pre-materialized ``tile_data`` when
    present (e.g. the small-image fallback tile that is pasted onto a black
    canvas). This avoids keeping every tile of every image resident in RAM.
    """
    data = tile.get('tile_data')
    if data is not None:
        return data
    handle = image_handles[tile['parent_image_path']]
    return np.array(handle.crop(tile['bounds_in_parent']).convert('RGB'))


def run_chunked_mast3r_matching(
    model, device, image_pairs_to_match, root_path, colmap_db,
    precomputed_transforms,
    tile_size=(512, 512), overlap_px=128, batch_size=4, conf_thr=1.5,
    dedup_distance=2.0, min_len_track=2, min_overlap_area=1.0, tile_pairing='per_a',
    pair_diagnostics=False, kpt_stride=8):
    """
    The COMPLETE REPLACEMENT for the original `run_mast3r_matching` function.
    It orchestrates the tiling, matching, aggregation, and database export.

    Defaults reflect audit decisions: ``conf_thr`` defaults to 1.5 (raw pointmap
    ``conf``, not descriptor confidence), ``min_len_track`` defaults to 2 (kept
    consistent with the mapper's ``ignore_two_view_tracks=False``),
    ``min_overlap_area`` gates tile-pair candidates, and ``tile_pairing`` selects
    the pairing strategy ('per_a' best-B-per-A by default for speed; 'all' for
    many-to-many, audit H6). All are overridable via CLI by the orchestrator.
    """

    # Defensive: collapse duplicate / reverse-direction pairs so the symmetric
    # A<->B matching is never run twice. (tile_transform.py stores each pair in
    # both directions; matching is symmetric, so this typically halves iterations.)
    _seen = set()
    _unique = []
    for _p1, _p2 in image_pairs_to_match:
        _k = tuple(sorted((_p1, _p2)))
        if _k not in _seen:
            _seen.add(_k)
            _unique.append((_p1, _p2))
    image_pairs_to_match = _unique

    # --- Part A: Tiling and Tile-Pair Determination ---
    # Tiles are metadata-only (bounds) and pixels are cropped on demand from
    # lazily-opened source images, so we never hold all tiles in RAM (audit M4).
    tile_cache = {}
    image_handles = {}
    all_image_paths_abs = np.unique([p for pair in image_pairs_to_match for p in pair])

    for image_path in tqdm(all_image_paths_abs, total=len(all_image_paths_abs), desc="Preparing tiles"):
        if image_path not in tile_cache:
            tile_cache[image_path] = generate_image_tiles(
                image_path, tile_size, overlap_px, save_tile_data=False, keep_tile_data=False)
            image_handles[image_path] = Image.open(image_path)

    # --- Part B: Inference on Tile Pairs and Match Aggregation ---
    aggregated_matches_temp = {}

    # Cache computed tile pairs per canonical image pair to avoid redundant work.
    tile_pair_cache = {}

    # Running totals for the optional per-pair coverage/cost diagnostic.
    diag = {'n_all': 0, 'n_per_a': 0, 'n_selected': 0, 'cand_area': 0.0, 'per_a_area': 0.0}

    print("Beginning chunked feature matching...")
    for image_path1, image_path2 in tqdm(image_pairs_to_match, desc="Processing Image Pairs"):

        tiles1 = tile_cache[image_path1]
        tiles2 = tile_cache[image_path2]

        # Canonical (sorted) key so pair order does not matter.
        canonical_key = tuple(sorted((image_path1, image_path2)))

        if canonical_key in tile_pair_cache:
            cached_pairs = tile_pair_cache[canonical_key]
            if image_path1 == canonical_key[0]:
                overlapping_tile_pairs = cached_pairs
            else:
                overlapping_tile_pairs = [(b, a) for a, b in cached_pairs]
            transform_1_to_2 = None  # not needed on cache hit
        else:
            name1_base = os.path.splitext(os.path.basename(image_path1))[0]
            name2_base = os.path.splitext(os.path.basename(image_path2))[0]

            transform_1_to_2 = precomputed_transforms.get((name1_base, name2_base))

            # Fall back to the reverse transform (inverted) if A->B is missing.
            if transform_1_to_2 is None:
                transform_2_to_1 = precomputed_transforms.get((name2_base, name1_base))
                if transform_2_to_1 is not None:
                    transform_1_to_2 = cv2.invertAffineTransform(transform_2_to_1)
                else:
                    continue  # Skip if no transform is found in either direction

            overlapping_tile_pairs = determine_overlapping_tile_pairs(
                tiles1, tiles2, transform_1_to_2,
                min_overlap_area=min_overlap_area, mode=tile_pairing)

            if image_path1 == canonical_key[0]:
                tile_pair_cache[canonical_key] = overlapping_tile_pairs
            else:
                tile_pair_cache[canonical_key] = [(b, a) for a, b in overlapping_tile_pairs]

        # Optional per-pair cost/coverage diagnostic (audit-discussion). Reports the
        # many-to-many ('all') tile-pair count vs the best-B-per-A ('per_a') count,
        # and the fraction of the matchable area per_a would cover relative to all.
        if pair_diagnostics and transform_1_to_2 is not None:
            cands = _tile_pair_candidates(tiles1, tiles2, transform_1_to_2, min_overlap_area)
            cand_area = sum(a for a, _, _ in cands)
            _best = {}
            for _a, _ia, _ib in cands:
                _cur = _best.get(_ia)
                if _cur is None or _a > _cur[0]:
                    _best[_ia] = (_a, _ib)
            _per_a_area = sum(_a for _a, _ in _best.values())
            _n_all = len(cands)
            _n_per_a = len(_best)
            _cov = (_per_a_area / cand_area * 100.0) if cand_area > 0 else 0.0
            print(f"[diag] {os.path.basename(image_path1)} <-> {os.path.basename(image_path2)}: "
                  f"A_tiles={len(tiles1)} B_tiles={len(tiles2)} | "
                  f"all={_n_all} per_a={_n_per_a} mode={tile_pairing}->{len(overlapping_tile_pairs)} | "
                  f"per_a covers ~{_cov:.0f}% of all-area")
            diag['n_all'] += _n_all
            diag['n_per_a'] += _n_per_a
            diag['n_selected'] += len(overlapping_tile_pairs)
            diag['cand_area'] += cand_area
            diag['per_a_area'] += _per_a_area

        if not overlapping_tile_pairs:
            continue

        # Process all tile pairs for this image pair in batches.
        # NOTE: no explicit ``total=`` here so tqdm accounts for the final partial
        # batch correctly (audit L5).
        for i in tqdm(range(0, len(overlapping_tile_pairs), batch_size),
                      desc="Matching Tile Pairs", leave=False, unit="pairs", unit_scale=1):
            batch_of_pairs = overlapping_tile_pairs[i:i + batch_size]

            inference_input_batch = []
            batch_tensors = []  # reuse normalized tensors for extraction
            for tile_a, tile_b in batch_of_pairs:
                img_a_tensor = ImgNorm(_get_tile_data(tile_a, image_handles))
                img_b_tensor = ImgNorm(_get_tile_data(tile_b, image_handles))

                shape_a = torch.tensor(img_a_tensor.shape[-2:]).unsqueeze(0)
                shape_b = torch.tensor(img_b_tensor.shape[-2:]).unsqueeze(0)

                dict_a = {
                    'img': img_a_tensor.unsqueeze(0),
                    'true_shape': shape_a,
                    'instance': tile_a['tile_id'],
                }
                dict_b = {
                    'img': img_b_tensor.unsqueeze(0),
                    'true_shape': shape_b,
                    'instance': tile_b['tile_id'],
                }
                inference_input_batch.append((dict_a, dict_b))
                batch_tensors.append((img_a_tensor, img_b_tensor))

            # Run inference.
            with torch.no_grad():
                output = inference(inference_input_batch, model, device,
                                   batch_size=len(inference_input_batch), verbose=False)

            # Extract, re-project, and aggregate matches for each result in the batch.
            for j, (tile_a, tile_b) in enumerate(batch_of_pairs):
                img_a_tensor, img_b_tensor = batch_tensors[j]
                pred1_single = {key: val[j] for key, val in output['pred1'].items()}
                pred2_single = {key: val[j] for key, val in output['pred2'].items()}

                kpts1_local, kpts2_local = extract_matches_from_tile_prediction(
                    pred1_single,
                    pred2_single,
                    img_a_tensor.shape[-2:],
                    img_b_tensor.shape[-2:],
                    conf_thr,
                    device,
                    kpt_stride=kpt_stride,
                )

                if kpts1_local is None:
                    continue

                offset_a = tile_a["bounds_in_parent"][:2]
                offset_b = tile_b["bounds_in_parent"][:2]
                kpts1_global = kpts1_local + offset_a
                kpts2_global = kpts2_local + offset_b

                pair_key = (image_path1, image_path2)
                if pair_key not in aggregated_matches_temp:
                    aggregated_matches_temp[pair_key] = {"kpts0": [], "kpts1": []}

                aggregated_matches_temp[pair_key]["kpts0"].append(kpts1_global)
                aggregated_matches_temp[pair_key]["kpts1"].append(kpts2_global)

    # Inference is finished; the model is not needed again. Move it to CPU and free
    # the CUDA cache so the GPU is released before the (CPU-bound) aggregation,
    # track-building and COLMAP steps run with an idle GPU.
    try:
        model.cpu()
        torch.cuda.empty_cache()
        print("[gpu] model moved to CPU and CUDA cache freed after matching.")
    except Exception as _e:
        print(f"[gpu] could not release GPU after matching: {_e}")

    if pair_diagnostics:
        _tot_cov = (diag['per_a_area'] / diag['cand_area'] * 100.0) if diag['cand_area'] > 0 else 0.0
        print("[diag] === totals over all image pairs ===")
        print(f"[diag]   tile-pair inferences: all={diag['n_all']}  per_a={diag['n_per_a']}  "
              f"selected(mode={tile_pairing})={diag['n_selected']}")
        print(f"[diag]   per_a would cover ~{_tot_cov:.0f}% of the all-area; "
              f"all costs ~{diag['n_all'] / diag['n_per_a']:.1f}x per_a inferences"
              if diag['n_per_a'] else "[diag]   (no per_a pairs)")

    # --- Finalize aggregation: concatenate per-pair match arrays ---
    print("Finalizing match aggregation...")
    aggregated_matches_final = {}
    for pair_key, kpt_data in tqdm(aggregated_matches_temp.items(), total=len(aggregated_matches_temp), desc="Finalizing Matches", leave=False):
        if not kpt_data["kpts0"]:
            continue
        aggregated_matches_final[pair_key] = {
            "kpts0": np.concatenate(kpt_data["kpts0"], axis=0),
            "kpts1": np.concatenate(kpt_data["kpts1"], axis=0),
        }

    # --- Part C: Deduplicate and Export ---
    unique_kpts, indexed_matches = deduplicate_and_format_for_colmap(
        aggregated_matches_final, min_len_track=min_len_track, distance_threshold=dedup_distance)
    
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

    for (path1, path2), matches in tqdm(indexed_matches.items(), total=len(indexed_matches), desc="Exporting Matches", leave=False):
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

# Re-enable PIL's decompression-bomb guard with a generous limit instead of
# disabling it process-wide (audit M8). 500 MP accommodates large aerial frames
# while still rejecting absurd/malicious images; dimensions are additionally
# validated inside generate_image_tiles.
_DEFAULT_MAX_IMAGE_PIXELS = 500_000_000
Image.MAX_IMAGE_PIXELS = _DEFAULT_MAX_IMAGE_PIXELS


def generate_image_tiles(image_path, tile_size=(512, 512), overlap_px=128, save_tile_data=False,
                         keep_tile_data=True, max_image_pixels=_DEFAULT_MAX_IMAGE_PIXELS):
    """
    Edge-aligned tiling: divides a large image into overlapping tiles. Instead of
    padding edge tiles, the last tile in each row/column is aligned to the image
    edge (a full-sized tile), which may yield larger overlap for the last tile.

    Args:
        image_path (str): The file path to the high-resolution image.
        tile_size (tuple): The (width, height) of the tiles to generate. MUST be
                           divisible by the model's patch size.
        overlap_px (int): The *minimum* number of pixels of overlap between adjacent tiles.
        save_tile_data (bool): If True, dump a lightweight tile manifest JSON.
        keep_tile_data (bool): If True (default) tiles materialize ``tile_data`` as
                               before. The chunked matcher passes False and crops
                               pixels on demand via ``_get_tile_data`` to keep
                               memory low (audit M4).
        max_image_pixels (int): Per-image pixel cap for the decompression-bomb guard.

    Returns:
        list: A list of tile dictionaries, all of uniform target size.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_w, img_h = img.size
    except Exception as e:
        print(f"ERROR: Could not open image {image_path}. Reason: {e}")
        return []

    # Explicit decompression-bomb guard (audit M8).
    if max_image_pixels is not None and img_w * img_h > max_image_pixels:
        raise ValueError(
            f"Image {image_path} is {img_w}x{img_h} ({img_w * img_h} px), which "
            f"exceeds the configured limit of {max_image_pixels} px.")

    tiles_manifest = []
    target_w, target_h = tile_size

    # Check if the image is smaller than a single tile
    if img_w <= target_w or img_h <= target_h:
        # Fallback: paste the small image onto a black canvas of tile_size.
        # This materializes tile_data (a single, small tile) regardless of
        # keep_tile_data, since on-demand cropping could not reproduce the paste.
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

    # Edge-aligned coordinate generation: normal starting points plus a final
    # start that aligns the last tile's far edge with the image edge.
    y_starts = list(range(0, img_h - target_h, stride_h)) + [img_h - target_h]
    x_starts = list(range(0, img_w - target_w, stride_w)) + [img_w - target_w]

    # Prevent duplicate tiles when the image size is a perfect fit.
    y_starts = np.unique(y_starts).tolist()
    x_starts = np.unique(x_starts).tolist()

    tile_id_counter = 0
    for y in y_starts:
        for x in x_starts:
            bbox = (x, y, x + target_w, y + target_h)
            tile_img = img.crop(bbox)

            tile_info = {
                "tile_id": f"{os.path.basename(image_path)}_tile_{tile_id_counter:04d}",
                "parent_image_path": image_path,
                "bounds_in_parent": bbox,
                # Lazily cropped via _get_tile_data unless explicitly kept (audit M4).
                "tile_data": np.array(tile_img) if keep_tile_data else None,
            }
            tiles_manifest.append(tile_info)
            tile_id_counter += 1

    # Optionally dump a lightweight manifest (without pixel data).
    if save_tile_data:
        tiles_manifest_json = [
            {k: v for k, v in tile.items() if k != "tile_data"}
            for tile in tiles_manifest
        ]
        json.dump(tiles_manifest_json, open(f"{image_path}_tiles_manifest.json", 'w'), indent=4)

    return tiles_manifest


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
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


def _tile_pair_candidates(tiles_A, tiles_B, transform_A_to_B, min_overlap_area):
    """
    Return all overlapping (A,B) tile candidates as a list of (area, idx_A, idx_B),
    where `area` is the projected intersection area in px^2. Shared by the pairing
    selection and the per-pair coverage diagnostic.

    ``transform_A_to_B`` may be either a 2x3 affine (``cv2.transform``) or a full
    3x3 homography (``cv2.perspectiveTransform``). The latter is produced from EO
    by ``eo_to_tilepairs.py`` and is required for oblique imagery; the affine form
    remains the default (footprint-derived, near-nadir).
    """
    if transform_A_to_B is None or not tiles_A or not tiles_B:
        return []
    is_homography = np.asarray(transform_A_to_B).shape == (3, 3)
    candidates = []
    for ia, tile_a in enumerate(tiles_A):
        x_min, y_min, x_max, y_max = tile_a["bounds_in_parent"]
        corners_a = np.array([[[x_min, y_min]], [[x_max, y_min]],
                              [[x_max, y_max]], [[x_min, y_max]]], dtype=np.float32)
        if is_homography:
            transformed_corners = cv2.perspectiveTransform(corners_a, transform_A_to_B)
        else:
            transformed_corners = cv2.transform(corners_a, transform_A_to_B)
        projected_bbox_in_B = [np.min(transformed_corners[:, :, 0]),
                               np.min(transformed_corners[:, :, 1]),
                               np.max(transformed_corners[:, :, 0]),
                               np.max(transformed_corners[:, :, 1])]
        for ib, tile_b in enumerate(tiles_B):
            area = _calculate_intersection_area(projected_bbox_in_B, tile_b["bounds_in_parent"])
            if area >= min_overlap_area:
                candidates.append((area, ia, ib))
    return candidates


def determine_overlapping_tile_pairs(tiles_A, tiles_B, transform_A_to_B, min_overlap_area=1.0, mode='per_a'):
    """
    Find overlapping tile pairs between two images using a geometric transform.

    Audit H6 / performance: tile pairing trades coverage against inference cost
    (each tile pair is one model inference).

    - ``mode='per_a'`` (default): each A tile is matched to its single
      highest-overlap B tile; a B tile may be chosen by several A tiles (no
      "stealing"). Unlike the old strict one-to-one this never *drops* an A tile,
      so coverage is near-complete at ~one inference per A tile (fast). Residual
      thin slivers remain where an A tile straddles a B-grid boundary and spills
      into a B tile it did not pick.
    - ``mode='all'``: many-to-many. Every (A,B) above the threshold is kept. This
      is the only fully gap-free option, but it matches each physical region
      multiple times (e.g. ~4x for a 2x2 tile neighbourhood), which is largely
      redundant after deduplication and is ~3-4x slower.

    ``min_overlap_area`` and ``mode`` are exposed as CLI flags by the orchestrator.

    Args:
        tiles_A (list): Tile manifest for the source image A.
        tiles_B (list): Tile manifest for the destination image B.
        transform_A_to_B (np.ndarray): 2x3 affine OR 3x3 homography mapping
            A -> B pixels. A 3x3 triggers ``cv2.perspectiveTransform`` (needed for
            EO-derived homographies on oblique imagery); 2x3 uses ``cv2.transform``.
        min_overlap_area (float): Minimum intersection area (px^2) for a candidate.
        mode (str): 'per_a' (best-B-per-A, default) or 'all' (many-to-many).

    Returns:
        list: A list of (tile_a, tile_b) tuples.
    """
    candidates = _tile_pair_candidates(tiles_A, tiles_B, transform_A_to_B, min_overlap_area)

    if mode == 'all':
        return [(tiles_A[ia], tiles_B[ib]) for _, ia, ib in candidates]

    # 'per_a': each A tile -> its single highest-overlap B tile (B may repeat).
    best_for_a = {}
    for area, ia, ib in candidates:
        cur = best_for_a.get(ia)
        if cur is None or area > cur[0]:
            best_for_a[ia] = (area, ib)
    return [(tiles_A[ia], tiles_B[ib]) for ia, (_, ib) in sorted(best_for_a.items())]
