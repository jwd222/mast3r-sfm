import sys
import argparse
import torch
import numpy as np
import os
import re
import cv2
import trimesh
import math
from pathlib import Path
from PIL import Image
from typing import NamedTuple, Optional
from tqdm import tqdm
from collections import defaultdict

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'mast3r'))
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
import mast3r.utils.path_to_dust3r

from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs

from plyfile import PlyData, PlyElement


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def invert_matrix(mat):
    """Invert a torch or numpy matrix."""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'Unsupported matrix type: {type(mat)}')

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# Ensure save directories exist
def init_filestructure(save_path):
    save_path.mkdir(exist_ok=True, parents=True)
    images_path = save_path / 'images'
    masks_path = save_path / 'masks'
    sparse_path = save_path / 'sparse/0'
    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)
    sparse_path.mkdir(exist_ok=True, parents=True)
    return save_path, images_path, masks_path, sparse_path

# Save images and masks
def save_images_and_masks(imgs, masks, images_path, img_files, masks_path):
    for i, (image, name, mask) in enumerate(zip(imgs, img_files, masks)):
        imgname = Path(name).stem
        image_save_path = images_path / f"{imgname}.png"
        # mask_save_path = masks_path / f"{imgname}.png"
        rgb_image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)
        # mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2) * 255
        # Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)

# Save camera information
def save_cameras(focals, principal_points, sparse_path, imgs_shape):
    cameras_file = sparse_path / 'cameras.txt'
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            # camera index should be 1-indexed
            f.write(f"{i+1} PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal} {focal} {pp[0]} {pp[1]}\n")

# Save image transformations
def _save_images_txt(world2cam, img_files, sparse_path):
    images_file = sparse_path / 'images.txt'
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(world2cam.shape[0]):
            name = Path(img_files[i]).stem
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}.png\n\n")

# Save point cloud with normals
def save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path):
    pc = get_point_cloud(imgs, pts3d, masks)
    default_normal = [0, 1, 0]
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))
    save_path = sparse_path / 'points3D.ply'
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))
    with open(save_path, 'w') as f:
        f.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {int(color[0])} {int(color[1])} {int(color[2])} {normal[0]} {normal[1]} {normal[2]}\n")

# New function replacing the above
def _save_points3D_txt(sparse_path, pts3d, masks, imgs):
    """Saves the point cloud in COLMAP's points3D.txt format."""
    # First, get the point cloud data as you did before
    imgs_np = to_numpy(imgs)
    pts3d_np = to_numpy(pts3d)
    masks_np = to_numpy(masks)
    
    # Flatten the points and colors from all images
    points = np.concatenate([p[m] for p, m in zip(pts3d_np, masks_np.reshape(masks_np.shape[0], -1))])
    colors = np.concatenate([c[m] for c, m in zip(imgs_np, masks_np)])
    
    # The DUST3R/MASt3R output can have many duplicate points. Let's take every Nth point for a sparser cloud.
    # This also makes the COLMAP project more manageable.
    subsample_rate = 5 # Keep 1 in every 5 points
    points = points.reshape(-1, 3)[::subsample_rate]
    colors = (colors.reshape(-1, 3) * 255)[::subsample_rate].astype(np.uint8)

    # The format of points3D.txt is:
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK_LENGTH, TRACK...
    # Since we don't have the track, we'll set ERROR to 1 and TRACK_LENGTH to 0.
    
    filepath = sparse_path / 'points3D.txt'
    with open(filepath, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}, mean track length: 0\n".format(len(points)))
        
        for i, (point, color) in enumerate(zip(points, colors)):
            x, y, z = point
            r, g, b = color
            error = 1.0 # Placeholder error
            track_length = 0 # No track information
            #           POINT3D_ID,  X,  Y,  Z,  R,  G,  B, ERROR, TRACK_LENGTH
            f.write(f"{i+1} {x} {y} {z} {r} {g} {b} {error} {track_length}\n")

    print(f"[INFO] Created points3D.txt with {len(points)} points.")

# Generate point cloud
def get_point_cloud(imgs, pts3d, mask):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask.reshape(mask.shape[0], -1))])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    pts = pts.reshape(-1, 3)[::3]
    col = col.reshape(-1, 3)[::3]
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals
    return pct

#################################################################################################
def save_images_txt(world2cam, img_files, sparse_path, pts3d, masks, imgs, min_conf_thr=1.5):
    images_file = sparse_path / 'images.txt'
    # Create mapping from 2D points to 3D points
    point_id_counter = 1
    point_mappings = []  # For each image, store a mapping from 2D index to point ID
    points3D_data = []  # Store 3D point data for later saving
    
    # First pass: create point IDs and mappings
    for i in range(len(imgs)):
        h, w = masks[i].shape[:2]
        mapping = np.full((h, w), -1, dtype=np.int64)  # Default to -1 (no 3D point)
        mask = masks[i]
        
        # Get valid points
        valid_points = pts3d[i][mask.flatten()]
        valid_positions = np.argwhere(mask)
        
        # Create IDs for valid points
        for pt, (y, x) in zip(valid_points, valid_positions):
            mapping[y, x] = point_id_counter
            # Get color from image
            color = imgs[i][y, x] * 255
            points3D_data.append((point_id_counter, pt, color, i, (x, y)))
            point_id_counter += 1
        
        point_mappings.append(mapping)
    
    # Second pass: write images.txt with point correspondences
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i in range(world2cam.shape[0]):
            name = Path(img_files[i]).stem
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}.png\n")
            
            # Write 2D points with their 3D IDs
            mapping = point_mappings[i]
            h, w = mapping.shape
            points_line = []
            
            # We'll write every 5th point to reduce file size
            for y in range(0, h, 5):
                for x in range(0, w, 5):
                    point_id = mapping[y, x]
                    if point_id != -1:
                        points_line.append(f"{x} {y} {point_id}")
                    else:
                        points_line.append(f"{x} {y} -1")
            
            f.write(" ".join(points_line) + "\n")
    
    return points3D_data

def save_points3D_txt(sparse_path, points3D_data):
    points3D_file = sparse_path / 'points3D.txt'
    with open(points3D_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        # Group points by ID
        points_dict = {}
        for point_id, pt, color, img_id, (x, y) in points3D_data:
            if point_id not in points_dict:
                points_dict[point_id] = {
                    'pt': pt,
                    'color': color,
                    'tracks': []
                }
            points_dict[point_id]['tracks'].append((img_id, x, y))
        
        # Write each 3D point
        for point_id, data in points_dict.items():
            x, y, z = data['pt']
            r, g, b = data['color'].astype(int)
            tracks = data['tracks']
            
            # Format tracks: (IMAGE_ID, POINT2D_IDX)
            track_str = f"{len(tracks)} " + " ".join(
                f"{img_id} {x + y * 10000}"  # Create unique 2D index
                for img_id, x, y in tracks
            )
            
            f.write(f"{point_id} {x} {y} {z} {r} {g} {b} 1.0 {track_str}\n")
#################################################################################################
def save_colmap_sparse_model(sparse_path, scene, img_files, imgs):
    """
    Saves the sparse reconstruction in COLMAP's text format, including
    images.txt with 2D keypoints and points3D.txt with full track information.
    This version filters out points with track lengths < 2 to ensure
    compatibility with COLMAP's bundle adjuster.
    """
    print("[INFO] Building COLMAP sparse model with full tracks...")
    
    try:
        # Call the expensive function only ONCE and store the original observations
        unique_pts3d, all_observations = scene.get_correspondences()
        unique_pts3d_np = to_numpy(unique_pts3d)
        initial_num_points = len(unique_pts3d_np)
        
        if initial_num_points == 0:
            print("[WARNING] No correspondences found to build a sparse model.")
            return

    except AttributeError:
        print("[ERROR] scene.get_correspondences() not found.")
        print("Please ensure you have modified sparse_ga.py correctly.")
        return

    # --- NEW: Filter points to ensure all have at least 2 observations ---
    print(f"[INFO] Initial number of 3D points: {initial_num_points}")
    
    filtered_observations = []
    filtered_unique_pts3d_np = []
    # This map will store the old ID -> new ID mapping
    old_to_new_pt3d_id = {}
    
    for old_id, (track_obs, pt3d) in enumerate(zip(all_observations, unique_pts3d_np)):
        if len(track_obs) > 1:
            # This is the new, dense index for the point
            new_id = len(filtered_observations)
            filtered_observations.append(track_obs)
            filtered_unique_pts3d_np.append(pt3d)
            old_to_new_pt3d_id[old_id] = new_id

    num_unique_points = len(filtered_observations)
    unique_pts3d_np = np.array(filtered_unique_pts3d_np)
    observations = filtered_observations # Overwrite with the filtered list
    
    print(f"[INFO] Filtered to {num_unique_points} points with track length > 1.")
    
    if num_unique_points == 0:
        print("[WARNING] No points with sufficient track length. Cannot create a valid sparse model.")
        return

    # Create data structures for writing files
    images_data = defaultdict(lambda: {'points2D': []})
    points3D_data = defaultdict(lambda: {'track': []})
    
    # --- LOOP 1: Populate data structures using filtered data ---
    iterator = tqdm(enumerate(observations), desc="[1/3] Populating track data", total=num_unique_points, unit="points")
    for new_pt3d_id, track_obs in iterator:
        for img_id, pt2d in track_obs:
            pt2d_idx = len(images_data[img_id]['points2D'])
            # Here, we need to find the original pt3d_id that corresponds to this observation,
            # but since we are iterating through the filtered `observations`, we can directly use the new_pt3d_id.
            images_data[img_id]['points2D'].append({'xy': pt2d, 'pt3d_id': new_pt3d_id + 1}) # Use new 1-based ID
            points3D_data[new_pt3d_id]['track'].append({'img_id': img_id + 1, 'pt2d_idx': pt2d_idx})

        if track_obs:
            first_img_id, first_pt2d = track_obs[0]
            h, w, _ = imgs[first_img_id].shape
            u, v = int(round(first_pt2d[0])), int(round(first_pt2d[1]))
            if 0 <= u < w and 0 <= v < h:
                color = (imgs[first_img_id][v, u] * 255).astype(np.uint8)
                points3D_data[new_pt3d_id]['rgb'] = color

    # --- LOOP 2: Write images.txt ---
    images_file = sparse_path / 'images.txt'
    world2cam = invert_matrix(scene.get_im_poses().detach()).cpu().numpy()
    
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(img_files)}\n")

        iterator = tqdm(range(len(img_files)), desc="[2/3] Writing images.txt", unit="images")
        for i in iterator:
            img_id_colmap = i + 1
            cam_id = i + 1
            
            rotation_matrix = world2cam[i, :3, :3]
            qvec = rotmat2qvec(rotation_matrix)
            tvec = world2cam[i, :3, 3]
            
            f.write(f"{img_id_colmap} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {cam_id} {Path(img_files[i]).name}\n")
            
            # --- MODIFIED: Ensure POINT3D_ID is valid or -1 ---
            # We need to reconstruct the full point list for each image,
            # including points that were filtered out.
            # This requires a more careful construction of the `images_data` dictionary.
            # Let's rebuild it properly before this loop.
            pass # Placeholder, the logic will be handled below.

    # --- Correctly build image data and write images.txt ---
    # Re-get ALL observations to build the full 2D point list for each image
    # _, all_observations = scene.get_correspondences()
    full_images_data = defaultdict(list)
    for old_pt3d_id, track_obs in enumerate(all_observations):
        for img_id, pt2d in track_obs:
            # Get new ID or -1 if the point was filtered out
            new_pt3d_id = old_to_new_pt3d_id.get(old_pt3d_id, -1)
            if new_pt3d_id != -1:
                new_pt3d_id += 1 # Make it 1-based for COLMAP
            full_images_data[img_id].append({'xy': pt2d, 'pt3d_id': new_pt3d_id})

    # Now, write the images.txt file using this correctly built data
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(img_files)}\n")
        iterator = tqdm(range(len(img_files)), desc="[2/3] Writing images.txt", unit="images")
        for i in iterator:
            # Writing the pose line is the same as before
            img_id_colmap = i + 1; cam_id = i + 1; rotation_matrix = world2cam[i, :3, :3]; qvec = rotmat2qvec(rotation_matrix); tvec = world2cam[i, :3, 3]
            f.write(f"{img_id_colmap} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {cam_id} {Path(img_files[i]).name}\n")
            
            # Write the points line using the data we just created
            points2D = full_images_data[i]
            point_entries = [f"{p['xy'][0]} {p['xy'][1]} {p['pt3d_id']}" for p in points2D]
            f.write(" ".join(point_entries) + "\n")

    print(f"[INFO] Created images.txt with {len(img_files)} images.")

    # --- LOOP 3: Write points3D.txt using filtered data ---
    points3D_file = sparse_path / 'points3D.txt'
    mean_track_length = np.mean([len(p['track']) for p in points3D_data.values()]) if points3D_data else 0

    with open(points3D_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {num_unique_points}, mean track length: {mean_track_length:.4f}\n")
        
        iterator = tqdm(range(num_unique_points), desc="[3/3] Writing points3D.txt", unit="points")
        for new_pt3d_id in iterator:
            xyz = unique_pts3d_np[new_pt3d_id]
            info = points3D_data[new_pt3d_id]
            rgb = info.get('rgb', [200, 200, 200])
            error = 1.0
            track = info['track']
            
            track_str = " ".join([f"{obs['img_id']} {obs['pt2d_idx']}" for obs in track])
            
            f.write(f"{new_pt3d_id + 1} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {error} {track_str}\n")
            
    print(f"[INFO] Created points3D.txt with {num_unique_points} points.")
#################################################################################################

def main(image_dir, save_dir, model_path, device, batch_size, image_size, schedule, lr, niter, min_conf_thr, tsdf_thresh):
    # Load model and images
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    image_files = sorted([str(x) for x in Path(image_dir).iterdir() if x.suffix in ['.png', '.JPG', '.jpg', '.PNG']],
                         key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))
    images = load_images(image_files, size=image_size)

    # Generate pairs and run inference
    # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    # output = inference(pairs, model, device, batch_size=1, verbose=True)
    
    # This is the best strategy for the sequential South Building dataset.
    # It creates pairs between an image and its 10 neighbors, runs the matching
    # in both directions for robustness, and avoids any unnecessary filtering.
    pairs = make_pairs(images, scene_graph='swin-10', prefilter=None, symmetrize=True)

    cache_dir = os.path.join(save_dir, 'cache')
    if os.path.exists(cache_dir):
        os.system(f'rm -rf {cache_dir}')
    scene = sparse_global_alignment(image_files, pairs, cache_dir,
                                    model, lr1=args.lr1, niter1=args.niter1, lr2=args.lr2, niter2=args.niter2,
                                    device=args.device, opt_depth=False,  
                                    shared_intrinsics=args.shared_intrinsics,
                                    matching_conf_thr=args.matching_conf_thr)
    # Extract scene information
    world2cam = invert_matrix(scene.get_im_poses().detach()).cpu().numpy()
    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    imgs = np.array(scene.imgs)

    tsdf = TSDFPostProcess(scene, TSDF_thresh=tsdf_thresh)
    pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=True))
    masks = np.array(to_numpy([c > min_conf_thr for c in confs]))

    # Main execution
    save_path, images_path, masks_path, sparse_path = init_filestructure(Path(save_dir))
    save_images_and_masks(imgs, masks, images_path, image_files, masks_path)
    save_cameras(focals, principal_points, sparse_path, imgs_shape=imgs.shape)
    
    # ########################### GROK ##############################################
    # _save_images_txt(world2cam, image_files, sparse_path)
    # save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
    # ###### Save points3D.txt and images.txt
    # save_points3D_txt(sparse_path, pts3d, masks, imgs)
    # ########################### GROK ##############################################
   
    # ###################### GEMINI ###############################################
    # NEW: Generate the complete COLMAP sparse model with tracks
    save_colmap_sparse_model(sparse_path, scene, image_files, imgs)
    
    # This creates a separate DENSE point cloud for viewing, which is fine to keep.
    # Note: This .ply is different from the sparse points in points3D.txt
    save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
    # ###################### GEMINI ###############################################
    
    # ######################### DS #######################################
    # # Save images.txt and get keypoint information
    # # Save images.txt and get point cloud data
    # points3D_data = save_images_txt(
    #     world2cam, image_files, sparse_path, 
    #     pts3d, masks, imgs, min_conf_thr
    # )
    
    # # Save points3D.txt
    # save_points3D_txt(sparse_path, points3D_data)
    # ######################### DS #######################################
    

    print(f'[INFO] Mast3R Reconstruction is successfully converted to COLMAP files in: {str(sparse_path)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--min_conf_thr', type=float, default=1.5, help='Minimum confidence threshold')
    parser.add_argument('--tsdf_thresh', type=float, default=0.0, help='TSDF threshold')
    ### newly added
    parser.add_argument('--lr1', type=float, default=0.07, help='Learning rate for first optimization stage')
    parser.add_argument('--niter1', type=int, default=500, help='Number of iterations for first optimization stage')
    parser.add_argument('--lr2', type=float, default=0.014, help='Learning rate for second optimization stage')
    parser.add_argument('--niter2', type=int, default=200, help='Number of iterations for second optimization stage')
    parser.add_argument('--matching_conf_thr', type=float, default=5.0, help='Confidence threshold for matches to be used in optimization')
    parser.add_argument('--shared_intrinsics', action='store_true', help='Use shared intrinsics for all cameras')

    args = parser.parse_args()
    main(args.image_dir, args.save_dir, args.model_path, args.device, args.batch_size, args.image_size, args.schedule, args.lr, args.niter, args.min_conf_thr, args.tsdf_thresh)