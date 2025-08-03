import os
import json
import geopandas as gpd
import rasterio
import numpy as np
import cv2
from itertools import combinations
from tqdm import tqdm

def get_pixel_coords_from_world(raster_path, world_coords):
    """
    Converts a list of world coordinates (e.g., UTM) to pixel coordinates (x, y)
    for a given georeferenced image.

    Args:
        raster_path (str): Path to the GeoTIFF image.
        world_coords (list of tuples): A list of (longitude, latitude) or (easting, northing) points.

    Returns:
        np.ndarray: An array of corresponding (x, y) pixel coordinates.
    """
    with rasterio.open(raster_path) as src:
        # The 'index' method is rasterio's way of doing the world -> pixel transformation
        # It returns (row, col) which is equivalent to (y, x)
        rows, cols = rasterio.transform.rowcol(src.transform, 
                                               [wc[0] for wc in world_coords], 
                                               [wc[1] for wc in world_coords])
        # We need to return as (x, y), so we stack (cols, rows)
        return np.vstack((cols, rows)).T

def calculate_transform_between_images(img_path_A, footprint_A, img_path_B, footprint_B):
    """
    Calculates the 2x3 affine transformation matrix to map pixel coordinates
    from Image A to Image B.

    Args:
        img_path_A (str): Path to the source image A.
        footprint_A (Polygon): The shapely Polygon for image A's footprint.
        img_path_B (str): Path to the destination image B.
        footprint_B (Polygon): The shapely Polygon for image B's footprint.

    Returns:
        np.ndarray or None: The 2x3 affine transformation matrix, or None if
                            the calculation fails (e.g., not enough points).
    """
    # Find the geographic region where the two images overlap.
    overlap_polygon = footprint_A.intersection(footprint_B)
    if overlap_polygon.is_empty or overlap_polygon.geom_type != 'Polygon':
        return None # No or insufficient overlap

    # The corners of the overlapping region are our control points.
    # We use their world coordinates.
    overlap_world_coords = list(overlap_polygon.exterior.coords)

    # We need at least 3 points to calculate an affine transform.
    if len(overlap_world_coords) < 3:
        return None

    # Get the pixel coordinates in BOTH images for these same world points.
    pixel_coords_in_A = get_pixel_coords_from_world(img_path_A, overlap_world_coords)
    pixel_coords_in_B = get_pixel_coords_from_world(img_path_B, overlap_world_coords)
    
    # We only need 3 corresponding points for cv2.getAffineTransform.
    # We use them as float32, which OpenCV expects.
    src_pts = np.float32(pixel_coords_in_A[:3])
    dst_pts = np.float32(pixel_coords_in_B[:3])

    # Let OpenCV calculate the 2x3 affine matrix.
    try:
        transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)
        return transform_matrix
    except cv2.error as e:
        print(f"OpenCV error calculating transform between {os.path.basename(img_path_A)} and {os.path.basename(img_path_B)}: {e}")
        return None


def main(footprints_path, image_dir, output_path, image_id_col='image_id'):
    """
    Main function to generate and save pairwise image transformations.

    Args:
        footprints_path (str): Path to the GeoPackage or Shapefile of footprints.
        image_dir (str): Directory containing the high-resolution GeoTIFF images.
        output_path (str): Path to save the output JSON file.
        image_id_col (str): The column name in the gdf that contains the image filename.
    """
    print("Loading footprints...")
    gdf = gpd.read_file(footprints_path)
    
    # Create a dictionary mapping image filenames to their full paths and geometries
    image_data = {}
    for idx, row in gdf.iterrows():
        img_filename = row[image_id_col]
        img_path = os.path.join(image_dir, img_filename + '.tif')
        if os.path.exists(img_path):
            image_data[img_filename] = {
                "path": img_path,
                "geometry": row.geometry
            }
        else:
            pass
            # print(f"Warning: Image file not found for footprint '{img_filename}'")

    print(f"Found {len(image_data)} matching images in the directory.")
    
    transforms = {}
    
    # Iterate through all unique pairs of images
    image_filenames = list(image_data.keys())
    for name_A, name_B in tqdm(combinations(image_filenames, 2), total=len(image_filenames) * (len(image_filenames) - 1) // 2, desc="Calculating transformations"):
        # print(f"Processing pair: {name_A} <-> {name_B}")
        
        data_A = image_data[name_A]
        data_B = image_data[name_B]

        # Calculate A -> B transform
        transform_A_to_B = calculate_transform_between_images(
            data_A['path'], data_A['geometry'],
            data_B['path'], data_B['geometry']
        )
        if transform_A_to_B is not None:
            # We save the numpy array as a list of lists for JSON serialization
            transforms[f"{name_A}__{name_B}"] = transform_A_to_B.tolist()

        # It's also useful to pre-calculate the inverse transform B -> A
        transform_B_to_A = calculate_transform_between_images(
            data_B['path'], data_B['geometry'],
            data_A['path'], data_A['geometry']
        )
        if transform_B_to_A is not None:
            transforms[f"{name_B}__{name_A}"] = transform_B_to_A.tolist()
            
    print(f"\nCalculated {len(transforms)} pairwise transformations.")
    
    # Save the dictionary to a JSON file
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=4)
        
    print(f"Successfully saved transformations to {output_path}")

if __name__ == '__main__':
    # --- Configuration ---
    # Path to your footprint file (e.g., .shp, .gpkg)
    FP_PATH = "/mnt/d/projects/wsl_projects/Projects/3_repo/mast3r/data/tobias/1/fp/footprints.shp"
    # Directory where your .tif files are stored
    IMG_DIR = "/mnt/d/projects/wsl_projects/Projects/3_repo/mast3r/data/tobias/1/images"
    # The output file that will store the results
    OUTPUT_JSON = "/mnt/d/projects/wsl_projects/Projects/3_repo/mast3r/data/tobias/1/fp/image_transforms.json"
    # The column name in your footprints file that holds the image filename (e.g., 'IMG_001.tif')
    IMAGE_ID_COLUMN = "id" 

    main(FP_PATH, IMG_DIR, OUTPUT_JSON, IMAGE_ID_COLUMN)
