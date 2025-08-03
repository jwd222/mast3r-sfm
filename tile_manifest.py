import os
from PIL import Image
import numpy as np
import cv2 # Using OpenCV for robust geometric transformations
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

# It's good practice to disable the DecompressionBomb check for large images
Image.MAX_IMAGE_PIXELS = None

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

def determine_overlapping_tile_pairs_from_images(image_path_A, image_path_B, transform_A_to_B, tile_size=(1024, 1024), overlap_px=256):
    tiles_manifest_A = generate_image_tiles(image_path_A, tile_size=tile_size, overlap_px=overlap_px)
    tiles_manifest_B = generate_image_tiles(image_path_B, tile_size=tile_size, overlap_px=overlap_px)

    # Get the required transform from the dictionary
    transform_A_to_B = image_transforms.get((image_A_name, image_B_name))

    if transform_A_to_B is not None:
        # You now have the exact 2x3 matrix needed for the Step 2 function
        # from our previous discussion.
        overlapping_tile_pairs = determine_overlapping_tile_pairs(
            tiles_manifest_A, 
            tiles_manifest_B, 
            transform_A_to_B
        )
    return overlapping_tile_pairs

if __name__ == "__main__":
    # Load the pre-calculated transforms
    with open('/mnt/d/projects/wsl_projects/Projects/3_repo/mast3r/data/tobias/1/fp/image_transforms_one_way_tobias.json', 'r') as f:
        transform_data_loaded = json.load(f)

    # Convert the lists back to numpy arrays
    image_transforms = {
        tuple(key.split('__')): np.array(value)
        for key, value in transform_data_loaded.items()
    }

    # Now, when you need to find overlapping tiles between image_A.tif and image_B.tif:
    image_path = Path("/mnt/d/projects/wsl_projects/Projects/3_repo/mast3r/data/tobias/1/images")
    image_A_name = "138001372_0021_01_0031_P00_01"
    image_B_name = "138001373_0021_01_0030_P00_01"

    overlapping_tiles = determine_overlapping_tile_pairs_from_images(
        str(image_path / (image_A_name + '.tif')),
        str(image_path / (image_B_name + '.tif')),
        image_transforms.get((image_A_name, image_B_name))
    )

    # tiles_manifest_A = generate_image_tiles(str(image_path / (image_A_name + '.tif')))
    # tiles_manifest_B = generate_image_tiles(str(image_path / (image_B_name + '.tif')))

    # # Get the required transform from the dictionary
    # transform_A_to_B = image_transforms.get((image_A_name, image_B_name))

    # if transform_A_to_B is not None:
    #     # You now have the exact 2x3 matrix needed for the Step 2 function
    #     # from our previous discussion.
    #     overlapping_tile_pairs = determine_overlapping_tile_pairs(
    #         tiles_manifest_A, 
    #         tiles_manifest_B, 
    #         transform_A_to_B
    #     )
    pass