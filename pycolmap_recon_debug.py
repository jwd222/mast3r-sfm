import pycolmap
from pathlib import Path


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    """Decode COLMAP's pair_id into (image_id1, image_id2)."""
    image_id2 = pair_id % (2**32)
    image_id1 = pair_id // (2**32)
    return image_id1, image_id2


def debug_colmap_database(db_path: Path):
    if not db_path.exists():
        print(f"Error: Database file not found at '{db_path}'")
        return

    print("=" * 80)
    print(f"DEBUGGING COLMAP DATABASE: {db_path.name}")
    print("=" * 80)

    try:
        db = pycolmap.Database()
        db.open(str(db_path))
        print("Successfully connected to the database.\n")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return

    # --- 1. Cameras and images ---
    cameras = db.read_all_cameras()
    images = db.read_all_images()
    print("--- Summary ---")
    print(f"Cameras found: {len(cameras)}")
    print(f"Images found:  {len(images)}\n")
    if len(images) == 0:
        print("Database contains no images. Reconstruction cannot proceed.")
        return

    # --- 2. Keypoints ---
    print("--- Keypoints Analysis ---")
    num_with_kpts = 0
    for i, img in enumerate(images):
        num_kpts = db.num_keypoints_for_image(img.image_id)
        if num_kpts > 0:
            num_with_kpts += 1
        if i < 5:
            print(f"  - Image ID {img.image_id} ({img.name}): {num_kpts} keypoints")
    if len(images) > 5:
        print(f"  ... and {len(images) - 5} more images.")
    print(f"Status: {'SUCCESS' if num_with_kpts > 0 else 'FAILED'}. "
          f"Found keypoints for {num_with_kpts} / {len(images)} images.\n")

    # --- 3. Matches ---
    print("--- Matches Analysis ---")
    pair_ids, match_arrays = db.read_all_matches()
    if len(pair_ids) == 0:
        print("Status: FAILED. No matches found in the `matches` table.")
    else:
        total_matches = sum(len(m) for m in match_arrays)
        print(f"Status: SUCCESS. Found {total_matches} matches "
              f"across {len(pair_ids)} image pairs.")
        for i, (pair_id, matches) in enumerate(zip(pair_ids, match_arrays)):
            img1, img2 = pair_id_to_image_ids(pair_id)
            if i >= 5:
                print(f"  ... and {len(pair_ids) - 5} more pairs with matches.")
                break
            print(f"  - Pair ({img1}, {img2}): {len(matches)} matches")
    print("")

    # --- 4. Two-view geometries ---
    print("--- Two-View Geometry Analysis ---")
    tv_pair_ids, tv_geoms = db.read_two_view_geometries()
    if len(tv_pair_ids) == 0:
        print("Status: WARNING. No verified geometries found.")
    else:
        total_inliers = sum(len(tvg.inlier_matches) for tvg in tv_geoms)
        print(f"Status: SUCCESS. Found {total_inliers} inlier matches "
              f"across {len(tv_pair_ids)} verified pairs.")
        for i, (pair_id, tvg) in enumerate(zip(tv_pair_ids, tv_geoms)):
            img1, img2 = pair_id_to_image_ids(pair_id)
            if i >= 5:
                print(f"  ... and {len(tv_pair_ids) - 5} more verified pairs.")
                break
            print(f"  - Verified Pair ({img1}, {img2}): "
                  f"{len(tvg.inlier_matches)} inliers")
    print("")

    db.close()
    print("=" * 80)
    print("Debug report complete.")
    print("=" * 80)


if __name__ == '__main__':
    database_file_path = Path(
        "/mnt/d/Projects/QI47/Jawad/mast3r/data/7/cache/cache/colmap.db"
    )
    debug_colmap_database(database_file_path)
