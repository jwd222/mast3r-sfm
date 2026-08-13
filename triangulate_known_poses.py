"""
triangulate_known_poses.py
==========================
Geometry-only stage: given a COLMAP database that already contains keypoints +
matches (from MASt3R matching, or any matcher) and the known exterior
orientation, triangulate a metric 3D point cloud in the local ENU frame -- with
NO incremental SfM / pose estimation.

It builds an in-memory ``pycolmap.Reconstruction`` whose cameras + posed images
come straight from the EO (EXIF/XMP), with **image_ids taken from the database**
(so triangulation's match lookup by id is correct), then calls
``pycolmap.triangulate_points`` and optionally a pose-fixed bundle adjustment.

Positioning in the pipeline
---------------------------
    MASt3R matching  ->  cache/colmap.db  (keypoints + matches, two-view verified)
    exif_to_colmap / compute_eo  ->  poses + intrinsics
                \\_______________/
                       v
            triangulate_known_poses.py  ->  sparse metric model (ENU)
"""
import os
import sys
import glob
import argparse
import sqlite3

import numpy as np
import pycolmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eo_to_tilepairs import compute_eo  # noqa: E402
from exif_to_colmap import rotmat_to_qvec  # noqa: E402


def db_image_ids(database_path):
    """Return {image_name: image_id} as registered in the COLMAP database."""
    con = sqlite3.connect(database_path)
    try:
        cur = con.execute("SELECT image_id, name FROM images")
        return {name: int(iid) for iid, name in cur.fetchall()}
    finally:
        con.close()


def write_posed_model(model_dir, recs, K, W, H, *, db_name_to_id, camera_model="SIMPLE_RADIAL",
                      k1=0.0, camera_id=1):
    """Write a COLMAP text model whose image_ids match the database.

    Returns the model dir; load it with ``pycolmap.Reconstruction(model_dir)``.
    (pycolmap 4.x makes Image.cam_from_world read-only, so we author text and
    load it rather than building the reconstruction object-by-object.)
    """
    os.makedirs(model_dir, exist_ok=True)
    focal_px = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    with open(os.path.join(model_dir, "cameras.txt"), "w") as fc, \
         open(os.path.join(model_dir, "images.txt"), "w") as fi:
        if camera_model == "SIMPLE_RADIAL":
            params = f"{focal_px} {cx} {cy} {k1}"
        elif camera_model == "PINHOLE":
            params = f"{focal_px} {focal_px} {cx} {cy}"
        else:
            params = f"{focal_px} {cx} {cy} {k1}"
        fc.write(f"{camera_id} {camera_model} {W} {H} {params}\n")

        placed = 0
        for r in recs:
            iid = db_name_to_id.get(r["basename"]) or db_name_to_id.get(r["name"])
            if iid is None:
                print(f"  [skip] {r['basename']} not registered in DB (no image_id)")
                continue
            q = rotmat_to_qvec(r["R"])
            t = r["t"]
            fi.write(f"{iid} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                     f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {camera_id} {r['basename']}\n\n")
            placed += 1
    open(os.path.join(model_dir, "points3D.txt"), "w").close()
    print(f"Wrote posed model ({placed} images, camera_id={camera_id}) -> {model_dir}")
    return model_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database_path", required=True)
    ap.add_argument("--image_path", required=True, help="Directory of images.")
    ap.add_argument("--exif_dir", default=None, help="Default: <image_path>/../exif")
    ap.add_argument("--output_path", required=True, help="Output COLMAP sparse model dir.")
    ap.add_argument("--camera_model", default="SIMPLE_RADIAL")
    ap.add_argument("--k1", type=float, default=0.0)
    ap.add_argument("--refine_extrinsics", action="store_true",
                    help="Also run BA refining extrinsics (default: poses held fixed).")
    ap.add_argument("--refine_intrinsics", action="store_true",
                    help="Let triangulate_points refine intrinsics (default: fixed).")
    # EO convention knobs (forwarded to compute_eo -> exif_to_colmap).
    ap.add_argument("--yaw_sign", type=float, default=1.0)
    ap.add_argument("--pitch_sign", type=float, default=1.0)
    ap.add_argument("--roll_sign", type=float, default=1.0)
    ap.add_argument("--invert_cam_y", action="store_true")
    args = ap.parse_args()

    exif_dir = args.exif_dir or os.path.abspath(os.path.join(args.image_path, "..", "exif"))
    recs, K, enu0, W, H = compute_eo(
        args.image_path, exif_dir,
        yaw_sign=args.yaw_sign, pitch_sign=args.pitch_sign,
        roll_sign=args.roll_sign, invert_cam_y=args.invert_cam_y)

    name_to_id = db_image_ids(args.database_path)
    print(f"Database has {len(name_to_id)} registered images.")

    posed_model_dir = os.path.join(args.output_path, "_poses_input")
    write_posed_model(posed_model_dir, recs, K, W, H, db_name_to_id=name_to_id,
                      camera_model=args.camera_model, k1=args.k1)
    recon = pycolmap.Reconstruction(posed_model_dir)

    os.makedirs(args.output_path, exist_ok=True)
    recon = pycolmap.triangulate_points(
        recon, args.database_path, args.image_path, args.output_path,
        clear_points=True, refine_intrinsics=args.refine_intrinsics)
    print(f"After triangulation: {recon.num_points3D()} points, "
          f"{sum(1 for i in recon.images.values() if i.has_pose)} posed images.")

    if args.refine_extrinsics:
        opts = pycolmap.BundleAdjustmentOptions()
        opts.refine_extrinsics = True
        pycolmap.bundle_adjustment(recon, opts)
        print("Bundle adjustment (refining extrinsics) done.")

    # Pose-fixed refinement of 3D points (default): polish points only.
    if not args.refine_extrinsics:
        opts = pycolmap.BundleAdjustmentOptions()
        opts.refine_extrinsics = False
        opts.refine_focal_length = False
        opts.refine_extra_params = False
        pycolmap.bundle_adjustment(recon, opts)
        print("Bundle adjustment (points only, poses fixed) done.")

    recon.write(args.output_path)  # writes cameras3.bin/images.bin/points3D.bin + txt

    # Quick non-flatness / metric report.
    if recon.num_points3D() > 0:
        xyz = np.array([p.xyz for p in recon.points3D().values()])
        print("\n=== 3D cloud (ENU, meters) ===")
        print(f"  N points : {len(xyz)}")
        print(f"  X range  : [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]  span {xyz[:,0].ptp():.2f}")
        print(f"  Y range  : [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]  span {xyz[:,1].ptp():.2f}")
        print(f"  Z range  : [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]  span {xyz[:,2].ptp():.2f}  std {xyz[:,2].std():.2f}")
        mean_reproj = np.mean([i.mean_reprojection_error for i in recon.images.values()
                               if i.has_pose()]) if hasattr(next(iter(recon.images.values())), "mean_reprojection_error") else None
        if mean_reproj is not None:
            print(f"  mean reprojection error: {mean_reproj:.3f} px")
    print(f"\nWrote sparse model -> {args.output_path}")


if __name__ == "__main__":
    main()
