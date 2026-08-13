"""
eo_to_tilepairs.py
==================
Derive per-pair image-to-image transforms **directly from exterior orientation
+ intrinsics** -- no footprints, no georeferenced rasters, no transforms.json.

This is the EO analogue of ``tile_transform.py``: instead of intersecting ground
footprints, we compute the **ground-plane homography** that maps pixels of image
A into pixels of image B from the relative pose. The chunked MASt3R matcher then
uses it (via ``determine_overlapping_tile_pairs``) to pick which tiles to match,
exactly as before -- only the source of the transform changed.

Pipeline placement
------------------
Everything up to and including matching stays in **image/pixel space**; the lift
to world (ENU) coordinates happens later at ``point_triangulator`` time, using the
EO model from ``exif_to_colmap.py``.

Math (plane-induced homography, COLMAP world->cam convention)
-------------------------------------------------------------
For a ground plane  n_world^T X = d  (n_world = ENU up = [0,0,1], d = ground_Z):

    n_A   = R_A · n_world                       (plane normal in A's cam frame)
    d_A   = d + n_A · t_A                       (plane offset in A's cam frame)
    R_rel = R_B · R_A^T ,   t_rel = t_B - R_rel · t_A
    H~    = R_rel + (t_rel · n_A^T) / d_A       (normalised A-cam -> B-cam)
    H_AB  = K_B · H~ · K_A^-1                   (pixel A -> pixel B)

Two output forms (``--mode``):
  * ``homography`` : full 3x3 H (rigorous; needs the extended tile selector that
                     uses ``cv2.perspectiveTransform``).
  * ``affine``     : 2x3 fitted to the 4 mapped corners (drop-in for the existing
                     ``cv2.transform``-based selector). Accurate for near-nadir.

The single tunable is ``ground_Z`` (the homography's parallax term). It only
affects *which tiles get paired* and is robust to a rough guess; set it to the
mean terrain height in the local ENU frame (cameras fly above it, so typically a
negative value relative to the ENU origin = camera-altitude centroid).
"""
import os
import re
import json
import glob
import argparse
from itertools import combinations

import numpy as np
import cv2

from exif_to_colmap import (
    read_exif,
    read_xmp_ypr,
    lla2enu,
    ypr_to_world2cam,
)


def _actual_image_size(path):
    """Read the real pixel dimensions of an image (EXIF width/height can differ)."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (width, height)


def compute_eo(image_dir, exif_dir, *, yaw_sign=1.0, pitch_sign=1.0, roll_sign=1.0,
               invert_cam_y=False, focal_ratio_override=None):
    """Return (eo_records, K, enu_origin, W, H) for all images with usable EO.

    Each record carries the COLMAP world(ENU)->cam rotation ``R``, translation
    ``t`` and camera center ``C`` (all in the local ENU frame), plus image size.

    ``W``/``H`` and ``K`` use the image's **actual** pixel dimensions (read from
    the file), not the EXIF width/height, so the intrinsics match the keypoints'
    pixel space in the COLMAP database.
    """
    jpgs = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                  glob.glob(os.path.join(image_dir, "*.JPG")))
    if not jpgs:
        raise FileNotFoundError(f"No JPGs in {image_dir}")

    recs = []
    for jpg in jpgs:
        stem = os.path.splitext(os.path.basename(jpg))[0]
        exif_path = os.path.join(exif_dir, os.path.basename(jpg) + ".exif")
        if not os.path.isfile(exif_path):
            exif_path = os.path.join(exif_dir, stem + ".jpg.exif")
        if not os.path.isfile(exif_path):
            print(f"  [skip] no .exif for {os.path.basename(jpg)}")
            continue
        info = read_exif(exif_path)
        yaw, pitch, roll = read_xmp_ypr(jpg)
        if yaw is None or pitch is None or roll is None:
            print(f"  [skip] {os.path.basename(jpg)}: no Yaw/Pitch/Roll in XMP")
            continue
        fr = focal_ratio_override if focal_ratio_override is not None else info["focal_ratio"]
        W_actual, H_actual = _actual_image_size(jpg)
        recs.append(dict(name=stem, basename=os.path.basename(jpg), info=info,
                         focal_ratio=fr, yaw=yaw, pitch=pitch, roll=roll,
                         width=W_actual, height=H_actual))

    if not recs:
        raise RuntimeError("No images with usable EO (need GPS + YPR).")

    lats = [r["info"]["lat"] for r in recs]
    lons = [r["info"]["lon"] for r in recs]
    alts = [r["info"]["alt"] for r in recs]
    lat0, lon0, h0 = float(np.mean(lats)), float(np.mean(lons)), float(np.mean(alts))

    for r in recs:
        r["C"] = lla2enu(r["info"]["lat"], r["info"]["lon"], r["info"]["alt"], lat0, lon0, h0)
        r["R"] = ypr_to_world2cam(r["yaw"], r["pitch"], r["roll"],
                                  yaw_sign=yaw_sign, pitch_sign=pitch_sign,
                                  roll_sign=roll_sign, invert_cam_y=invert_cam_y)
        r["t"] = -r["R"] @ r["C"]

    # Shared camera: focal ratio is defined relative to the stored image width, so
    # scale it to the actual pixel width.
    W, H = recs[0]["width"], recs[0]["height"]
    focal_px = recs[0]["focal_ratio"] * W
    K = np.array([[focal_px, 0, W / 2.0], [0, focal_px, H / 2.0], [0, 0, 1.0]], dtype=float)
    return recs, K, (lat0, lon0, h0), W, H


def pair_homography(A, B, K, ground_z):
    """3x3 ground-plane homography mapping pixel(A) -> pixel(B)."""
    n_world = np.array([0.0, 0.0, 1.0])
    nA = A["R"] @ n_world
    dA = float(ground_z + nA @ A["t"])
    if abs(dA) < 1e-9:
        raise ValueError("Degenerate plane (camera on the ground plane); pick another ground_z.")
    R_rel = B["R"] @ A["R"].T
    t_rel = B["t"] - R_rel @ A["t"]
    H_tilde = R_rel + np.outer(t_rel, nA) / dA
    return K @ H_tilde @ np.linalg.inv(K)


def homography_to_affine(H, W, H_img, n_pts=8):
    """Reduce a 3x3 homography to a 2x3 affine by sampling the image extent."""
    xs = np.linspace(0, W, int(np.sqrt(n_pts)))
    ys = np.linspace(0, H_img, int(np.sqrt(n_pts)))
    src, dst = [], []
    for x in xs:
        for y in ys:
            v = H @ np.array([x, y, 1.0])
            if abs(v[2]) < 1e-9:
                continue
            src.append([x, y])
            dst.append([v[0] / v[2], v[1] / v[2]])
    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)
    aff, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
    return aff  # 2x3


def corners_in_B(H, W, H_img):
    """Map A's 4 corners through H into B's pixel space (Nx2 array)."""
    corners = np.array([[0, 0, 1], [W, 0, 1], [W, H_img, 1], [0, H_img, 1]], float).T
    m = H @ corners
    return (m[:2] / m[2]).T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--exif_dir", default=None,
                    help="Default: <image_dir>/../exif")
    ap.add_argument("--output", required=True, help="Output image_transforms.json path.")
    ap.add_argument("--ground_z", type=float, required=True,
                    help="Ground-plane height in local ENU meters (cameras are above it; "
                         "typically negative relative to the camera-altitude centroid).")
    ap.add_argument("--mode", choices=["homography", "affine"], default="affine",
                    help="Output form: 'homography' (3x3) or 'affine' (2x3, default).")
    ap.add_argument("--yaw_sign", type=float, default=1.0)
    ap.add_argument("--pitch_sign", type=float, default=1.0)
    ap.add_argument("--roll_sign", type=float, default=1.0)
    ap.add_argument("--invert_cam_y", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="Map every pair's corners and report overlap diagnostics.")
    args = ap.parse_args()

    exif_dir = args.exif_dir or os.path.abspath(os.path.join(args.image_dir, "..", "exif"))
    recs, K, enu0, W, H = compute_eo(
        args.image_dir, exif_dir,
        yaw_sign=args.yaw_sign, pitch_sign=args.pitch_sign,
        roll_sign=args.roll_sign, invert_cam_y=args.invert_cam_y)
    print(f"EO for {len(recs)} images. ENU origin (lat,lon,alt)={enu0}")
    print(f"Mode: {args.mode}  ground_z={args.ground_z} m  K focal={K[0,0]:.1f}px  image={W}x{H}")

    def invert(mat):
        m = mat if mat.shape == (3, 3) else np.vstack([mat, [0, 0, 1]])
        return np.linalg.inv(m)[:2] if mat.shape == (2, 3) else np.linalg.inv(m)

    transforms = {}
    for A, B in combinations(recs, 2):
        H_AB = pair_homography(A, B, K, args.ground_z)
        mat = H_AB if args.mode == "homography" else homography_to_affine(H_AB, W, H)
        mat = np.asarray(mat)
        transforms[f"{A['name']}__{B['name']}"] = mat.tolist()
        transforms[f"{B['name']}__{A['name']}"] = invert(mat).tolist()

        if args.verify:
            cb = corners_in_B(H_AB, W, H)
            inside = sum(0 <= x < W and 0 <= y < H for x, y in cb)
            print(f"  {A['name']}__{B['name']}: corners in B = "
                  f"[{', '.join(f'({x:.0f},{y:.0f})' for x, y in cb)}]  "
                  f"{inside}/4 inside")

    with open(args.output, "w") as f:
        json.dump(transforms, f, indent=4)
    print(f"\nWrote {len(transforms)} transforms ({args.mode}) -> {args.output}")


if __name__ == "__main__":
    main()
