"""
exif_to_colmap.py
=================
Convert ODM-style aerial images into a COLMAP sparse model whose camera POSES
come directly from EXIF/XMP -- no ``footprints.shp`` and no SfM pose estimation.

Inputs per image
----------------
* Position  : GPS latitude/longitude/altitude, read from the ODM ``<img>.exif``
              JSON (standard EXIF IFD values, stripped of attitude by ODM).
* Attitude  : Yaw / Pitch / Roll, parsed from the ``Camera:`` XMP packet that
              *is* present inside the raw JPG (DroneDeploy-style tags).
* Intrinsics: focal ratio + k1/k2, from the ODM camera string / .exif JSON.

Output
------
A COLMAP text sparse model in a **local ENU metric frame** (tangent plane at the
block centroid):

    <out>/cameras.txt   SIMPLE_RADIAL: [focal_px, cx, cy, k1]  (shared camera)
    <out>/images.txt    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    <out>/points3D.txt  empty (filled later by point_triangulator)

Feed this to::

    colmap point_triangulator \
        --database_path <db> --image_path <images> \
        --input_path <out> --output_path <out_tri>

Convention (the one thing you may need to flip)
-----------------------------------------------
Aerospace ZYX / DroneDeploy, parameterised so signs can be inverted if the
reprojection check fails:

    yaw   : heading, degrees, clockwise from North (positive)
    pitch : degrees, positive = nose up
    roll  : degrees, positive = right wing down
    body  : X forward, Y right, Z down
    world : local ENU (X East, Y North, Z Up)

        R_body->ned = Rz(yaw) Ry(pitch) Rx(roll)
        R_body->enu = R_ned2enu @ R_body->ned
        R_enu->cam  = R_body->cam @ R_body->enu.T      (R_body->cam fixed below)

COLMAP camera frame is X right, Y down, Z forward (optical axis +Z). For a nadir
camera the optical axis is body-Z (down), which fixes::

        R_body->cam = [[0, 1, 0],
                       [-1,0, 0],
                       [ 0, 0, 1]]

Sanity check printed at the end: the world-space optical axis must point
*downward* (Z ~ -1) for near-nadir images, and baselines must match the GPS
deltas. If triangulation reprojection is large, flip --yaw-sign / --pitch-sign /
--roll-sign (or --invert-cam-y) until the optical axis points down and the
reprojection error is small.
"""
import os
import re
import json
import glob
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

# NED (X=North,Y=East,Z=Down) -> ENU (X=East,Y=North,Z=Up)
_R_NED2ENU = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0]])

# Body (X fwd, Y right, Z down) -> COLMAP camera (X right, Y down, Z forward).
_R_BODY2CAM = np.array([[0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# WGS84 -> local ENU tangent plane
# --------------------------------------------------------------------------- #
def lla2enu_vec(lat_deg, lon_deg, h, lat0, lon0, h0):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    rad = np.pi / 180.0
    lat = np.asarray(lat_deg) * rad
    lon = np.asarray(lon_deg) * rad
    lat0r = lat0 * rad
    lon0r = lon0 * rad

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + h) * np.sin(lat)

    N0 = a / np.sqrt(1 - e2 * np.sin(lat0r) ** 2)
    x0 = (N0 + h0) * np.cos(lat0r) * np.cos(lon0r)
    y0 = (N0 + h0) * np.cos(lat0r) * np.sin(lon0r)
    z0 = (N0 * (1 - e2) + h0) * np.sin(lat0r)

    dx, dy, dz = x - x0, y - y0, z - z0
    sl, cl, sll, cll = np.sin(lat0r), np.cos(lat0r), np.sin(lon0r), np.cos(lon0r)
    # Standard ECEF->ENU rotation about the tangent point.
    east = -sll * dx + cll * dy
    north = -sl * cll * dx - sl * sll * dy + cl * dz
    up = cl * cll * dx + cl * sll * dy + sl * dz
    return east, north, up


def lla2enu(lat_deg, lon_deg, h, lat0, lon0, h0):
    e, n, u = lla2enu_vec(lat_deg, lon_deg, h, lat0, lon0, h0)
    return np.array([float(e), float(n), float(u)])


# --------------------------------------------------------------------------- #
# EXIF (.exif JSON) + XMP (raw JPG) readers
# --------------------------------------------------------------------------- #
def read_exif(exif_path):
    with open(exif_path) as f:
        d = json.load(f)
    g = d["gps"]
    return {
        "lat": g["latitude"],
        "lon": g["longitude"],
        "alt": g["altitude"],
        "width": d["width"],
        "height": d["height"],
        "focal_ratio": d.get("focal_ratio"),
        "camera": d.get("camera"),
    }


def read_xmp_ypr(jpg_path):
    """Return (yaw, pitch, roll) in degrees from the Camera: XMP tags, or Nones."""
    txt = open(jpg_path, "rb").read().decode("latin1", errors="ignore")
    m = re.search(r"<x:xmpmeta.*?</x:xmpmeta>", txt, re.S)
    if not m:
        return None, None, None
    xmp = m.group(0)

    def grab(name):
        mm = re.search(r"<(?:Camera|exif|tiff):%s>([^<]+)</" % name, xmp)
        return float(mm.group(1)) if mm else None

    return grab("Yaw"), grab("Pitch"), grab("Roll")


# --------------------------------------------------------------------------- #
# Yaw/Pitch/Roll -> world(ENU)-to-camera rotation
# --------------------------------------------------------------------------- #
def ypr_to_world2cam(yaw, pitch, roll, *,
                     yaw_sign=1.0, pitch_sign=1.0, roll_sign=1.0,
                     order="xyz", invert_cam_y=False):
    """Aerospace ZYX. Returns R_wc (world=ENU -> COLMAP camera)."""
    # R_body->ned = Rz(yaw) Ry(pitch) Rx(roll). scipy extrinsic 'xyz' with
    # angles [roll, pitch, yaw] yields exactly Rz(yaw)Ry(pitch)Rx(roll).
    R_b2ned = Rotation.from_euler(
        order,
        [roll_sign * roll, pitch_sign * pitch, yaw_sign * yaw],
        degrees=True,
    ).as_matrix()
    R_b2enu = _R_NED2ENU @ R_b2ned
    R_b2c = _R_BODY2CAM.copy()
    if invert_cam_y:
        R_b2c = R_b2c.copy()
        R_b2c[:, 1] *= -1  # flip camera Y (image vertical) axis sign
    # world(ENU) -> cam = R_body->cam @ R_body->enu^-1 = R_b2c @ R_b2enu.T
    R_wc = R_b2c @ R_b2enu.T
    return R_wc


def rotmat_to_qvec(R):
    # COLMAP qvec = [qw, qx, qy, qz] for R_wc. scipy returns [x,y,z,w].
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True, help="Dir of JPGs (with XMP).")
    ap.add_argument("--exif_dir", default=None,
                    help="Dir of ODM <img>.exif JSON. Default: <image_dir>/../exif.")
    ap.add_argument("--output", required=True, help="Output COLMAP model dir.")
    ap.add_argument("--camera_model", default="SIMPLE_RADIAL",
                    help="COLMAP camera model (default SIMPLE_RADIAL).")
    ap.add_argument("--focal_ratio", type=float, default=None,
                    help="Override focal ratio (else read from .exif).")
    ap.add_argument("--k1", type=float, default=0.0, help="Radial k1 (default 0).")
    ap.add_argument("--yaw_sign", type=float, default=1.0)
    ap.add_argument("--pitch_sign", type=float, default=1.0)
    ap.add_argument("--roll_sign", type=float, default=1.0)
    ap.add_argument("--invert_cam_y", action="store_true")
    ap.add_argument("--shared_camera", action="store_true", default=True,
                    help="All images share one camera (default: true).")
    args = ap.parse_args()

    exif_dir = args.exif_dir or os.path.join(args.image_dir, "..", "exif")
    exif_dir = os.path.abspath(exif_dir)

    jpgs = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) +
                  glob.glob(os.path.join(args.image_dir, "*.JPG")))
    if not jpgs:
        raise SystemExit(f"No JPGs found in {args.image_dir}")

    records = []
    for jpg in jpgs:
        stem = os.path.splitext(os.path.basename(jpg))[0]
        exif_path = os.path.join(exif_dir, stem + ".jpg.exif")
        if not os.path.isfile(exif_path):  # try without .jpg in name
            exif_path = os.path.join(exif_dir, os.path.basename(jpg) + ".exif")
        if not os.path.isfile(exif_path):
            print(f"  [skip] no .exif for {os.path.basename(jpg)}")
            continue
        info = read_exif(exif_path)
        yaw, pitch, roll = read_xmp_ypr(jpg)
        if yaw is None or pitch is None or roll is None:
            print(f"  [skip] {os.path.basename(jpg)}: no Yaw/Pitch/Roll in XMP")
            continue
        fr = args.focal_ratio if args.focal_ratio is not None else info["focal_ratio"]
        records.append(dict(path=jpg, name=os.path.basename(jpg), info=info,
                            focal_ratio=fr, yaw=yaw, pitch=pitch, roll=roll))
    if not records:
        raise SystemExit("No images produced usable EO (need GPS + YPR).")

    # Local ENU tangent plane at the block centroid.
    lats = [r["info"]["lat"] for r in records]
    lons = [r["info"]["lon"] for r in records]
    alts = [r["info"]["alt"] for r in records]
    lat0, lon0, h0 = np.mean(lats), np.mean(lons), np.mean(alts)

    # Intrinsics (shared camera).
    r0 = records[0]
    W, H = r0["info"]["width"], r0["info"]["height"]
    focal_px = r0["focal_ratio"] * W
    cx, cy = W / 2.0, H / 2.0
    print(f"Camera: {args.camera_model}  {W}x{H}  focal={focal_px:.2f}px "
          f"(ratio {r0['focal_ratio']:.5f})  cx,cy=({cx:.1f},{cy:.1f})  k1={args.k1}")
    print(f"ENU origin (tangent): lat={lat0:.8f} lon={lon0:.8f} alt={h0:.3f} m\n")

    os.makedirs(args.output, exist_ok=True)
    cam_id = 1
    with open(os.path.join(args.output, "cameras.txt"), "w") as fc, \
         open(os.path.join(args.output, "images.txt"), "w") as fi:
        # Camera line
        if args.camera_model == "SIMPLE_RADIAL":
            params = f"{focal_px} {cx} {cy} {args.k1}"
        elif args.camera_model in ("PINHOLE",):
            params = f"{focal_px} {focal_px} {cx} {cy}"
        else:
            params = f"{focal_px} {cx} {cy} {args.k1}"
        fc.write(f"{cam_id} {args.camera_model} {W} {H} {params}\n")

        print(f"{'image':18s} {'ENU center (E,N,U) m':30s} {'optical axis world (E,N,U)':30s} YPR(deg)")
        prev_c = None
        for i, r in enumerate(records, start=1):
            C = lla2enu(r["info"]["lat"], r["info"]["lon"], r["info"]["alt"], lat0, lon0, h0)
            R_wc = ypr_to_world2cam(
                r["yaw"], r["pitch"], r["roll"],
                yaw_sign=args.yaw_sign, pitch_sign=args.pitch_sign,
                roll_sign=args.roll_sign, invert_cam_y=args.invert_cam_y)
            q = rotmat_to_qvec(R_wc)
            t = -R_wc @ C
            # Optical axis in world = R_wc^-1 @ [0,0,1] (direction camera looks).
            opt_world = R_wc.T @ np.array([0.0, 0.0, 1.0])

            base = ""
            if prev_c is not None:
                d = float(np.linalg.norm(C[:2] - prev_c[:2]))
                base = f"  base={d:.1f}m"
            print(f"{r['name'][:18]:18s} "
                  f"({C[0]:8.2f},{C[1]:8.2f},{C[2]:8.2f})   "
                  f"({opt_world[0]:+.2f},{opt_world[1]:+.2f},{opt_world[2]:+.2f})   "
                  f"y={r['yaw']:+6.1f} p={r['pitch']:+6.1f} r={r['roll']:+6.1f}{base}")
            prev_c = C

            fi.write(f"{i} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                     f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {cam_id} {r['name']}\n")
            fi.write("\n")  # COLMAP expects a blank line / points line per image

    # Empty points file (triangulator will fill it).
    open(os.path.join(args.output, "points3D.txt"), "w").close()

    print(f"\nWrote COLMAP model -> {args.output}")
    print("Verify: optical-axis Z should be ~ -1 (looking down). "
          "If not, or reprojection is bad after triangulation, flip signs:")
    print("  --yaw-sign -1 / --pitch-sign -1 / --roll-sign -1 / --invert_cam_y")


if __name__ == "__main__":
    main()
