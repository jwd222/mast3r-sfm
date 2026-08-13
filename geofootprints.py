"""
geofootprints.py
================
Derive per-image ground **footprint polygons** directly from georeferenced
GeoTIFFs, and (optionally) feed them straight into ``tile_transform.py`` to
produce ``image_transforms.json`` -- all without a ``footprints.shp``.

Why this exists
---------------
``tile_transform.py`` computes the per-pair 2D affine (image A -> image B
pixels) from the *intersection* of two image footprints. Historically those
footprints came from ``footprints.shp``. But the footprint of a georeferenced
raster is fully determined by its own geotransform: the 4 raster corners
projected into world coordinates. So if your images are GeoTIFFs (carrying
``ModelPixelScale`` / ``ModelTiepoint`` / GeoKeys), you do not need any external
footprint file at all.

Notes
-----
* Corners are mapped with the raster's full affine transform, so this is correct
  even for rotated GeoTIFFs (it is NOT the axis-aligned ``src.bounds`` envelope).
* For near-nadir aerial frames this matches the old shp-derived footprints well.
  For strongly oblique frames the raster's single affine placement is an
  approximation; in that case footprints should instead be projected from EO.
* All images are reprojected to a single common CRS (the first image's CRS) so
  that polygon intersections are consistent across the block.
* The emitted ``id`` column holds the filename **stem** (no extension), matching
  ``tile_transform.py`` which appends ``.tif`` itself.
"""
import os
import argparse

import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
from tqdm import tqdm

# Same set of raster extensions used elsewhere in the repo.
_DEFAULT_EXTS = (".tif", ".tiff")


def footprint_from_geotiff(path):
    """Return ``(Polygon, CRS)`` for the raster's 4 corners in world coords.

    Uses the full affine geotransform, so rotated rasters get a rotated
    (non-axis-aligned) footprint -- unlike ``rasterio.Dataset.bounds``.
    """
    with rasterio.open(path) as src:
        w, h = src.width, src.height
        # (col, row) -> world via the affine. (0,0)=UL, (W,0)=UR, (W,H)=LR, (0,H)=LL.
        xs, ys = [], []
        for col, row in [(0, 0), (w, 0), (w, h), (0, h)]:
            x, y = src.transform * (col, row)
            xs.append(x)
            ys.append(y)
        return Polygon(zip(xs, ys)), src.crs


def build_footprints(image_dir, exts=_DEFAULT_EXTS, id_col="id"):
    """Build a GeoDataFrame of footprints (one row per georeferenced image)."""
    files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(exts)
    )
    if not files:
        raise FileNotFoundError(
            f"No georeferenced images matching {exts} found in {image_dir}"
        )

    rows = []
    target_crs = None
    for fn in tqdm(files, desc="Reading GeoTIFF footprints"):
        path = os.path.join(image_dir, fn)
        try:
            poly, crs = footprint_from_geotiff(path)
        except rasterio.errors.RasterioIOError as exc:
            print(f"  [skip] cannot open {fn}: {exc}")
            continue
        if crs is None:
            print(f"  [skip] {fn}: GeoTIFF has no CRS; cannot place footprint")
            continue
        if target_crs is None:
            target_crs = crs
        # Reproject to the common CRS so pairwise intersections are valid.
        if crs != target_crs:
            poly = gpd.GeoSeries([poly], crs=crs).to_crs(target_crs).iloc[0]

        rows.append({id_col: os.path.splitext(fn)[0], "geometry": poly})

    if not rows:
        raise RuntimeError("No usable georeferenced images produced footprints.")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)
    total_bounds = gdf.total_bounds
    print(
        f"Built {len(gdf)} footprints in CRS {target_crs} "
        f"(block bounds: x[{total_bounds[0]:.1f}, {total_bounds[2]:.1f}] "
        f"y[{total_bounds[1]:.1f}, {total_bounds[3]:.1f}])."
    )
    return gdf


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Derive image footprints from georeferenced GeoTIFFs (no .shp "
            "needed) and optionally build image_transforms.json."
        )
    )
    ap.add_argument("--image_dir", required=True,
                    help="Directory of georeferenced GeoTIFF images.")
    ap.add_argument("--output", required=True,
                    help="Output footprint vector (e.g. footprints.gpkg or .shp).")
    ap.add_argument("--id_col", default="id",
                    help="Column name for the image filename stem (default: 'id').")
    ap.add_argument("--transforms_json", default=None,
                    help="If set, also run tile_transform.py to produce this "
                         "image_transforms.json from the generated footprints.")
    args = ap.parse_args()

    gdf = build_footprints(args.image_dir, id_col=args.id_col)
    # Driver is inferred from extension (.gpkg / .shp / .geojson ...).
    gdf.to_file(args.output)
    print(f"Wrote footprints -> {args.output}")

    if args.transforms_json:
        # Reuse the existing, unmodified tile_transform pipeline.
        from tile_transform import main as build_transforms
        build_transforms(
            footprints_path=args.output,
            image_dir=args.image_dir,
            output_path=args.transforms_json,
            image_id_col=args.id_col,
        )
        print(f"Wrote pairwise transforms -> {args.transforms_json}")


if __name__ == "__main__":
    main()
