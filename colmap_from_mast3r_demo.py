#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Command-line reconstruction script for MASt3R-SfM (chunked aerial pipeline).
#
# This script bypasses the Gradio UI to run the reconstruction and directly
# outputs COLMAP sparse model files.
# --------------------------------------------------------
import os
import argparse
import logging
import shutil
from pathlib import Path

from mast3r.demo_glomap import get_args_parser as get_demo_args_parser, get_reconstructed_scene_J
from mast3r.model import AsymmetricMASt3R
from dust3r.demo import set_print_with_timestamp

import mast3r.utils.path_to_dust3r  # noqa

log = logging.getLogger(__name__)


def get_cli_args_parser():
    """Defines the command-line arguments for the script."""
    parser = get_demo_args_parser()

    # --- Inputs / outputs ---
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing input images (searched recursively).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the directory where COLMAP results will be saved.')

    # --- Camera model ---
    parser.add_argument('--shared_intrinsics', action='store_true',
                        help='Optimize one set of intrinsics for all cameras (single-camera strip).')

    # --- Transforms / pair selection ---
    parser.add_argument('--transforms_json', type=str, default=None,
                        help='Path to image_transforms.json. If omitted, auto-derived '
                             'from <image_dir>/../fp/image_transforms.json (audit C1).')
    parser.add_argument('--min_matches', type=int, default=15,
                        help='Minimum matches for an image pair to enter geometric '
                             'verification; pairs below this are dropped and logged (audit M10).')

    # --- Chunked matching knobs (audit M3) ---
    parser.add_argument('--conf_thr', type=float, default=1.5,
                        help='Confidence threshold applied to raw pointmap conf (audit H4).')
    parser.add_argument('--min_len_track', type=int, default=2,
                        help='Minimum track length (images). 2 keeps two-view tracks for '
                             'narrow strips; must match the mapper setting (audit H2).')
    parser.add_argument('--min_overlap_area', type=float, default=1.0,
                        help='Minimum projected tile overlap area (px^2) to accept a tile '
                             'pair candidate (audit H6).')
    parser.add_argument('--tile_pairing', type=str, default='per_a', choices=['per_a', 'all'],
                        help="Tile-pairing strategy: 'per_a' (default) matches each A tile to its "
                             "single best-overlap B tile (B may repeat, no stealing) -- fast, "
                             "near-complete coverage; 'all' is many-to-many (fully gap-free but "
                             "~3-4x slower and largely redundant after dedup).")
    parser.add_argument('--pair_diagnostics', action='store_true',
                        help="Print per-image-pair and total tile-pair counts (all vs per_a) and an "
                             "estimated coverage fraction, so you can compare cost/coverage before "
                             "committing to a long run.")
    parser.add_argument('--kpt_stride', type=int, default=8,
                        help="Spatial subsample stride (px) for SfM tie points: keep ~1 correspondence "
                             "per stride x stride cell. Cuts the dense-match/connection count ~stride^2 x "
                             "to avoid the multi-million-connection track-building blowup, with uniform "
                             "coverage. <=1 keeps all dense matches.")
    parser.add_argument('--tile_size', type=int, nargs=2, default=[512, 512],
                        metavar=('W', 'H'), help='Tile (width height) for chunked matching.')
    parser.add_argument('--overlap_px', type=int, default=128,
                        help='Minimum pixel overlap between adjacent tiles.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Tile-pair batch size for inference (tune to GPU memory).')

    # Remove Gradio-specific arguments that are no longer needed.
    for action in parser._actions:
        if action.dest in ['share', 'server_name', 'server_port', 'gradio_delete_cache', 'silent']:
            action.required = False
            action.default = argparse.SUPPRESS

    parser.prog = 'colmap_from_mast3r_demo.py'
    parser.description = 'Run chunked MASt3R SfM from the command line to generate a COLMAP sparse model.'
    return parser


def _collect_images(input_dir: Path):
    """Recursively collect image files (audit L7)."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    return sorted([str(p) for p in input_dir.rglob('*') if p.suffix.lower() in image_extensions])


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    parser = get_cli_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        log.error("Input directory not found at %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    filelist = _collect_images(input_dir)
    if len(filelist) < 2:
        log.error("Found %d images. At least 2 are required.", len(filelist))
        return

    log.info("Found %d images in %s.", len(filelist), input_dir)
    log.info("Outputs will be saved in %s.", output_dir)

    # --- Load Model ---
    weights_path = args.weights if args.weights else "naver/" + args.model_name
    log.info("Loading model from %s ...", weights_path)
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    log.info("Model loaded successfully.")

    # --- Run Reconstruction ---
    # Pass output_dir directly as outdir; the orchestrator adds a single 'cache'
    # level (audit M1, previously <output>/cache/cache).
    log.info("Starting reconstruction process...")
    try:
        scene_state, _ = get_reconstructed_scene_J(
            glomap_bin=args.glomap_bin,
            outdir=str(output_dir),
            gradio_delete_cache=False,
            model=model,
            retrieval_model=args.retrieval_model,
            device=args.device,
            silent=False,
            current_scene_state=None,
            filelist=filelist,
            transparent_cams=False,
            cam_size=0.05,
            shared_intrinsics=args.shared_intrinsics,
            transforms_json=args.transforms_json,
            conf_thr=args.conf_thr,
            min_len_track=args.min_len_track,
            tile_size=tuple(args.tile_size),
            overlap_px=args.overlap_px,
            batch_size=args.batch_size,
            min_matches=args.min_matches,
            min_overlap_area=args.min_overlap_area,
            tile_pairing=args.tile_pairing,
            pair_diagnostics=args.pair_diagnostics,
            kpt_stride=args.kpt_stride,
        )
    except Exception as e:
        log.error("An error occurred during reconstruction: %s", e)
        import traceback
        traceback.print_exc()
        return

    # Audit C3: guard against a missing scene even though the orchestrator now
    # raises on hard failures.
    if scene_state is None:
        log.error("Reconstruction produced no scene state.")
        return

    # --- Locate and Save COLMAP files ---
    # Audit C4: copy the reconstruction directory actually returned by the mapper
    # (scene_state.reconstruction_id), not a hard-coded '0'.
    colmap_recon_path = Path(scene_state.cache_dir) / "reconstruction" / str(scene_state.reconstruction_id)
    if not colmap_recon_path.is_dir():
        log.error("COLMAP reconstruction directory was not found after the process.")
        log.error("Expected path: %s", colmap_recon_path)
        return

    final_colmap_dir = output_dir / "sparse" / "0"
    if final_colmap_dir.exists():
        log.info("Removing existing sparse model at %s", final_colmap_dir)
        shutil.rmtree(final_colmap_dir)

    log.info("Copying COLMAP model from %s to %s", colmap_recon_path, final_colmap_dir)
    shutil.copytree(colmap_recon_path, final_colmap_dir)

    print("\n-------------------------------------------")
    print("Success!")
    print(f"The COLMAP sparse model has been saved to: {final_colmap_dir.resolve()}")
    print("It contains: cameras.bin, images.bin, and points3D.bin")
    print("You can now open this model with the COLMAP GUI or other tools.")
    print("-------------------------------------------")


if __name__ == '__main__':
    main()
