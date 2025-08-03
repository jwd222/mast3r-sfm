#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Command-line reconstruction script for MASt3R-SfM.
#
# This script bypasses the Gradio UI to run the reconstruction
# and directly outputs COLMAP sparse model files.
# --------------------------------------------------------
import os
import torch
import argparse
import shutil
from pathlib import Path

# --- Important: Make sure the mast3r library is in the Python path
# This is usually handled by your environment setup.
from mast3r.demo_glomap import get_args_parser as get_demo_args_parser, get_reconstructed_scene_J
from mast3r.model import AsymmetricMASt3R
from dust3r.demo import set_print_with_timestamp

# Add project-specific imports to the path
import mast3r.utils.path_to_dust3r  # noqa

def get_cli_args_parser():
    """Defines the command-line arguments for the script."""
    # Inherit arguments from the original demo parser
    parser = get_demo_args_parser()

    # --- Add CLI-specific arguments ---
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the directory where COLMAP results will be saved.')
    
    # --- Expose arguments that were previously UI elements ---
    parser.add_argument('--scenegraph_type', type=str, default='swin',
                        choices=['complete', 'retrieval', 'swin', 'logwin', 'oneref'],
                        help='Strategy for selecting image pairs for matching.')
    parser.add_argument('--winsize', type=int, default=3,
                        help='Window size for "swin" or "logwin" scenegraphs. For "retrieval", this is the number of neighbors (Na).')
    parser.add_argument('--win_cyclic', action='store_true',
                        help='Use cyclic window for "swin" or "logwin" scenegraphs.')
    parser.add_argument('--refid', type=int, default=0,
                        help='Reference image index for "oneref" scenegraph. For "retrieval", this is k-nn.')
    parser.add_argument('--shared_intrinsics', action='store_true',
                        help='Optimize one set of intrinsics for all cameras.')
    
    # Remove Gradio-specific arguments that are no longer needed
    for action in parser._actions:
        if action.dest in ['share', 'server_name', 'server_port', 'gradio_delete_cache', 'silent']:
            action.required = False
            action.default = argparse.SUPPRESS

    parser.prog = 'run_reconstruction_cli.py'
    parser.description = 'Run MASt3R SfM from the command line to generate a COLMAP sparse model.'
    
    return parser

def main():
    """Main execution function."""
    parser = get_cli_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    # --- 1. Setup paths and inputs ---
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.is_dir():
        print(f"Error: Input directory not found at {input_dir}")
        return

    # Create a cache directory inside the main output directory for intermediate files
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files in the input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    filelist = sorted([str(p) for p in input_dir.iterdir() if p.suffix.lower() in image_extensions])

    if len(filelist) < 2:
        print(f"Error: Found {len(filelist)} images. At least 2 are required.")
        return

    print(f"Found {len(filelist)} images in {input_dir}.")
    print(f"Outputs will be saved in {output_dir}.")

    # --- 2. Load Model ---
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
        
    print(f"Loading model from {weights_path}...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    print("Model loaded successfully.")

    # --- 3. Run Reconstruction ---
    print("\nStarting reconstruction process...")
    try:
        # This is the core function from the original script
        scene_state, _ = get_reconstructed_scene_J(
            glomap_bin=args.glomap_bin,
            outdir=str(cache_dir),
            gradio_delete_cache=False,  # Ensure cache is not deleted
            model=model,
            retrieval_model=args.retrieval_model,
            device=args.device,
            silent=False, # We want to see the logs
            image_size=args.image_size,
            current_scene_state=None,  # No UI state
            filelist=filelist,
            # These were UI inputs, now provided as args
            scenegraph_type=args.scenegraph_type,
            winsize=args.winsize,
            win_cyclic=args.win_cyclic,
            refid=args.refid,
            shared_intrinsics=args.shared_intrinsics,
            # Dummy values for unused UI parameters
            transparent_cams=False,
            cam_size=0.05
        )
        print("Reconstruction function finished.")

    except Exception as e:
        print(f"\nAn error occurred during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Locate and Save COLMAP files ---
    # The COLMAP model is saved in `<cache_dir>/reconstruction/0`
    colmap_recon_path = Path(scene_state.cache_dir) / "reconstruction" / "0"

    if not colmap_recon_path.is_dir():
        print("\nError: COLMAP reconstruction directory was not found after the process.")
        print(f"Expected path: {colmap_recon_path}")
        return
        
    # Define the final destination for the sparse model
    final_colmap_dir = output_dir / "sparse" / "0"
    if final_colmap_dir.exists():
        print(f"Removing existing sparse model at {final_colmap_dir}")
        shutil.rmtree(final_colmap_dir)
        
    print(f"Copying COLMAP model from {colmap_recon_path} to {final_colmap_dir}")
    shutil.copytree(colmap_recon_path, final_colmap_dir)

    print("\n-------------------------------------------")
    print("Success!")
    print(f"The COLMAP sparse model has been saved to: {final_colmap_dir.resolve()}")
    print("It contains: cameras.bin, images.bin, and points3D.bin")
    print("You can now open this model with the COLMAP GUI or other tools.")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()