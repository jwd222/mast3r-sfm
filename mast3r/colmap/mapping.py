# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# colmap mapper/colmap point_triangulator/glomap mapper from mast3r matches
# --------------------------------------------------------
import pycolmap
import os
import os.path as path
import time
import kapture.io
import kapture.io.csv
import subprocess
import PIL
from tqdm import tqdm
import PIL.Image
import numpy as np
from typing import List, Tuple, Union

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.database import export_matches, get_im_matches

import mast3r.utils.path_to_dust3r  # noqa
from dust3r_visloc.datasets.utils import get_resize_function

import kapture
from kapture.converter.colmap.database_extra import get_colmap_camera_ids_from_db, get_colmap_image_ids_from_db
from kapture.utils.paths import path_secure

from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.inference import inference


def scene_prepare_images(root: str, maxdim: int, patch_size: int, image_paths: List[str]):
    images = []
    # image loading
    for idx in tqdm(range(len(image_paths))):
        rgb_image = PIL.Image.open(os.path.join(root, image_paths[idx])).convert('RGB')

        # resize images
        W, H = rgb_image.size
        resize_func, _, to_orig = get_resize_function(maxdim, patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))

        # image dictionary
        images.append({'img': rgb_tensor.unsqueeze(0),
                       'true_shape': np.int32([rgb_tensor.shape[1:]]),
                       'to_orig': to_orig,
                       'idx': idx,
                       'instance': image_paths[idx],
                       'orig_shape': np.int32([H, W])})
    return images


def remove_duplicates(images, image_pairs):
    pairs_added = set()
    pairs = []
    for (i, _), (j, _) in image_pairs:
        smallidx, bigidx = min(i, j), max(i, j)
        if (smallidx, bigidx) in pairs_added:
            continue
        pairs_added.add((smallidx, bigidx))
        pairs.append((images[i], images[j]))
    return pairs


def run_mast3r_matching(model: AsymmetricMASt3R, maxdim: int, patch_size: int, device,
                        kdata: kapture.Kapture, root_path: str, image_pairs_kapture: List[Tuple[str, str]],
                        colmap_db,
                        dense_matching: bool, pixel_tol: int, conf_thr: float, skip_geometric_verification: bool,
                        min_len_track: int):
    assert kdata.records_camera is not None
    image_paths = kdata.records_camera.data_list()
    image_path_to_idx = {image_path: idx for idx, image_path in enumerate(image_paths)}
    image_path_to_ts = {kdata.records_camera[ts, camid]: (ts, camid) for ts, camid in kdata.records_camera.key_pairs()}

    images = scene_prepare_images(root_path, maxdim, patch_size, image_paths)
    image_pairs = [((image_path_to_idx[image_path1], image_path1), (image_path_to_idx[image_path2], image_path2))
                   for image_path1, image_path2 in image_pairs_kapture]
    matching_pairs = remove_duplicates(images, image_pairs)

    colmap_camera_ids = get_colmap_camera_ids_from_db(colmap_db, kdata.records_camera)
    colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)
    im_keypoints = {idx: {} for idx in range(len(image_paths))}

    im_matches = {}
    image_to_colmap = {}
    for image_path, idx in image_path_to_idx.items():
        _, camid = image_path_to_ts[image_path]
        colmap_camid = colmap_camera_ids[camid]
        colmap_imid = colmap_image_ids[image_path]
        image_to_colmap[idx] = {
            'colmap_imid': colmap_imid,
            'colmap_camid': colmap_camid
        }

    # compute 2D-2D matching from dust3r inference
    for chunk in tqdm(range(0, len(matching_pairs), 8)):
        pairs_chunk = matching_pairs[chunk:chunk + 8] #Jawad: why are we chunking by 4 here?
        output = inference(pairs_chunk, model, device, batch_size=1, verbose=False)
        pred1, pred2 = output['pred1'], output['pred2']
        # TODO handle caching
        #Jawad: rewrite this function with multiple threads/processes
        im_images_chunk = get_im_matches(pred1=pred1, pred2=pred2, pairs=pairs_chunk, image_to_colmap=image_to_colmap,
                                         im_keypoints=im_keypoints, conf_thr=conf_thr, is_sparse=not dense_matching,
                                         pixel_tol=pixel_tol)
        im_matches.update(im_images_chunk.items())

    # filter matches, convert them and export keypoints and matches to colmap db
    colmap_image_pairs = export_matches(
        colmap_db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification)
    colmap_db.commit()

    return colmap_image_pairs


def pycolmap_run_triangulator(colmap_db_path, prior_recon_path, recon_path, image_root_path):
    print("running mapping")
    reconstruction = pycolmap.Reconstruction(prior_recon_path)
    pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        refine_intrinsics=False,
    )


def pycolmap_run_mapper(colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    reconstructions = pycolmap.incremental_mapping(
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        options=pycolmap.IncrementalPipelineOptions({'multiple_models': False,
                                                     'extract_colors': True,
                                                     })
    )


def _glomap_run_mapper(glomap_bin, colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    args = [
        'mapper',
        '--database_path',
        colmap_db_path,
        '--image_path',
        image_root_path,
        '--output_path',
        recon_path
    ]
    args.insert(0, glomap_bin)
    glomap_process = subprocess.Popen(args)
    glomap_process.wait()

    if glomap_process.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {glomap_process.returncode} )')
        
def glomap_run_mapper(glomap_bin, colmap_db_path, recon_path, image_root_path, options=None, timeout=None):
    """
    Runs the GLOMAP mapper with added flexibility to pass custom options.

    Args:
        glomap_bin (str): Path to the GLOMAP executable.
        colmap_db_path (str): Path to the COLMAP database.
        recon_path (str): Path to the desired output reconstruction directory.
        image_root_path (str): Path to the root directory of the images.
        options (dict, optional): A dictionary of additional command-line options
                                  to pass to GLOMAP, e.g.,
                                  {"--TrackEstablishment.min_track_length": "2"}.
                                  Defaults to None.
        timeout (float, optional): Maximum wall-clock seconds to wait for GLOMAP.
                                   If exceeded the process is killed and a
                                   ``TimeoutError`` is raised. ``None`` waits forever
                                   (audit M9).
    """
    print("running mapping with custom options...")

    # --- Base arguments required for any run ---
    args = [
        glomap_bin,
        'mapper',
        '--database_path',
        colmap_db_path,
        '--image_path',
        image_root_path,
        '--output_path',
        recon_path
    ]

    # --- Append the custom options ---
    # The `options` dictionary allows for flexible configuration.
    if options:
        for option, value in options.items():
            args.append(option)
            if value is not None and str(value) != "":
                args.append(str(value))

    print(f"Executing GLOMAP with command: {' '.join(args)}")

    # --- Run the subprocess ---
    glomap_process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert glomap_process.stdout is not None

    # Print output in real-time, with an optional hard timeout (audit M9).
    deadline = time.monotonic() + timeout if timeout is not None else None
    while True:
        output = glomap_process.stdout.readline()
        if output == '' and glomap_process.poll() is not None:
            break
        if output:
            print(output.strip())
        if deadline is not None and time.monotonic() > deadline:
            glomap_process.kill()
            raise TimeoutError(
                f'\nGLOMAP exceeded the timeout of {timeout}s and was killed.')

    return_code = glomap_process.poll()

    if return_code != 0:
        raise ValueError(
            f'\nGLOMAP Subprocess Error (Return code: {return_code})'
        )


def colmap_run_incremental_mapper(
    database_path: str,
    image_path: str,
    output_path: str,
    custom_options: dict = None # Allows for advanced user overrides
):
    """
    Runs the COLMAP incremental mapper with a configuration highly optimized
    for challenging, low-parallax aerial datasets.

    Args:
        database_path (str): Path to the COLMAP database file.
        image_path (str): Path to the root directory containing the images.
        output_path (str): Path to the directory where the reconstruction will be saved.
        custom_options (dict, optional): Dictionary to override any specific option.

    Returns:
        tuple: (reconstruction, reconstruction_id) where reconstruction_id is the
               on-disk sub-directory name (e.g. '0') of the returned reconstruction.
               Callers should copy ``output_path/reconstruction_id`` rather than a
               hard-coded ``reconstruction/0`` (see audit C4).
    """
    print("Running COLMAP's incremental mapper with optimized aerial settings...")

    # --- Step 1: Instantiate the top-level options class ---
    # This is the main configuration object for the entire pipeline.
    options = pycolmap.IncrementalPipelineOptions()

    # Only ever write a single reconstruction to disk. Combined with returning the
    # actual reconstruction id (below) this removes the ambiguity where the largest
    # model could be written to reconstruction/1 while callers copied reconstruction/0.
    options.multiple_models = False

    # --- Step 2: Set the nested Triangulation options ---
    # These options control how 3D points are created from 2D matches.

    # CRITICAL FIX 1: Allow two-view tracks.
    # The default (`True`) ignores all matches from a 2-view scene. We must set it to `False`.
    # NOTE: this must stay consistent with `min_len_track` in the chunked matcher (audit H2).
    # With min_len_track=2 two-view tracks exist in the DB, so this flag must remain False.
    options.triangulation.ignore_two_view_tracks = False

    # CRITICAL FIX 2: Lower the minimum triangulation angle.
    # The default (1.5 degrees) is too strict for low-parallax aerial data.
    options.triangulation.min_angle = 0.005

    # --- Step 3: Set the nested Mapper options ---
    # These options control the main SfM process: initialization, filtering, etc.

    # Also lower the post-bundle-adjustment filter to be consistent with the triangulation setting.
    options.mapper.filter_min_tri_angle = 0.005

    # Increase the initial error tolerance to help find a good starting pair with our high-res images.
    options.mapper.init_max_error = 12.0

    # Increase the final reprojection error filter to avoid discarding good points.
    options.mapper.filter_max_reproj_error = 12.0

    # --- Step 4: Handle custom user overrides (for advanced use) ---
    if custom_options:
        options.mergedict(custom_options)

    # --- Step 5: Print a summary and run the mapper ---
    print("Using the following optimized pipeline options:")
    # The .summary() method provides a clean, readable output of all settings.
    print(options.summary())

    # The pycolmap documentation uses `incremental_mapping`. Let's use that.
    # This function takes the paths and the fully configured options object.
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=options
    )

    if not reconstructions:
        raise RuntimeError("Incremental mapping failed to produce a reconstruction.")

    print(f"Successfully created {len(reconstructions)} reconstruction(s).")

    # Find and return the largest reconstruction (most images registered).
    largest_recon_id = max(reconstructions, key=lambda rid: reconstructions[rid].num_reg_images())

    return reconstructions[largest_recon_id], largest_recon_id


def kapture_import_image_folder_or_list(images_path: Union[str, Tuple[str, List[str]]], use_single_camera=False) -> kapture.Kapture:
    """
    Build a kapture dataset from an image folder or an explicit (root, list) pair.

    Note on ``use_single_camera`` (the ``--shared_intrinsics`` path, audit L9):
    when False (default) a *separate* camera is created for every image, which
    over-parameterizes intrinsics for a single-camera aerial strip. Pass
    ``use_single_camera=True`` to share one camera across all images of identical
    dimensions; heterogeneous resolutions then raise an AssertionError.
    """
    images = kapture.RecordsCamera()

    if isinstance(images_path, str):
        images_root = images_path
        file_list = [path.relpath(path.join(dirpath, filename), images_root)
                     for dirpath, dirs, filenames in os.walk(images_root)
                     for filename in filenames]
        file_list = sorted(file_list)
    else:
        images_root, file_list = images_path

    sensors = kapture.Sensors()
    for n, filename in enumerate(file_list):
        # test if file is a valid image
        try:
            # lazy load
            with PIL.Image.open(path.join(images_root, filename)) as im:
                width, height = im.size
                model_params = [width, height]
        except (OSError, PIL.UnidentifiedImageError):
            # It is not a valid image: skip it
            print(f'Skipping invalid image file {filename}')
            continue

        camera_id = f'sensor'
        if use_single_camera and camera_id not in sensors:
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)
        elif use_single_camera:
            assert sensors[camera_id].camera_params[0] == width and sensors[camera_id].camera_params[1] == height
        else:
            camera_id = camera_id + f'{n}'
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)

        images[(n, camera_id)] = path_secure(filename)  # don't forget windows

    return kapture.Kapture(sensors=sensors, records_camera=images)
