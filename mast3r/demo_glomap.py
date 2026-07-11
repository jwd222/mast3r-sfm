#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Reconstruction entry points (chunked MASt3R -> COLMAP).
# --------------------------------------------------------
import pycolmap
import os
import json
import shutil
import logging
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import PIL.Image

from kapture.converter.colmap.database_extra import kapture_to_colmap, get_colmap_image_ids_from_db
from kapture.converter.colmap.database import COLMAPDatabase

from mast3r.colmap.mapping import kapture_import_image_folder_or_list, colmap_run_incremental_mapper
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import mast3r.utils.path_to_dust3r  # noqa

from chunked_utils import run_chunked_mast3r_matching

log = logging.getLogger(__name__)


class GlomapRecon:
    def __init__(self, world_to_cam, intrinsics, points3d, imgs):
        self.world_to_cam = world_to_cam
        self.intrinsics = intrinsics
        self.points3d = points3d
        self.imgs = imgs


class GlomapReconState:
    def __init__(self, glomap_recon, should_delete=False, cache_dir=None, outfile_name=None,
                 reconstruction_id=0):
        self.glomap_recon = glomap_recon
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete
        # On-disk reconstruction sub-directory name (e.g. '0') actually used by the
        # mapper. Callers must copy cache_dir/reconstruction/<reconstruction_id>
        # rather than a hard-coded '0' (audit C4).
        self.reconstruction_id = reconstruction_id

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')
    parser.add_argument('--glomap_bin', default='glomap', type=str, help='glomap bin')
    parser.add_argument('--retrieval_model', default=None, type=str, help="retrieval_model to be loaded")

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    parser.prog = 'mast3r demo'
    return parser


def get_reconstructed_scene_J(glomap_bin, outdir, gradio_delete_cache, model, retrieval_model, device, silent,
                              current_scene_state, filelist, transparent_cams, cam_size, shared_intrinsics,
                              transforms_json=None, conf_thr=1.5, min_len_track=2,
                              tile_size=(512, 512), overlap_px=128, batch_size=4,
                              min_matches=15, min_overlap_area=1.0, dedup_distance=2.0,
                              tile_pairing='per_a', pair_diagnostics=False, kpt_stride=8, **kw):
    """
    Chunked MASt3R -> COLMAP reconstruction orchestrator.

    Raises ``RuntimeError`` on hard failures (chunked matching error, mapper
    failure, or a 0-point reconstruction) instead of returning ``(None, None)``
    so callers cannot dereference a missing scene (audit C3).

    Defaults reflect audit decisions: ``conf_thr`` 1.5 (raw pointmap conf),
    ``min_len_track`` 2 (consistent with the mapper's
    ``ignore_two_view_tracks=False``), ``min_overlap_area`` gates tile-pair
    candidates, ``tile_pairing`` selects best-B-per-A (default, fast) vs
    many-to-many (audit H6). All are overridable via the CLI entry script.
    """
    cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Setup paths and load pre-computed transforms.
    root_path = os.path.commonpath(filelist)

    # Audit C1: make the transforms JSON configurable. Use an explicit path if
    # provided, otherwise auto-derive it relative to the image root, and fail
    # loudly if it cannot be found (instead of a hard-coded absolute path).
    if transforms_json:
        transforms_json_path = transforms_json
    else:
        transforms_json_path = os.path.join(root_path, '..', 'fp', 'image_transforms.json')
    if not os.path.isfile(transforms_json_path):
        raise FileNotFoundError(
            f"Transforms JSON not found at {transforms_json_path}. Pass --transforms_json "
            f"explicitly, or place image_transforms.json at <image_dir>/../fp/.")

    with open(transforms_json_path, 'r') as f:
        transform_data_loaded = json.load(f)

    # Audit M5: robust pair-key parsing. Keys are 'name1__name2'; reject any key
    # that does not split into exactly two parts (e.g. a basename containing '__').
    precomputed_transforms = {}
    for key, value in transform_data_loaded.items():
        parts = key.split('__')
        if len(parts) != 2:
            raise ValueError(
                f"Malformed transform key {key!r}: expected exactly one '__' separator.")
        precomputed_transforms[tuple(parts)] = np.array(value)

    # 2. Generate the list of relative and absolute image pairs from the transforms.
    filelist_relpath = [os.path.relpath(f, root_path).replace('\\', '/') for f in filelist]
    basename_to_relpath_map = {os.path.splitext(os.path.basename(p))[0]: p for p in filelist_relpath}
    # Deduplicate to UNDIRECTED pairs. tile_transform.py stores every pair in both
    # directions (A__B and B__A), and MASt3R matching is symmetric, so processing
    # both would repeat every inference. This halves the loop iterations (44 -> 22
    # for the tobias/1 dataset) with no loss of correspondences.
    image_pairs_rel = []
    seen_pairs = set()
    for name1, name2 in precomputed_transforms.keys():
        if name1 in basename_to_relpath_map and name2 in basename_to_relpath_map:
            rp1, rp2 = basename_to_relpath_map[name1], basename_to_relpath_map[name2]
            pair_key = tuple(sorted((rp1, rp2)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            image_pairs_rel.append((rp1, rp2))
    image_pairs_abs = [(os.path.join(root_path, p1), os.path.join(root_path, p2)) for p1, p2 in image_pairs_rel]
    log.info("Matching %d unique image pair(s) (deduplicated from transforms JSON).", len(image_pairs_rel))

    # 3. Setup Kapture and COLMAP database.
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)
    colmap_db_path = os.path.join(cache_dir, 'colmap.db')
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)
    os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)

    # Audit H7: try/finally guarantees the connection is always closed, even when
    # matching raises (previously the library called exit(1), audit C2).
    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)

        indexed_matches = run_chunked_mast3r_matching(
            model=model,
            device=device,
            image_pairs_to_match=image_pairs_abs,
            root_path=root_path,
            colmap_db=colmap_db,
            precomputed_transforms=precomputed_transforms,
            conf_thr=conf_thr,
            dedup_distance=dedup_distance,
            min_len_track=min_len_track,
            min_overlap_area=min_overlap_area,
            tile_pairing=tile_pairing,
            pair_diagnostics=pair_diagnostics,
            kpt_stride=kpt_stride,
            tile_size=tile_size,
            overlap_px=overlap_px,
            batch_size=batch_size,
        )
    except Exception as e:
        raise RuntimeError(f"Error during chunked matching: {e}") from e
    finally:
        colmap_db.close()

    if not indexed_matches:
        raise RuntimeError("Chunked matching resulted in no valid matches.")

    # 4. Create pairs.txt for pycolmap verification.
    # Audit H1: canonicalize the match lookup (match keys are sorted tuples, but
    # image_pairs_rel preserves the JSON direction which is frequently reversed).
    # Audit M10: the minimum-match threshold is configurable and dropped pairs
    # are logged instead of being silently discarded.
    print("verify_matches")
    pairs_path = os.path.join(cache_dir, 'pairs.txt')
    dropped_pairs = 0
    with open(pairs_path, "w") as f:
        for rel_path1, rel_path2 in image_pairs_rel:
            abs_path1 = os.path.join(root_path, rel_path1)
            abs_path2 = os.path.join(root_path, rel_path2)
            key = tuple(sorted((abs_path1, abs_path2)))
            matches = indexed_matches.get(key)
            if matches is not None and len(matches) > min_matches:
                f.write(f"{rel_path1} {rel_path2}\n")
            else:
                dropped_pairs += 1
    if dropped_pairs:
        log.info("Dropped %d image pair(s) with <= %d matches during pairs.txt generation.",
                 dropped_pairs, min_matches)

    pycolmap.verify_matches(colmap_db_path, pairs_path)

    reconstruction_path = os.path.join(cache_dir, "reconstruction")
    if os.path.isdir(reconstruction_path):
        shutil.rmtree(reconstruction_path)
    os.makedirs(reconstruction_path, exist_ok=True)

    # COLMAP incremental mapper (multiple_models disabled inside the mapper).
    try:
        # Audit C4: returns (reconstruction, reconstruction_id); copy the dir
        # matching reconstruction_id instead of a hard-coded '0'.
        output_recon, reconstruction_id = colmap_run_incremental_mapper(
            database_path=colmap_db_path,
            image_path=root_path,
            output_path=reconstruction_path
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred during COLMAP incremental mapping: {e}") from e

    # Audit H5: .glb export is intentionally disabled, so no temp file is created
    # (previously tempfile.mktemp created an orphan file every run).
    outfile_name = None

    print("Reconstruction Summary:")
    print(output_recon.summary())

    if output_recon.num_points3D() == 0:
        raise RuntimeError("Reconstruction was created but contains 0 3D points.")

    colmap_image_id_to_name = {i: img.name for i, img in output_recon.images.items()}

    # The in-memory scene (poses, intrinsics, points, full-res images) is ONLY
    # needed for the (currently disabled) .glb export. Building it would load
    # every full-resolution image into RAM and run pycolmap-version-sensitive pose
    # extraction -- both wasteful when no export is requested. Skip entirely when
    # outfile_name is None. The caller only needs cache_dir + reconstruction_id.
    if outfile_name is not None:
        colmap_world_to_cam = {}
        colmap_intrinsics = {}
        images = {}
        for colmap_imgid, colmap_image in output_recon.images.items():
            # Audit H8 / pycolmap 4.x: cam_from_world may be a method (call it) or
            # a property (Rigid3d); .matrix may be a method (call) or attribute.
            cfw = colmap_image.cam_from_world
            if callable(cfw):
                cfw = cfw()
            if cfw is None:
                continue  # unregistered image
            mat = cfw.matrix() if callable(getattr(cfw, 'matrix', None)) else cfw.matrix
            colmap_world_to_cam[colmap_imgid] = mat

            camera = output_recon.cameras[colmap_image.camera_id]
            K = np.eye(3)
            K[0, 0] = camera.focal_length_x
            K[1, 1] = camera.focal_length_y
            K[0, 2] = camera.principal_point_x
            K[1, 2] = camera.principal_point_y
            colmap_intrinsics[colmap_imgid] = K

            with PIL.Image.open(os.path.join(root_path, colmap_image.name)) as im:
                images[colmap_imgid] = np.asarray(im)

        points3D = [(pts3d.xyz, pts3d.color) for pts3d in output_recon.points3D.values()]
        scene = GlomapRecon(colmap_world_to_cam, colmap_intrinsics, points3D, images)
        scene_state = GlomapReconState(scene, gradio_delete_cache, cache_dir, outfile_name,
                                       reconstruction_id=reconstruction_id)
        outfile = get_3D_model_from_scene(silent, scene_state, transparent_cams, cam_size)
    else:
        scene_state = GlomapReconState(None, gradio_delete_cache, cache_dir, outfile_name,
                                       reconstruction_id=reconstruction_id)
        outfile = None
    return scene_state, outfile


def get_3D_model_from_scene(silent, scene_state, transparent_cams=False, cam_size=0.05):
    """
    Extract a 3D model (glb file) from a reconstructed scene.

    Note: .glb export is currently disabled at the orchestrator level
    (``outfile_name`` is None), in which case this returns None without building
    the trimesh scene. Re-enable by providing a real outfile path.
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    recon = scene_state.glomap_recon

    scene = trimesh.Scene()
    pts = np.stack([p[0] for p in recon.points3d], axis=0)
    col = np.stack([p[1] for p in recon.points3d], axis=0)
    pct = trimesh.PointCloud(pts, colors=col)
    scene.add_geometry(pct)

    # add each camera
    cams2world = []
    for i, (id, pose_w2c_3x4) in enumerate(recon.world_to_cam.items()):
        intrinsics = recon.intrinsics[id]
        focal = (intrinsics[0, 0] + intrinsics[1, 1]) / 2.0
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        pose_w2c = np.eye(4)
        pose_w2c[:3, :] = pose_w2c_3x4
        pose_c2w = np.linalg.inv(pose_w2c)
        cams2world.append(pose_c2w)
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else recon.imgs[id], focal,
                      imsize=recon.imgs[id].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)

    return outfile
