# Chunked MASt3R → COLMAP Pipeline — Detailed Review

This document traces the **complete data flow** of the high‑resolution (aerial) reconstruction pipeline, starting from the CLI entry point `colmap_from_mast3r_demo.py`, through the orchestrator `get_reconstructed_scene_J` in `mast3r/demo_glomap.py`, into the tiling/matching engine `run_chunked_mast3r_matching` in `chunked_utils.py`, and finally to the COLMAP incremental mapper.

Every step is illustrated with **real values from the `data/tobias/1/` dataset**:

| Property | Value |
|---|---|
| Images | 8 aerial ortho‑photos (TIFF, RGB) |
| Per‑image size | **4398 × 3321** px (W × H) |
| Names | `138001372_…_0031_P00_01.tif` … `138001379_…_0024_P00_01.tif` (a sequential flight strip: `0031 → 0024`) |
| Pre‑computed transforms | `data/tobias/1/fp/image_transforms.json` (44 entries = 22 image pairs × 2 directions) |
| Transform type | Near‑pure **translation**, identity rotation (e.g. `0031 → 0030` = translate `(-13, -662)` px) |
| Tile config (hardcoded) | 512 × 512, overlap 128 → stride 384 → **108 tiles per image** |
| Resulting COLMAP DB | `data/tobias/1/cache/cache/colmap.db` (8 images registered) |

---

## 0. High‑level architecture

```
colmap_from_mast3r_demo.py  (CLI entry point)
        │  parses args, loads MASt3R model, gathers image filelist
        ▼
get_reconstructed_scene_J()            [mast3r/demo_glomap.py:503]
        │
        ├── 1. Load pre-computed affine transforms (footprints JSON)
        ├── 2. Build absolute + relative image-pair lists from transforms
        ├── 3. kapture_import_image_folder_or_list()  → cameras+images into COLMAP DB
        │
        ├── 4. run_chunked_mast3r_matching()          [chunked_utils.py:567]  ← CORE
        │        ├── A. Tile every image (512×512, edge-aligned, 108 tiles/img)
        │        ├── B. For each image pair:
        │        │       - pick overlapping tile pairs via the affine transform
        │        │       - run MASt3R/DUSt3R inference on tile batches
        │        │       - extract 2D-2D matches (dense 3D-point reciprocal NN)
        │        │       - re-project tile-local matches → full-image coords
        │        ├── C. Build tracks (DisjointSet), filter len≥3, write
        │             keypoints + matches into the COLMAP DB
        │
        ├── 5. pycolmap.verify_matches()  (geometric verification via pairs.txt)
        ├── 6. colmap_run_incremental_mapper()  (pycolmap SfM)
        └── 7. Pack poses/intrinsics/points into GlomapReconState
        ▼
colmap_from_mast3r_demo.py copies reconstruction/0 → output/sparse/0
```

The key idea: these images are far too large (≈14.6 MP each) to feed to MASt3R at once, so the pipeline **tiles** them, **restricts matching to geometrically overlapping tile pairs** (using the footprint transforms), runs inference per tile, then **re‑projects** all matches back into full‑image pixel coordinates before building SfM tracks.

---

## 1. Entry point — `colmap_from_mast3r_demo.py`

### 1.1 Argument parsing (`get_cli_args_parser`, line 26)

It inherits the full DUSt3R/MASt3R argument parser (`get_demo_args_parser`) and adds four CLI‑only flags:

- `--input_dir`, `--output_dir` (required)
- `--scenegraph_type`, `--winsize`, `--win_cyclic`, `--refid`, `--shared_intrinsics`

> ⚠️ **Important caveat:** these scenegraph/window arguments are **not actually used** by `get_reconstructed_scene_J` in the current chunked implementation. The image pairs are derived **entirely from the transforms JSON**, not from a sliding window. The inherited `--image_size`, `--retrieval_model`, `--device`, `--weights`, `--model_name`, `--glomap_bin` *are* used. See the audit document.

### 1.2 `main()` (line 61) — step by step

1. **Resolve paths** (lines 68–77). `input_dir = data/tobias/1/images`, `output_dir = <user>`. It then makes `cache_dir = output_dir/"cache"`.
2. **Gather images** (lines 80–81): non‑recursive `iterdir()`, filtered by extension, sorted.
   For `tobias/1` this yields the 8 `.tif` files (the `_tiles_manifest.json` and `tiles_bounds.geojson` are skipped by the suffix filter).
3. **Load model** (lines 91–98): `AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")` on `args.device`.
4. **Call the orchestrator** (lines 104–124):
   ```python
   scene_state, _ = get_reconstructed_scene_J(
       glomap_bin=args.glomap_bin,
       outdir=str(cache_dir),          # = output_dir/cache
       gradio_delete_cache=False,
       model=model, retrieval_model=args.retrieval_model, device=args.device,
       silent=False, image_size=args.image_size,
       current_scene_state=None, filelist=filelist,
       scenegraph_type=..., winsize=..., win_cyclic=..., refid=...,
       shared_intrinsics=..., transparent_cams=False, cam_size=0.05)
   ```
   Note `outdir` is already `…/cache`. Combined with step 2 of the orchestrator this produces a **double‑nested** `…/cache/cache` (see §3.1 and the audit).
5. **Copy the sparse model** (lines 133–156): it expects the reconstruction at `Path(scene_state.cache_dir)/"reconstruction"/"0"` and copies it to `output_dir/sparse/0`.

---

## 2. The orchestrator — `get_reconstructed_scene_J` (`mast3r/demo_glomap.py:503`)

Signature:
```python
def get_reconstructed_scene_J(glomap_bin, outdir, gradio_delete_cache, model,
    retrieval_model, device, silent, image_size, current_scene_state, filelist,
    transparent_cams, cam_size, scenegraph_type, winsize, win_cyclic, refid,
    shared_intrinsics, **kw):
```

### Phase 1 — Load transforms (lines 516–522)
```python
root_path = os.path.commonpath(filelist)          # = .../data/tobias/1/images
TRANSFORMS_JSON_PATH = "/mnt/c/.../data/tobias/1/fp/image_transforms.json"   # HARDCODED
precomputed_transforms = {
    tuple(key.split('__')): np.array(value)
    for key, value in json.load(open(TRANSFORMS_JSON_PATH)).items()
}
```
- `root_path` = the common parent of the filelist = `data/tobias/1/images`.
- Each JSON key is `"<basenameA>__<basenameB>"` (extension‑less), value is a 2×3 affine matrix. Splitting on `'__'` gives `(nameA, nameB)` tuples.
- Example entry: `"138001372_…_0031_P00_01__138001373_…_0030_P00_01"` →
  `[[1.0, 0, -13], [0, 1.0, -662]]`  ⇒  pixel `(x,y)` in 0031 maps to `(x-13, y-662)` in 0030.

> ⚠️ The transforms path is **hardcoded to this machine/dataset**. This is the single biggest portability bug — see the audit.

### Phase 2 — Build pair lists (lines 524–531)
```python
filelist_relpath = [relpath(f, root_path) with '\\'->'/']      # e.g. "138001372_..._0031_P00_01.tif"
basename_to_relpath_map = {basename(path): path}
image_pairs_rel  = [(map[n1], map[n2]) for (n1,n2) in transforms if both present]
image_pairs_abs  = [(join(root,n1), join(root,n2)) ...]
```
For `tobias/1`: **22 unique image pairs** (each image matched to its next ~4 neighbors down the strip). Because the JSON stores **both directions**, `image_pairs_abs` actually contains 44 entries (22 pairs duplicated in opposite order). The downstream `tile_pair_cache` canonicalization handles this duplication.

The 22 pairs (by trailing frame number) are essentially:
```
0031↔{0030,0029,0028,0027}, 0030↔{0029,0028,0027,0026},
0029↔{0028,0027,0026,0025}, 0028↔{0027,0026,0025,0024},
0027↔{0026,0025,0024}, 0026↔{0025,0024}, 0025↔0024
```

### Phase 3 — Kapture → COLMAP DB (lines 533–541)
```python
kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)
colmap_db_path = cache_dir + '/colmap.db'          # .../cache/cache/colmap.db
colmap_db = COLMAPDatabase.connect(colmap_db_path)
kapture_to_colmap(kdata, ..., database=colmap_db, ...)
```
`kapture_import_image_folder_or_list` (`mast3r/colmap/mapping.py:293`) registers one `UNKNOWN_CAMERA` per image (or a single shared camera when `--shared_intrinsics`), reading `width,height` from each TIFF (4398×3321). After this step the DB has **8 images, 8 cameras, 0 keypoints, 0 matches** (verified: `images=8`).

### Phase 4 — Chunked matching (line 544) → see §3
```python
indexed_matches = run_chunked_mast3r_matching(
    model=model, device=device,
    image_pairs_to_match=image_pairs_abs,
    root_path=root_path, colmap_db=colmap_db,
    precomputed_transforms=precomputed_transforms,
    conf_thr=3.0, dedup_distance=2.0)
colmap_db.close()
```
On failure the code does **`exit(1)`** (line 558) — kills the whole Python process; flagged in the audit.

### Phase 5 — Geometric verification (lines 563–575)
```python
# write pairs.txt only for pairs with >15 matches
for rel1, rel2 in image_pairs_rel:
    if (abs(join(root,rel1)), abs(join(root,rel2))) in indexed_matches \
       and len(indexed_matches[...]) > 15:
        f.write(f"{rel1} {rel2}\n")
pycolmap.verify_matches(colmap_db_path, cache_dir + '/pairs.txt')
```
`verify_matches` runs two‑view geometry (essential‑matrix RANSAC) per pair and fills the `two_view_geometries` table.

> ⚠️ **Key‑ordering hazard:** `indexed_matches` keys are *canonical* (`(min_path, max_path)` lexicographically — see §3.4), but the lookup here uses the JSON's stored order `(abs1, abs2)`, which may be reversed. Reversed pairs are silently dropped from `pairs.txt`. See audit.

### Phase 6 — Incremental SfM (lines 623–633)
```python
output_recon = colmap_run_incremental_mapper(
    database_path=colmap_db_path, image_path=root_path,
    output_path=reconstruction_path)   # .../cache/cache/reconstruction
```
`colmap_run_incremental_mapper` (`mast3r/colmap/mapping.py:218`) builds `pycolmap.IncrementalPipelineOptions` tuned for low‑parallax aerial data:
- `triangulation.ignore_two_view_tracks = False`
- `triangulation.min_angle = 0.005`, `mapper.filter_min_tri_angle = 0.005`
- `mapper.init_max_error = 12.0`, `mapper.filter_max_reproj_error = 12.0`

and returns the **largest** reconstruction object.

> ⚠️ **Inconsistency:** the chunked matcher **discards all tracks shorter than 3 images** (`min_len_track=3`), so allowing two‑view tracks in the mapper is moot — they never reach the DB.

### Phase 7 — Pack results (lines 638–690)
Reads poses (`cam_from_world`), intrinsics (fx, fy, cx, cy → 3×3 K), and 3D points out of the `output_recon` object, loads each image pixel array, and wraps everything in `GlomapReconState`. `get_3D_model_from_scene` would normally export a `.glb`, but that export is **commented out** (lines 728–731), so only a temp file path is produced.

---

## 3. The core engine — `run_chunked_mast3r_matching` (`chunked_utils.py:567`)

```python
def run_chunked_mast3r_matching(model, device, image_pairs_to_match, root_path,
    colmap_db, precomputed_transforms,
    tile_size=(512,512), overlap_px=128, batch_size=4, conf_thr=3.0,
    dedup_distance=2.0):
```
Note the caller passes **only** `conf_thr` and `dedup_distance`; the tile geometry uses the defaults (512/128/batch 4). Tile size and overlap are **not exposed to the CLI**.

### Part A — Tiling (lines 577–582)
```python
tile_cache = {}
all_image_paths_abs = unique flatten of all pairs
for image_path in all_image_paths_abs:
    tile_cache[image_path] = generate_image_tiles(image_path, tile_size, overlap_px, save_tile_data=False)
```

#### `generate_image_tiles` (`chunked_utils.py:828`) — edge‑aligned tiling
For a `4398 × 3321` image with `tile=512, overlap=128 ⇒ stride=384`:
```python
y_starts = unique( range(0, 3321-512, 384) + [3321-512] )
        = unique([0,384,768,1152,1536,1920,2304,2688] + [2809])
        = [0,384,768,1152,1536,1920,2304,2688,2809]        # 9 rows
x_starts = unique([0,384,...,3840] + [3886])
        = [0,384,768,1152,1536,1920,2304,2688,3072,3456,3840,3886]   # 12 cols
```
⇒ **12 × 9 = 108 tiles per image**, each exactly 512×512. The last row/col tiles are shifted to align with the image edge (larger overlap there). Each tile dict stores:
```python
{
  "tile_id": "138001372_..._0031_P00_01.tif_tile_0042",
  "parent_image_path": ".../images/138001372_..._0031_P00_01.tif",
  "bounds_in_parent": (x0, y0, x0+512, y0+512),     # crop box
  "tile_data": np.array(512,512,3)
}
```
All 108 arrays are held **in RAM** in `tile_cache`. For 8 images that is `8 × 108 × 512 × 512 × 3 ≈ 678 MB`. (Memory note in audit.)

### Part B — Per‑pair tile matching (lines 584–749)

For each `(image_path1, image_path2)` in `image_pairs_to_match` (44 entries):

**B.1 Resolve the transform** (lines 602–632)
- Canonicalize the pair key to avoid recomputing the same pair twice.
- Look up `precomputed_transforms[(name1, name2)]`. If missing, try the reverse and invert with `cv2.invertAffineTransform`.
- For `0031 → 0030` the transform is `[[1,0,-13],[0,1,-662]]`.

**B.2 Determine overlapping tile pairs** — `determine_overlapping_tile_pairs` (`chunked_utils.py:955`), a **one‑to‑one** matcher:
1. For each tile in A, transform its 4 corners into B, compute the projected bbox, and find the **single** B tile with the largest intersection area.
2. **De‑conflict**: if several A tiles claim the same B tile, keep only the claim with the largest area.

#### Concrete example (`0031 → 0030`)
Take A tile at **col 0, row 2**, `bounds_in_parent = (0, 768, 512, 1280)`.
Applying `out = (x − 13, y − 662)`:
```
projected bbox in B = (-13, 106, 499, 618)
```
Compare against B tile **col 0, row 0**, `bounds = (0, 0, 512, 512)`:
```
intersection = ( max(-13,0), max(106,0), min(499,512), min(618,512) )
             = (0, 106, 499, 512)
area         = (499-0) × (512-106) = 499 × 406 = 202,594 px²
```
This is the **best** match for B tile (col 0, row 0). In total **6** A tiles overlap that one B tile, so the de‑conflict pass keeps only this winner `(A col0,row2) ↔ (B col0,row0)` and discards the other 5 claims. (This aggressive one‑to‑one rule can drop genuinely useful matches — see audit.)

**B.3 Batched inference** (lines 652–700)
Tile pairs are batched (default 4). Each tile is normalized with `ImgNorm` (`ToTensor` + normalize mean/std 0.5), wrapped with `true_shape` and a unique `instance` id, and fed to `dust3r.inference.inference(...)`.
```python
output = inference(inference_input_batch, model, device,
                   batch_size=len(inference_input_batch), verbose=False)
```
`output['pred1']`/`output['pred2']` hold batched `pts3d`, `pts3d_in_other_view`, `conf`.

**B.4 Extract matches per tile** — `extract_matches_from_tile_prediction` (`chunked_utils.py:467`):
- Confidence mask: keep pixels with `conf ≥ conf_thr` (3.0) in **both** views.
- Build dense 2D grids + per‑pixel 3D points for each view.
- Match via `find_reciprocal_matches` (reciprocal nearest neighbors in 3D).
- Returns `matches_im0, matches_im1` in **tile‑local** coordinates (pixel `(x,y)` of the 512×512 tile).

> Design note: this uses the **DUSt3R 3D‑point dense‑matching** path, **not** MASt3R descriptors (`desc`/`desc_conf`). MASt3R's descriptor head is effectively unused here.

**B.5 Re‑project to full image & aggregate** (lines 736–749)
```python
offset_a = tile_a["bounds_in_parent"][:2]      # (x0, y0)
kpts1_global = kpts1_local + offset_a          # tile-local -> full-image pixel
kpts2_global = kpts2_local + offset_b
aggregated_matches_temp[(img1, img2)]["kpts0"].append(kpts1_global)  # .kpts1 analogous
```
Continuing the example: a match found at tile‑local `(120, 80)` in A tile (col0,row2) and `(133, 742)` in B tile (col0,row0) becomes:
```
A global = (120+0,   80+768) = (120, 848)   in image 0031
B global = (133+0,  742+0)   = (133, 742)   in image 0030
```
After all tile pairs for an image pair are done, the per‑pair lists are concatenated (`aggregated_matches_final`, lines 751–770) into single `(N,2)` arrays.

### Part C — Track building & COLMAP export (lines 772–822)

**C.1 `deduplicate_and_format_for_colmap`** (`chunked_utils.py:531`) runs three sub‑steps:

1. `_cluster_keypoints` (line 104): gather every raw keypoint per image (across all pairs it participates in), then `np.unique` to drop *exact* duplicates → `unique_kpts_per_image[path]`.

2. `_build_and_merge_tracks` (line 156): build a `cKDTree` per image; for each raw match, query the nearest unique keypoint within `distance_threshold` (2.0 px) in each image, and **union** the two `(path, idx)` nodes in a custom `DisjointSet`. The DSU enforces the invariant *"a track never contains two keypoints from the same image"* (`DisjointSet.union`, line 66, checks `image_sets` disjointness before merging). Tracks are collected as lists of `(image_path, kpt_index)`.

3. `_filter_and_finalize` (line 362): drop tracks with `< min_len_track` (3) images; assign each surviving keypoint a **new sequential index** per image; emit all pairwise `combinations(track, 2)` as indexed matches, with the pair key forced to canonical order `(path1 < path2)`.

   Example: a 4‑view track `{(0031,i₁),(0030,i₂),(0029,i₃),(0028,i₄)}` (length 4 ≥ 3) produces **6** match rows (`C(4,2)=6`) and registers 4 keypoints (one per image, if first time seen).

Output: `unique_kpts` `{path: (M,2) float}` and `indexed_matches` `{(path1,path2): (K,2) int}`.

**C.2 Write to COLMAP** (lines 776–821)
```python
colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)   # name -> id
# keypoints: +0.5 to center in pixel, written with add_keypoints
colmap_db.add_keypoints(colmap_id, keypoints + 0.5)
# matches: canonical (id1,id2) dedup, swap columns if id1>id2
colmap_db.add_matches(id1, id2, matches)
colmap_db.commit()
```
The `pairs_added_to_db` set guarantees each image‑pair row is written **once** (COLMAP's `matches` table has a unique constraint on the pair).

After Part C, `indexed_matches` is returned to the orchestrator for the `pairs.txt` / verify step.

---

## 4. Post‑matching — verification & SfM (back in `get_reconstructed_scene_J`)

1. **`pairs.txt`** is written from `image_pairs_rel`, keeping only pairs with `>15` matches (line 571).
2. **`pycolmap.verify_matches`** runs essential‑matrix RANSAC per pair → populates `two_view_geometries`.
3. **`colmap_run_incremental_mapper`** runs pycolmap SfM on the DB + images, writing `reconstruction/0` (and possibly `1`, …).
4. The largest reconstruction is read back into `GlomapReconState`.

Finally `colmap_from_mast3r_demo.py` copies `…/cache/cache/reconstruction/0` → `<output_dir>/sparse/0` (containing `cameras.bin`, `images.bin`, `points3D.bin`).

---

## 5. Worked end‑to‑end example (`tobias/1`)

| Stage | Concrete value |
|---|---|
| `root_path` | `…/data/tobias/1/images` |
| `filelist` length | 8 `.tif` files |
| Transforms loaded | 44 entries (22 unordered pairs) |
| `cache_dir` (inside orchestrator) | `…/data/tobias/1/cache/cache` |
| `colmap.db` after kapture step | cameras=8, images=8, kpts=0, matches=0 |
| Tiles per image | 108 (12 cols × 9 rows), 512×512, stride 384 |
| Total tiles in `tile_cache` | 8 × 108 = **864 tiles ≈ 678 MB RAM** |
| Example transform `0031→0030` | `[[1,0,-13],[0,1,-662]]` (≈662 px forward overlap) |
| Example tile pair | A(col0,row2)`[0,768,512,1280]` ↔ B(col0,row0)`[0,0,512,512]`, overlap **202,594 px²** |
| A tiles overlapping that B tile | 6 → 5 dropped by one‑to‑one de‑conflict |
| Example match reprojection | A tile‑local `(120,80)` → A global `(120,848)`; B tile‑local `(133,742)` → B global `(133,742)` |
| Track filter | `min_len_track = 3` (two‑view tracks discarded) |
| `conf_thr` (on `conf`) | 3.0 |
| `dedup_distance` | 2.0 px |
| Final DB tables written | `keypoints`, `matches` (per‑pair unique) |
| Mapper options | `min_angle=0.005`, `init_max_error=12`, `filter_max_reproj_error=12` |
| Output copied to | `<output_dir>/sparse/0` |

### Data‑structure evolution for one match
```
inference output (batched tensors, tile-local 3D pts + conf)
   └─ extract_matches_from_tile_prediction  →  matches_im0, matches_im1 (tile-local 2D)
        └─ + bounds_in_parent offset        →  kpts1_global, kpts2_global (full-image px)
             └─ aggregated_matches_final[pair]["kpts0"/"kpts1"]  (concatenated per pair)
                  └─ _cluster_keypoints     →  unique_kpts_per_image[path]   (exact dedup)
                       └─ _build_and_merge_tracks (cKDTree + DisjointSet) → tracks
                            └─ _filter_and_finalize (len≥3) → final_keypoints, final_matches_indexed
                                 └─ colmap_db.add_keypoints / add_matches → COLMAP DB
```

---

## 6. Summary of the control flow (who calls what)

| Caller | Function | File:line | Purpose |
|---|---|---|---|
| CLI `main` | `AsymmetricMASt3R.from_pretrained` | colmap_from_mast3r_demo.py:97 | load model |
| CLI `main` | `get_reconstructed_scene_J` | demo_glomap.py:503 | orchestrate |
| orchestrator | `kapture_import_image_folder_or_list` | mapping.py:293 | register images/cameras |
| orchestrator | `kapture_to_colmap` | kapture lib | seed COLMAP DB |
| orchestrator | `run_chunked_mast3r_matching` | chunked_utils.py:567 | **tiling + matching** |
| chunked matcher | `generate_image_tiles` | chunked_utils.py:828 | tile one image |
| chunked matcher | `determine_overlapping_tile_pairs` | chunked_utils.py:955 | one‑to‑one tile pairing |
| chunked matcher | `inference` | dust3r lib | MASt3R forward pass |
| chunked matcher | `extract_matches_from_tile_prediction` | chunked_utils.py:467 | 2D‑2D matches per tile |
| chunked matcher | `deduplicate_and_format_for_colmap` | chunked_utils.py:531 | build tracks |
| chunked matcher | `_cluster_keypoints` / `_build_and_merge_tracks` / `_filter_and_finalize` | chunked_utils.py:104/156/362 | track pipeline |
| orchestrator | `pycolmap.verify_matches` | pycolmap | two‑view geometry |
| orchestrator | `colmap_run_incremental_mapper` | mapping.py:218 | pycolmap SfM |

> **What is intentionally *not* used:** the MASt3R descriptor head, the retrieval model, the scenegraph/window CLI flags, and `image_size`. Matching is driven solely by the footprint transforms JSON and the DUSt3R 3D‑point dense path.
