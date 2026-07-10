# Chunked MASt3R → COLMAP Pipeline — Code Audit

A security/correctness/maintainability audit of the high‑resolution (aerial) reconstruction pipeline (`colmap_from_mast3r_demo.py`, `mast3r/demo_glomap.py`, `chunked_utils.py`, `mast3r/colmap/mapping.py`). Findings are grouped by severity. Each item lists the **location**, the **problem**, the **impact**, and a **fix**.

Severity legend: **Critical** (wrong results / crashes / unrecoverable), **High** (correctness or robustness bugs), **Medium** (design/maintenance risk), **Low** (cosmetic / dead code).

---

## CRITICAL

### C1 — Hardcoded transforms path (makes the script non‑portable)
- **Where:** `mast3r/demo_glomap.py:517`
  ```python
  TRANSFORMS_JSON_PATH = "/mnt/c/.../data/tobias/1/fp/image_transforms.json"
  ```
- **Problem:** The entire pair‑selection logic depends on a JSON file whose location is burned into source. It only works on this one machine for this one dataset.
- **Impact:** Running on any other dataset (or another machine) silently produces wrong/empty pair lists.
- **Fix:** Add a CLI argument (`--transforms_json`) threaded through `get_reconstructed_scene_J`, or auto‑derive from `root_path` (e.g. `<root>/../fp/image_transforms.json`). Fail loudly if the file is missing.

### C2 — `exit(1)` inside library code
- **Where:** `mast3r/demo_glomap.py:558`
  ```python
  except Exception as e:
      print(f'Error during chunked matching: {e}')
      colmap_db.close()
      exit(1)
  ```
- **Problem:** A library function aborts the whole interpreter instead of raising. This bypasses caller error handling, prevents cleanup (temp files, the `.glb`), and makes it unusable as an importable API.
- **Impact:** Unrecoverable crash; untestable; resource leaks.
- **Fix:** `raise` (or `raise RuntimeError(...) from e`) after closing the DB.

### C3 — Caller dereferences `scene_state` even on failure
- **Where:** `colmap_from_mast3r_demo.py:104` vs `mast3r/demo_glomap.py:633/654`
- **Problem:** `get_reconstructed_scene_J` returns `None, None` on mapper failure or 0‑point reconstruction. The caller then does `Path(scene_state.cache_dir)` → `AttributeError: 'NoneType'`.
- **Impact:** Masked error: the real cause (failed SfM) is hidden behind an unrelated `NoneType` traceback.
- **Fix:** In `main()`, check `if scene_state is None: print(...); return`. Have the orchestrator raise instead of returning `None` on hard failures.

### C4 — `reconstruction/0` may not be the reconstruction actually used
- **Where:** `colmap_from_mast3r_demo.py:135` vs `mast3r/colmap/mapping.py:288`
- **Problem:** `colmap_run_incremental_mapper` returns the **largest** reconstruction object, but `colmap_from_mast3r_demo.py` copies the on‑disk dir `reconstruction/0` unconditionally. If pycolmap writes multiple models and the largest is `1` (or `0` is a degenerate/duplicate model), the wrong (or empty) model is copied.
- **Impact:** Silently shipping the wrong sparse model.
- **Fix:** Either disable multiple models (`multiple_models=False`) and re‑run if `0` is empty, or copy the dir corresponding to the returned `reconstruction_id` rather than hardcoding `0`.

---

## HIGH

### H1 — `pairs.txt` / match lookup uses non‑canonical pair ordering
- **Where:** `mast3r/demo_glomap.py:571`
  ```python
  if (abs_path1, abs_path2) in indexed_matches and len(...) > 15:
  ```
- **Problem:** `indexed_matches` keys are canonicalized to `(min_path, max_path)` inside `_filter_and_finalize` (`chunked_utils.py:395‑397`). But `image_pairs_rel` preserves the JSON's stored direction, which is frequently reversed. Reversed pairs fail the `in` check.
- **Impact:** Pairs silently dropped from geometric verification → weaker/incomplete SfM graph. With the JSON storing both directions this is partially masked, but it is a latent correctness bug.
- **Fix:** Canonicalize before lookup:
  ```python
  key = tuple(sorted((abs_path1, abs_path2)))
  if key in indexed_matches and len(indexed_matches[key]) > 15:
  ```

### H2 — Track‑length filter vs. mapper config contradict each other
- **Where:** `chunked_utils.py:773` (`min_len_track=3`) vs `mast3r/colmap/mapping.py:246` (`ignore_two_view_tracks=False`).
- **Problem:** The matcher discards every track seen in fewer than 3 images. The mapper is then configured to *allow* two‑view tracks — which can never exist in the DB.
- **Impact:** For a narrow strip (`tobias/1`), many genuine 2‑view correspondences are thrown away, thinning the SfM graph; the lenient mapper setting gives nothing in return. Likely the root cause of sparse reconstructions on this dataset.
- **Fix:** Pick one strategy. Either set `min_len_track=2` (and keep the mapper setting) for strip/aerial data, or raise it deliberately and remove the contradictory mapper flag. Make `min_len_track` a CLI parameter.

### H3 — MASt3R descriptors are never used (quality regression)
- **Where:** `chunked_utils.py:467` (`extract_matches_from_tile_prediction` uses `pred['pts3d']`/`pred['conf']`).
- **Problem:** The chunked matcher uses the **DUSt3R 3D‑point dense‑matching** path, ignoring MASt3R's learned descriptor head (`desc`/`desc_conf`) which is the model's key advantage for matching.
- **Impact:** Lower match precision/coverage than the stock MASt3R pipeline, especially across viewpoint changes.
- **Fix:** If descriptors are desired, port the descriptor path from `get_im_matches` (`mast3r/colmap/database.py:99`). If the 3D‑point path is intentional for tiles, document why, and reconsider `conf_thr=3.0` (that threshold was tuned for `desc_conf`, not raw pointmap `conf`).

### H4 — Threshold semantics mismatch (`conf_thr`)
- **Where:** `mast3r/demo_glomap.py:551` (`conf_thr=3.0`) → `chunked_utils.py:492` (`conf >= conf_thr`).
- **Problem:** `3.0` is a reasonable threshold for MASt3R **descriptor confidence** (`desc_conf`, which lives in a different range). Here it is applied to raw pointmap `conf`, whose scale differs.
- **Impact:** Filtering may be far too strict (or too loose) than intended → few/no matches on some tiles.
- **Fix:** Validate the `conf` distribution on sample tiles and pick a dataset‑appropriate threshold; expose it as a CLI flag.

### H5 — `tempfile.mktemp` + orphan temp file
- **Where:** `mast3r/demo_glomap.py:643` and the commented‑out export at `728‑731`.
- **Problem:** `tempfile.mktemp` is deprecated and has a race condition (TOCTOU). Worse, the `.glb` export is commented out, so the temp path is created but **never written** → orphan file each run.
- **Impact:** Minor security smell; disk litter.
- **Fix:** Use `tempfile.mkstemp`/`NamedTemporaryFile` only if/when you actually export, or remove the dead temp creation entirely.

### H6 — Aggressive one‑to‑one tile pairing drops valid matches
- **Where:** `chunked_utils.py:955` (`determine_overlapping_tile_pairs`).
- **Problem:** Each B tile is claimed by exactly one A tile (best area); all other overlapping A tiles are discarded. In the worked example, **6** A tiles overlap one B tile and 5 are dropped.
- **Impact:** Lost correspondences at tile seams and in high‑overlap regions; can create coverage gaps that hurt track continuity.
- **Fix:** Allow multiple A→B claims above a minimum overlap area, or keep the one‑to‑one rule only for the *de‑conflict among A tiles* but allow an A tile to match several B tiles (and vice‑versa) when justified by area. Make the min‑area threshold explicit.

### H7 — No atomicity / cleanup around the COLMAP DB
- **Where:** `mast3r/demo_glomap.py:535‑536` (`os.remove(colmap_db_path)` then rebuild).
- **Problem:** If the process is killed mid‑run, a half‑written DB remains. There is no `try/finally` closing the connection across the matching block.
- **Impact:** Stale/corrupt DB on the next run; the `os.remove` can also race if two runs target the same dir.
- **Fix:** Use a context manager / `try/finally` for `colmap_db`, and consider writing to a temp DB then renaming atomically.

### H8 — `cam_from_world` API fragility + brittle iteration break
- **Where:** `mast3r/demo_glomap.py:663‑680`.
- **Problem:** Code branches on `callable(colmap_image.cam_from_world)` to support different pycolmap versions, and breaks the loop at `idx+1 == num_reg_images` ("bug with the iterable"). Both signal reliance on undocumented behavior.
- **Impact:** Breaks silently across pycolmap versions; the magic `break` can skip/duplicate the last image.
- **Fix:** Pin/verify the pycolmap API (modern versions expose `cam_from_world` as a `Rigid3d` with `.matrix()`). Iterate with a plain guard and drop the magic `break`.

---

## MEDIUM

### M1 — Double‑nested cache directory
- **Where:** `colmap_from_mast3r_demo.py:76` sets `cache_dir = output_dir/"cache"`, then passes it as `outdir`; `get_reconstructed_scene_J:510` does `cache_dir = os.path.join(outdir, 'cache')` again → `<output>/cache/cache`.
- **Impact:** Confusing layout; the caller's copy logic (`Path(scene_state.cache_dir)/"reconstruction"/"0"`) only works because it reads back the same doubled path.
- **Fix:** Pass `output_dir` directly and let the orchestrator add one `cache` level; or remove the orchestrator's extra join.

### M2 — CLI flags that do nothing
- **Where:** `colmap_from_mast3r_demo.py:38‑48` (`--scenegraph_type`, `--winsize`, `--win_cyclic`, `--refid`) and `--image_size`.
- **Problem:** These are threaded into `get_reconstructed_scene_J` but never read (pairs come from the transforms JSON; tiles are fixed 512×512).
- **Impact:** Misleading UX; users think they are tuning the pipeline.
- **Fix:** Remove unused flags, or actually wire them (e.g., expose `tile_size`, `overlap_px`, `min_len_track`, `conf_thr` instead).

### M3 — Tile geometry / thresholds not configurable
- **Where:** `run_chunked_mast3r_matching` defaults `tile_size=(512,512)`, `overlap_px=128`, `batch_size=4`, `min_len_track=3` (`chunked_utils.py:570/773`).
- **Impact:** Cannot adapt to GPU memory or dataset resolution without editing code.
- **Fix:** Expose as CLI args.

### M4 — ~678 MB resident tile cache (RAM)
- **Where:** `chunked_utils.py:577‑582` keeps every tile's `tile_data` numpy array for all images simultaneously.
- **Impact:** 8 images × 108 tiles × 512²×3 ≈ **678 MB**; scales linearly and will OOM on larger sets.
- **Fix:** Lazy‑load tiles from a per‑image cache on disk (PIL crop), or evict `tile_data` once a tile has been matched, or memory‑map the source images.

### M5 — Delimiter fragility for transform keys
- **Where:** `mast3r/demo_glomap.py:521` `tuple(key.split('__'))`.
- **Problem:** If an image basename ever contains `__`, the split yields >2 parts and the tuple lookup fails.
- **Fix:** Use a more robust separator or store pairs as a JSON object `{ "pair": ["a","b"], "M": [[...]] }`.

### M6 — `DisjointSet` invariant can reject legitimate merges
- **Where:** `chunked_utils.py:66` (`union` refuses if image sets intersect).
- **Problem:** The "one keypoint per image per track" invariant is correct in principle, but the *transitive* enforcement can block a merge that would have been fine, fragmenting tracks.
- **Impact:** Shorter tracks → more dropped (len<3) → sparser SfM.
- **Fix:** Consider nearest‑representative selection instead of full rejection, and benchmark track lengths before/after.

### M7 — Name collision: two `deduplicate_and_format_for_colmap`
- **Where:** `chunked_utils.py:531` (used) vs `mast3r/demo_glomap.py:394` (different implementation, unused).
- **Impact:** Confusion about which runs; risk someone calls the wrong one.
- **Fix:** Delete the dead `demo_glomap.py` copy (and the other duplicate helpers there — see L1).

### M8 — Large‑image DoS protection disabled globally
- **Where:** `chunked_utils.py:826` `Image.MAX_IMAGE_PIXELS = None`.
- **Problem:** Disables PIL's decompression‑bomb guard for the whole process.
- **Impact:** A malformed/huge image can exhaust memory.
- **Fix:** Set it scoped, or validate image dimensions explicitly and reject absurd sizes.

### M9 — Unbounded subprocess output loop
- **Where:** `mast3r/colmap/mapping.py:203‑208` (the `glomap_run_mapper` stdout loop).
- **Problem:** Reads stdout line‑by‑line forever; no timeout; currently unused (mapper is pycolmap now) but still present.
- **Fix:** Use `subprocess.run(..., check=True)` with a timeout, or delete if GLOMAP is fully retired.

### M10 — `> 15` magic threshold silently drops pairs
- **Where:** `mast3r/demo_glomap.py:571`.
- **Problem:** Hardcoded `len(...) > 15`; pairs with fewer matches never get verified.
- **Fix:** Expose as a CLI flag and log how many pairs are dropped.

---

## LOW / DEAD CODE / STYLE

### L1 — Substantial dead/duplicate code in `demo_glomap.py`
Unused, broken, or superseded functions that should be removed to reduce confusion and maintenance load:
- `generate_image_tiles` (demo_glomap.py:94) — duplicate of `chunked_utils.generate_image_tiles`.
- `_check_bbox_intersection` / `determine_overlapping_tile_pairs` (demo_glomap.py:161/171) — superseded by the `chunked_utils` versions.
- `run_chunked_pipeline` (demo_glomap.py:220) — references undefined `run_inference_on_tiles`/`deduplicate_and_format_for_colmap`.
- `run_inference_on_tiles` (demo_glomap.py:302) — uses nonexistent keys `output['pred_pts1']`, `output['confidence']`; would crash if called.
- `export_chunked_matches_to_colmap` (demo_glomap.py:464) — duplicate logic, unused.
- `deduplicate_and_format_for_colmap` (demo_glomap.py:394) — duplicate of `chunked_utils.py:531`, different behavior.
- `main_demo` (demo_glomap.py:736) — the old Gradio entry, references the old `get_reconstructed_scene` (no `_J`).

### L2 — Duplicate method in `DisjointSet`
- **Where:** `chunked_utils.py:90‑102` defines `get_all_subsets` **twice** (second overrides the first).
- **Fix:** Delete the duplicate.

### L3 — GPU variants reference unavailable modules
- **Where:** `chunked_utils.py:14‑16` comment out `cupy`/`cupyx`, yet `__cluster_keypoints` (129), `__build_and_merge_tracks` (221), `__filter_and_finalize` (407) use them. They are name‑mangled (private) and never called, but would `NameError` if ever invoked.
- **Fix:** Remove the GPU stubs or guard them behind a real import check.

### L4 — Unused imports
- **Where:** `chunked_utils.py:13` `from multiprocessing import Pool` (unused), `import time` (only in comments); `demo_glomap.py:37‑38` duplicate `import os`/`import numpy as np`.

### L5 — Wrong `tqdm` `total` for tail batches
- **Where:** `chunked_utils.py:652` `total=len(overlapping_tile_pairs)//batch_size` under‑counts the final partial batch.
- **Fix:** Drop the explicit `total` (tqdm infers it from the range).

### L6 — Cosmetic double prints
- **Where:** `mast3r/demo_glomap.py:646‑648` prints `summary()` twice (one commented). Minor.

### L7 — Non‑recursive image discovery
- **Where:** `colmap_from_mast3r_demo.py:81` uses `iterdir()` (not recursive).
- **Impact:** Sub‑foldered image sets are silently ignored.
- **Fix:** Use `rglob` if recursive collection is intended, or document the single‑folder constraint.

### L8 — No structured logging / no exception context
- Heavy use of `print`; the top‑level `except Exception` (colmap_from_mast3r_demo.py:127) prints the message + traceback but loses the failure type for programmatic handling.
- **Fix:** Adopt `logging` and propagate typed exceptions.

### L9 — `kapture_import_image_folder_or_list` creates one camera per image by default
- **Where:** `mast3r/colmap/mapping.py:318‑325`.
- For a single‑camera aerial strip, this over‑parameterizes intrinsics unless `--shared_intrinsics` is passed. Document the implication.

---

## Security summary

No classic injection vectors were found (SQL is via the kapture/pycolmap DB helpers; subprocesses use argument lists, not shells). The main security‑adjacent concerns are:

| # | Concern | Severity |
|---|---|---|
| C2 | `exit(1)` in library code | Critical (availability/cleanup) |
| H5 | `tempfile.mktemp` race + orphan file | High |
| M8 | `MAX_IMAGE_PIXELS = None` (decompression‑bomb guard off) | Medium |
| H7 | Non‑atomic DB writes, no `try/finally` on the connection | High (integrity) |
| C1 | Hardcoded absolute path | Critical (operability, not exploitable) |

---

## Recommended priority order

1. **C1** make the transforms path configurable (unblocks any new dataset).
2. **C2 + C3** stop using `exit(1)`/`None` returns; raise and handle in the CLI.
3. **H1** canonicalize the `pairs.txt` lookup (correctness of the SfM graph).
4. **H2 + H3 + H4** revisit track‑length, descriptor usage, and `conf_thr` (biggest quality levers).
5. **H6 / M4** tune tile pairing + RAM usage for scale.
6. **M1/M2/M3** clean up the cache nesting and expose real CLI knobs.
7. **L1** delete the large block of dead/duplicate code in `demo_glomap.py`.
