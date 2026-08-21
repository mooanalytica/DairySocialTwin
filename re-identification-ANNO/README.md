# Dairy Cow Re-identification Occurrence Review

This repository contains the dataset-specific quality-control tools used to
review dairy-cow re-identification occurrences and publish corrected S6
detections and visualization videos. It covers occurrence segmentation,
30-second review-clip generation, a local review interface, review persistence,
corrected export, and same-frame global-ID collision audits.

The repository does not include farm video, annotations, processed detections,
model weights, generated review clips, or corrected outputs. See
[`DATA_ACCESS.md`](DATA_ACCESS.md) before restoring any private asset.

## Repository contents

- `count_bbox_occurrences.py` builds the fixed occurrence index from the S6
  detection export.
- `verify_bbox_occurrences.py` independently checks occurrence counts,
  intervals, ordering, timing, and hashes.
- `generate_clips.py` creates the 2,414 fixed-length review clips with one
  static red target box.
- `review_server.py`, `review_lock.py`, and `web/` provide the LAN review UI and
  atomic occurrence-level review storage.
- `export_corrected_results.py` applies the completed reviews and creates the
  corrected 33-column CSV plus two full visualization videos.
- `count_reviewed_frame_gid_duplicates.py` audits current review decisions for
  same-frame identity collisions before export.
- `count_corrected_frame_gid_duplicates.py` audits an exported corrected CSV.
- `tests/` contains synthetic unit fixtures and does not contain farm imagery.

## Supported environment

This project is supported only on the laboratory Linux workstation with two
NVIDIA RTX 5090 GPUs. GPU work must use physical GPU 1 through the process
environment. Clip generation and full-video export require `h264_nvenc`; there
is no CPU or OpenCV encoding fallback.

Use the existing DCSNA environment:

```bash
cd ~/DCSNA
source ~/.venvs/dcsna/bin/activate
cd /home/hyw/re-identification-ANNO
```

The direct PyPI imports are listed in `requirements.txt`. The supported setup
also requires:

- `ffmpeg` and `ffprobe` from the same installation, with working
  `h264_nvenc` support;
- the internal S6 source tree at `/home/hyw/re-identification-S6/src`, which
  supplies the `cowtrack` modules used by the exporter;
- the fixed S6 exports and restricted source videos described below.

### Fixed path layout

The project checkout must be located at:

```text
/home/hyw/re-identification-ANNO
```

The occurrence counting and verification utilities retain their original
fixed output path, `/home/hyw/re-identification-QC`. That path must resolve to
the project checkout. On a clean workstation, create the compatibility link
once:

```bash
ln -s /home/hyw/re-identification-ANNO /home/hyw/re-identification-QC
```

The fixed S6 inputs are expected under:

```text
/home/hyw/re-identification-S6/work/dairy_farm_1_gopro1_20250505/06_export/
```

This includes `detections_with_global_id.csv`,
`qa/global_track_summary.csv`, and the two tracked videos in `qa/videos/`.
The raw videos used by the final renderer are expected at:

```text
/mnt/dairycow_sna/FULLDATA/Dairy Farm Videos/May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO/GX040006.MP4
/mnt/dairycow_sna/FULLDATA/Dairy Farm Videos/May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO/GX050006.MP4
```

Missing files, changed hashes, ambiguous mappings, unsupported GPU selection,
or failed NVENC checks are fatal errors by design.

## Workflow

### 1. Rebuild the occurrence index

These commands create `occurrence_segments.csv` and
`bbox_occurrence_summary.json`. The counter refuses to overwrite existing
results.

```bash
cd /home/hyw/re-identification-ANNO
source ~/.venvs/dcsna/bin/activate
python3 -u count_bbox_occurrences.py
python3 -u verify_bbox_occurrences.py
```

Occurrences are formed independently inside each source clip. A global-ID
change always starts a new occurrence, and a same-ID stream is split when more
than 30 frames are missing.

### 2. Generate the review cache

The generator reads the existing S6 tracked videos, preserves their current
colored boxes and G-ID labels, and adds one fixed red box. The box is taken from
the occurrence anchor, scaled to 1920x1080, expanded to 1.25 times its original
width and height about its center, and clipped to the output frame.

```bash
cd /home/hyw/re-identification-ANNO
source ~/.venvs/dcsna/bin/activate
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python3 -u generate_clips.py --workers 2 \
  2>&1 | tee logs/generate_clips.log
```

The generated cache contains `cached_clips/O000001.mp4` through
`cached_clips/O002414.mp4` and `cached_clips/manifest.json`. Every clip is 900
frames at 30000/1001 fps. Completed outputs are validated with `ffprobe` and
SHA-256, and generation can resume safely. The entire cache is reproducible and
must not be committed.

Use `--start-index`, `--end-index`, or repeated `--occurrence-id` arguments for
a bounded selection. Use `--prepare-only` to build and validate the manifest
without encoding video.

### 3. Start the review interface

```bash
cd /home/hyw/re-identification-ANNO
source ~/.venvs/dcsna/bin/activate
python3 -u review_server.py --host 0.0.0.0 --port 9922
```

From another machine on the laboratory LAN, open:

```text
http://172.17.6.39:9922
```

The interface automatically selects the next unreviewed occurrence. Playback
starts three seconds before the occurrence anchor when that lead-in exists.
Reviews are written atomically by the server. Update inputs such as `1`, `2`,
and `10` are stored as `G0001`, `G0002`, and `G0010`.

### 4. Audit current decisions

Before export, check the complete review-to-detection mapping and report any
same-frame corrected G-ID collisions:

```bash
cd /home/hyw/re-identification-ANNO
source ~/.venvs/dcsna/bin/activate
python3 -u count_reviewed_frame_gid_duplicates.py
```

The final exporter performs the same collision protection and aborts before
video rendering if a frame would contain the same assigned G-ID more than once.

### 5. Export corrected results

The output directory must not already exist. The exporter has no overwrite,
resume, CPU-encoder, or partial-publication mode.

```bash
cd /home/hyw/re-identification-ANNO
source ~/.venvs/dcsna/bin/activate
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python3 -u export_corrected_results.py \
  --output-dir /home/hyw/re-identification-ANNO/corrected_export \
  2>&1 | tee logs/export_corrected_results.log
```

The exporter publishes exactly:

- `corrected_export/detections_with_global_id.csv`;
- `corrected_export/qa/videos/GX040006_tracked.mp4`;
- `corrected_export/qa/videos/GX050006_tracked.mp4`.

Accept decisions retain the original global ID. Update decisions change all
consistent identity fields and recompute global ordering fields. A reviewed
multiple-cow bbox remains `valid=true` but receives `global_track_id=-1`,
`display_global_id=-1`, blank UUID/status/basis/global-order fields, and
`invalid_reason=multiple_cows`. It is drawn as an unlabelled dashed gray box.
The original 364 `valid=false` rows remain unchanged and are not drawn.

To validate the complete mapping without writing an output CSV or opening
video, use:

```bash
python3 -u export_corrected_results.py --validate-only
```

After export, audit the corrected CSV with:

```bash
python3 -u count_corrected_frame_gid_duplicates.py \
  --input-csv corrected_export/detections_with_global_id.csv
```

## Generated and excluded artifacts

The following paths are intentionally excluded from GitHub:

| Path | Classification | Recovery |
| --- | --- | --- |
| `occurrence_segments.csv` | Processed index | Run `count_bbox_occurrences.py` |
| `bbox_occurrence_summary.json` | Processed summary | Run `count_bbox_occurrences.py` |
| `cached_clips/` | More than 2,400 generated review videos | Run `generate_clips.py` |
| `corrected_export/` | Corrected CSV and rendered videos | Run `export_corrected_results.py` |
| `logs/` | Runtime logs | Recreated by the documented commands |
| `__pycache__/`, lock files, temporary files | Runtime state | Recreated automatically |

## Tests

The tests use synthetic temporary fixtures. They do not generate media or
start the review server. Because `export_corrected_results.py` imports the
internal S6 `cowtrack` modules, run the suite only from the supported DCSNA
environment with the S6 source tree available.

```bash
cd /home/hyw/re-identification-ANNO
source ~/.venvs/dcsna/bin/activate
python3 -B -m unittest discover -s tests -v
```

No test or validation command is run automatically during repository setup.
