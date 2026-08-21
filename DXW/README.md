# Dairy Cattle Social Network Analysis Web Tools

This repository contains the source code for two internal research interfaces
used to inspect dairy-cattle trajectories and social-network analysis results.
It is maintained for [Moo Analytica](https://mooanalytica.com/).

The repository is intentionally source-only. Commercial-farm video, manual
annotations, re-identification records, trained models, full configurations,
processed results, and runtime-ready caches are restricted and are not part of
the GitHub repository. See [DATA_ACCESS.md](DATA_ACCESS.md).

## Components

- `WebUIL/` serves the floor-plan and track-ID video interface. It reads Stage1
  tracking/keypoint CSVs, joins the strict re-identification index, displays
  precomputed Stage2 interaction results, and generates the durable SNA input
  checkpoint and downstream analysis outputs.
- `Dashboard/` is a read-only dashboard for the current Farm 1 / Camera 1 SNA
  result. It renders Figures 1-10, cattle information cards, fixed cattle
  photos, appearance intervals, and deep links back to WebUIL.

Stage1 is visual perception and CSV generation: detection, ByteTrack tracking,
keypoint pose estimation, and optional identity assignment. Stage2 begins after
those CSVs and covers candidate-pair selection, temporal features, interaction
classification, valence classification, and social-network analysis.

## Supported system

Use the laboratory NVIDIA workstation only. The supported runtime is the
existing laboratory virtual environment:

```bash
cd /home/hyw/DCSNA
source /home/hyw/.venvs/dcsna/bin/activate
```

The exact Python, PyTorch, and CUDA versions are managed by that workstation
environment and are not independently claimed by this repository. Do not
replace or reinstall PyTorch from these requirements files. Local Stage2 code
is loaded from `/home/hyw/DCSNA-ALL`.

GPU work must use the second GPU through `CUDA_VISIBLE_DEVICES=1`. Video-cache
generation requires an FFmpeg build with working `h264_nvenc`; an NVENC failure
is an error and must not silently fall back to CPU encoding. `ffmpeg` and
`ffprobe` must be available on the system path.

`Dashboard/requirements.txt` pins the verified standalone Dashboard scientific
stack. `WebUIL/requirements.txt` records the directly imported non-PyTorch
packages that are known from this checkout; the laboratory environment remains
authoritative for external Stage2 dependencies.

## Restore restricted assets

Obtain private assets from the internal Dropbox folder:

```text
Yiwen Huang SNA Project Summer 2026/private_assets/
```

Restore every item to its original path. The Dropbox staging tree uses
`DXW/<original repository-relative path>`; remove the staging prefix when
restoring. For example:

```text
private_assets/DXW/WebUIL/sna_inputs
    -> /home/hyw/DXW/WebUIL/sna_inputs
private_assets/DXW/WebUIL/.cache/stage2_precomputed
    -> /home/hyw/DXW/WebUIL/.cache/stage2_precomputed
private_assets/DXW/Dashboard/.cache/cow_photos
    -> /home/hyw/DXW/Dashboard/.cache/cow_photos
```

During repository cleanup, assets awaiting manual Dropbox transfer are staged
under `MOVE_IT/DXW/<original repository-relative path>`. `MOVE_IT/` is ignored
by Git and must never be committed.

The current repository-local private asset set comprises:

- `WebUIL/sna_inputs/F1_Gopro1_20250505/`
- `WebUIL/sna_outputs/F1_Gopro1_20250505/`
- `WebUIL/.cache/reidentification_1-1.sqlite3`
- `WebUIL/.cache/stage2_precomputed/`
- `WebUIL/.cache/track_id_videos/`
- `WebUIL/stage2_clip_val_all_bbox_paths_by_category.csv`
- Dashboard trajectory, appearance, and fixed-photo caches under
  `Dashboard/.cache/`

The applications also require these external private paths:

- `/home/hyw/DCSNA-ALL`
- `/home/hyw/FloorPlanAnnoEN/output15`
- `/home/hyw/FloorPlanAnnoSR/output15`
- `/mnt/data4t/hyw/Stage1_segmented`
- `/mnt/data4t/hyw/re-identification-results/1-1.csv`
- `/mnt/drive_bf/BF/re-identification-11B-GOOD/work/dairy_farm_1_gopro1_20250505_all11/06_export_complete_sequence`
- `/home/hyw/time_sequence_detector_v2/output/1_1.csv`
- `/mnt/dairycow_sna/FULLDATA/Dairy Farm Videos/May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO`

Preserve authoritative full video paths recorded in manifests and indices.
GX-style basenames are not unique and must not be used as a fallback identity.

## Start WebUIL

After restoring the private assets:

```bash
cd /home/hyw/DXW/WebUIL
source /home/hyw/.venvs/dcsna/bin/activate
CUDA_VISIBLE_DEVICES=1 python3 -u app.py --host 0.0.0.0 --port 9922
```

Open `http://172.17.6.39:9922/` from another machine on the laboratory LAN.

## Start Dashboard

```bash
cd /home/hyw/DXW/Dashboard
./run_dashboard.sh
```

Open `http://172.17.6.39:2299/` from another machine on the laboratory LAN.

## Regenerate excluded outputs

Generated files are intentionally excluded from Git. Run these commands only
after all private inputs have been restored.

```bash
cd /home/hyw/DXW/WebUIL
source /home/hyw/.venvs/dcsna/bin/activate

python3 -u reid_index.py
CUDA_VISIBLE_DEVICES=1 python3 -u generate_stage2_precomputed.py \
  --all --overwrite --cuda-visible-devices 1
CUDA_VISIBLE_DEVICES=1 python3 -u generate_video_cache.py \
  --all --overwrite --encoder h264_nvenc --workers 2
CUDA_VISIBLE_DEVICES=1 python3 -u generate_sna_precomputed.py --overwrite
python3 -u generate_sna_tf_outputs.py \
  --sample F1_Gopro1_20250505 --overwrite
```

Dashboard trajectory, appearance, and cattle-photo caches are validated and
rebuilt at Dashboard startup when absent. Raw videos and the re-identification
index are required when the fixed cattle-photo cache must be rebuilt.

## Data integrity rules

- Stage1 `tracking_boxes.csv` and `keypoints.csv` files are read-only inputs.
  Display transformations are applied in memory and must not be written back to
  those CSVs.
- Frame identifiers are zero-based source-video frame indices. Event intervals
  use inclusive start and end frames.
- Tracking IDs are local to a single full-path video and cannot be joined across
  videos by ID alone.
- Any missing, ambiguous, or non-bijective path, segment, or identity mapping is
  a data-contract error.

## Tests

The commands below document the intended checks; they were not executed during
the source-only repository cleanup.

```bash
cd /home/hyw/DXW/WebUIL
source /home/hyw/.venvs/dcsna/bin/activate
python3 -m unittest discover -s tests -v

cd /home/hyw/DXW/Dashboard
source /home/hyw/.venvs/dcsna/bin/activate
MPLCONFIGDIR=/home/hyw/DXW/Dashboard/.cache/matplotlib \
  python3 -m unittest discover -s tests -v
```
