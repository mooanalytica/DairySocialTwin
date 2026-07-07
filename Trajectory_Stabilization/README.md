# Trajectory Stabilization

This repository contains a lightweight bounding-box visualization workflow for dairy-cow interaction review. It does not run Stage 1 perception, detection, tracking, pose estimation, or model training. The retained scripts use existing Stage 1 CSV outputs, selected-ID CSVs, annotation records, and clean video clips to render the bounding boxes that were used by the downstream sample-construction logic.

## Repository Scope

The current repository keeps only the code needed to regenerate bounding-box visualization videos from existing external data:

- `visualize_stage2_used_bbox.py` renders the selected pair bounding boxes, annotation ROI overlays, and a CSV sidecar for each rendered sample.
- `train_stage2_valence_inception.py` is retained as a helper module because the visualization script imports its data classes, CSV loaders, path normalization, sample construction, and selected-ID repair logic. In this repository layout it is not intended to be used for training.
- `stage2_pair_features.py` contains shared pair-feature encoding utilities used by the helper module.
- `requirements_windows.txt` lists the minimal Python packages used by the retained visualization path.

The removed files are generated outputs, Stage 1 perception code, Stage 1 model weights, cluster job scripts, vendored perception frameworks, old agent instructions, and runtime cache files.

## External Data

The visualization workflow expects external data folders to be supplied by command-line arguments or by editing the default paths in `visualize_stage2_used_bbox.py`. The checked-in defaults are historical local paths and should be treated as examples of the required folder roles:

- `--index-csv`: a `visualization_index.csv` file that maps reviewed clips to annotation CSVs, clean videos, Stage 1 `tracking_boxes.csv`, Stage 1 `keypoints.csv`, and valence labels.
- `--clean-video-root`: clean source clips used for rendering.
- `--v0520-root`: Stage 1 output root containing `tracking_boxes.csv`, `keypoints.csv`, and `manifest.json`.
- `--annotation-root`: annotation CSV root.
- `--selected-id-root`: selected-ID CSV root containing `frame` and track-pair choices.
- `--bbox-good-root`, `--bbox-goodtest-root`, `--bbox-bad-root`: reviewed clip-list folders used to filter and label visualization samples.
- `--out-dir`: output folder for regenerated visualization videos and sidecar CSV files.

Generated videos and generated CSV sidecars are not committed. Recreate them by running the visualization script against the external data.

## Model Files

No Stage 1 model weights are included. The retained visualization path does not require detection, tracking, pose-estimation, or identity model weights.

Stage 2 checkpoints are also not included in this cleaned repository. The visualization helper can read keypoint metadata from a checkpoint passed with `--template-ckpt`, but if that file is absent the helper falls back to 27 default keypoint names and default metadata. Supplying a checkpoint is optional for the bbox-only visualization workflow.

## Environment

Use the project Python environment prepared for this codebase. On Windows, the expected environment name has historically been `DCSNA`.

Install the minimal retained dependencies with:

```powershell
pip install -r requirements_windows.txt
```

If PyTorch is already managed separately in the environment, keep the existing PyTorch installation and install the remaining packages as needed.

## Usage

The script is configured through explicit paths. A typical invocation looks like this:

```powershell
python visualize_stage2_used_bbox.py `
  --index-csv "Z:\BBOX_VIS-V0604\output\visualization_index.csv" `
  --clean-video-root "Z:\Interaction_Annotator-V0601_03_AND_05R\output\videos" `
  --v0520-root "Z:\V0604_SEG_S1_TI" `
  --annotation-root "Z:\Interaction_Annotator-V0601_03_AND_05R\output\annotations" `
  --selected-id-root "Z:\BBOX_VIS-V0604\output_selected_IDs" `
  --bbox-good-root "Z:\BBOX_VIS-V0604\GOOD" `
  --bbox-goodtest-root "Z:\BBOX_VIS-V0604\GOODTEST" `
  --bbox-bad-root "Z:\BBOX_VIS-V0604\BAD" `
  --out-dir "output\stage2_used_bbox_vis_v0604_good"
```

Use `--list-samples` first when checking that the index and external folders resolve to the expected samples. The renderer writes regenerated MP4 files and sidecar CSV files under `--out-dir`.

## Coordinate Notes

The Stage 1 CSV coordinates are original frame pixel coordinates. The renderer opens the clean clip and scales boxes according to the manifest frame size when the render video resolution differs from the Stage 1 coordinate frame.

The output CSV sidecar records the pair boxes used for rendering and ROI metadata. It is generated output and should not be committed.

## Repository Hygiene

This cleaned repository intentionally excludes:

- generated visualization outputs,
- runtime logs and cache files,
- model checkpoints and large binary weights,
- Stage 1 perception code and vendored perception frameworks,
- cluster submission scripts,
- old coding-agent instruction files.

Those artifacts can be restored from external storage or regenerated from the external data pipeline when needed.
