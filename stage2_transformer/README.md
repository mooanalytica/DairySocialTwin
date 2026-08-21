# Stage 2 Transformer Interaction Analysis

This repository contains the Stage 2 interaction-analysis workflow for tracked cattle. It consumes Stage 1 tracking and keypoint CSV files, predicts whether a pair is interacting, and then classifies detected interactions as friendly or unfriendly.

Stage 1 detection, tracking, pose estimation, identity classification, and CSV generation are outside this repository's scope.

## Architecture

The two-stage cascade uses separate temporal Transformer classifiers:

1. The interaction gate (`PTIHead`) consumes paired keypoint trajectories directly and predicts interaction versus no interaction.
2. The valence classifier (`ValenceTransformer`) consumes the existing hand-crafted pair-feature sequence and predicts friendly versus unfriendly for interactions that pass the gate.

Both classifiers use `--pti-max-frames` to limit temporal sequence length. They are trained and stored as separate checkpoints.

## Repository layout

- `train_stage2_valence_inception.py`: trains and evaluates the interaction gate and valence Transformer.
- `stage2_pti_head.py`: paired-trajectory Transformer interaction gate.
- `stage2_pair_features.py`: pair-local temporal feature construction.
- `stage2_dynamic_threshold.py`: dynamic interaction-threshold utilities.
- `run_stage2_from_csv.py`: Stage 2 inference from existing Stage 1 CSV outputs.
- `tests/test_pti_head.py`: focused tests for the paired-trajectory interaction head.

## Workstation environment

The supported environment is the Linux dual-RTX-5090 workstation. Use GPU 1 through the process environment; do not change CUDA device selection inside the PyTorch code.

```bash
cd ~/DCSNA
source ~/.venvs/dcsna/bin/activate
python3 -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build compatible with the workstation separately. Run GPU work with:

```bash
CUDA_VISIBLE_DEVICES=1 python3 <script.py> <arguments>
```

The site data roots used by the current master index are:

- `/mnt/data4t/hyw/20250530`
- `/mnt/data4t/hyw/20260604`

Legacy command-line defaults may be local placeholders. Pass the applicable Linux paths explicitly when a command requires a data root.

## Restricted assets

Training data, annotations, the authoritative master index, trained checkpoints, and generated results are not distributed in this repository. Authorized project members can retrieve the staged assets from the internal Dropbox location:

```text
Yiwen Huang SNA Project Summer 2026/private_assets/stage2_transformer/
```

Restore each asset to the same relative path in the repository, removing the `private_assets/stage2_transformer/` prefix. The active workflow expects at least:

```text
stage2_index/stage2_master_index.csv
stage2_index/stage2_index_build_summary.json
stage2_index/stage2_split_summary.csv
models_new/stage2_interaction_gate_best.pt
models_new/stage2_valence_transformer_best.pt
```

See `DATA_ACCESS.md` for handling restrictions.

## Training

The authoritative master index defines the included clips and data split. Entries marked as ignored by the index build summary remain excluded.

```bash
CUDA_VISIBLE_DEVICES=1 python3 train_stage2_valence_inception.py \
  --task two_stage_cascade \
  --stage2-index-csv stage2_index/stage2_master_index.csv \
  --profile 32gb
```

Use `--pti-max-frames` to set the maximum sequence length used by both Transformer classifiers. Run long commands with unbuffered output (`python3 -u`) when logs must be monitored in real time.

## Stage 2 inference

`run_stage2_from_csv.py` operates only on existing Stage 1 outputs; it does not run detection, tracking, or pose estimation.

```bash
CUDA_VISIBLE_DEVICES=1 python3 run_stage2_from_csv.py \
  --input-root <stage1-output-root> \
  --interaction-gate-ckpt models_new/stage2_interaction_gate_best.pt \
  --valence-ckpt models_new/stage2_valence_transformer_best.pt
```

Use full, unambiguous video paths. A camera filename such as `GX010007.MP4` is not a unique video identifier. When a processed basename contains a content identifier, such as `SV_GX010006_a89ee5282cc5`, resolve that identifier through the annotation/index metadata to the actual full video path. If the mapping is missing or ambiguous, stop with an error.

## Data conventions

- `frame`, `start_frame`, and `end_frame` are zero-based source-video frame indices.
- Event intervals are inclusive: `duration_s = (end_frame - start_frame + 1) / fps`.
- Track IDs are local to one video. Always combine a track ID with the resolved video path when identifying an animal trajectory.
- Tracking boxes and keypoints use original input-frame pixel coordinates, not coordinates from a resized annotated video.
- `tracking_boxes.csv` stores boxes as `x`, `y`, `w`, and `h`, with the origin at the upper-left.
- `keypoints.csv` stores all 27 pose points as `kpt_i_x`, `kpt_i_y`, and `kpt_i_conf`.

Generated checkpoints, logs, videos, indexes, and Stage 1/Stage 2 CSV products are intentionally excluded by `.gitignore`.
