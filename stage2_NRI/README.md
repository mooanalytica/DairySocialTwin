# Stage 2 NRI Interaction Analysis

This repository contains the Stage 2 interaction-analysis code for dairy-cow keypoint trajectories. Stage 1 perception is outside the scope of this repository. Stage 2 consumes existing tracking and keypoint CSV files and predicts whether a pair interacts and, conditionally, whether the interaction is friendly or unfriendly.

## Model architecture

The production workflow uses two separately trained neural models:

1. An NRI interaction gate predicts video-level and pair-level interaction probabilities.
2. An NRI valence classifier runs on likely interacting pairs and predicts friendly or unfriendly interaction conditional on interaction.

The interaction loss treats annotated pairs in positive videos as positive without treating unannotated pairs as negative. All valid pairs in a no-interaction video may be used as negative pairs. Valence supervision is applied only to positive interaction samples.

`CascadeInteractionModel` in `nri_interaction_head.py` is a modular reference implementation that combines both heads. The production training workflow uses separate gate and valence checkpoints; it is not described as an end-to-end trained cascade.

## Repository layout

- `nri_interaction_head.py`: NRI encoders, message-passing heads, cascade modules, and loss functions.
- `train_stage2_valence_inception.py`: current NRI gate and NRI valence training entry point. The historical filename is retained for compatibility with existing imports and commands.
- `run_stage2_from_csv.py`: Stage 2 inference from `tracking_boxes.csv` and `keypoints.csv`.
- `stage2_pair_features.py`: pair-local feature construction used by the training sample pipeline.
- `stage2_dynamic_threshold.py`: validation-time calibration and inference-time valence thresholds.

## Workstation environment

The supported environment is the project group's Linux workstation with an NVIDIA 5090 GPU. Activate the existing environment and expose the second GPU:

```bash
cd ~/DCSNA
source ~/.venvs/dcsna/bin/activate
export CUDA_VISIBLE_DEVICES=1
```

Use a CUDA-enabled PyTorch installation compatible with the workstation. PyTorch is intentionally not pinned in `requirements.txt` because its package build must match the installed CUDA runtime. Install the remaining Stage 2 dependencies with:

```bash
python -m pip install -r requirements.txt
```

Training and inference require CUDA and intentionally refuse CPU execution.

## Restricted assets

The public repository does not contain farm data, annotations, the master index, trained checkpoints, or processed evaluation results. Authorized project members can retrieve the corresponding directory from:

```text
Yiwen Huang SNA Project Summer 2026/private_assets/stage2_NRI/
```

Restore files to their original paths under the project root. Do not retain the staging prefixes `MOVE_IT/` or `stage2_NRI/`. The expected restored locations include:

```text
models_new/stage2_interaction_gate_best.pt
models_new/stage2_valence_nri_best.pt
stage2_index/stage2_master_index.csv
stage2_index/stage2_split_summary.csv
stage2_index/stage2_index_build_summary.json
```

The checkpoints and index files may contain restricted metadata and must not be redistributed. See `DATA_ACCESS.md`.

## Data paths

The restored master index references authorized datasets mounted at:

```text
/mnt/data4t/hyw/20250530
/mnt/data4t/hyw/20260604
```

The master index stores full, unambiguous paths to the source video, annotation, Stage 1 manifest, tracking CSV, and keypoint CSV. A bare camera filename such as `GX010006.MP4` is not a valid cross-dataset identifier.

Some command-line defaults are site-local placeholders. For reproducible runs, restore the canonical private assets and pass explicit input and output paths where applicable. Stage 2 output directories must remain below the project root.

## Training

Train the default two-checkpoint NRI workflow:

```bash
CUDA_VISIBLE_DEVICES=1 python train_stage2_valence_inception.py \
  --task two_stage_cascade \
  --stage2-index-csv stage2_index/stage2_master_index.csv \
  --profile 32gb
```

The command first trains `NRIInteractionHead`, then initializes and trains `NRIValenceHead`. The canonical checkpoints are written under `models_new/`. Training logs are written under `run_logs/`; both directories are intentionally excluded from the public repository.

## Inference from Stage 1 CSV files

```bash
CUDA_VISIBLE_DEVICES=1 python run_stage2_from_csv.py \
  --input-root /path/to/stage1_outputs \
  --output-dir ./output_s2 \
  --interaction-gate-ckpt models_new/stage2_interaction_gate_best.pt \
  --valence-ckpt models_new/stage2_valence_nri_best.pt
```

Each input sample folder must contain `tracking_boxes.csv` and `keypoints.csv`. If present, `manifest.json` supplies video metadata. Output may include `interactions.csv`, `no_interactions.csv`, per-class adjacency CSV files, proximity summaries, and a Stage 2 manifest.

Frame indices are zero-based. Event intervals are inclusive, so event duration is `(end_frame - start_frame + 1) / fps`. Bounding boxes and keypoints remain in the original Stage 1 input-frame coordinate system.

Generated logs, caches, checkpoints, indexes, media, and processed results are excluded through `.gitignore`. Restore the required private inputs as described above; runtime outputs are written by the training and inference workflows.
