# Latent State Engine (Stage 2) (Transformer-based)

This repository contains the Stage 2 interaction analysis code for dairy cow social interaction modeling. Stage 2 starts from precomputed Stage 1 CSV outputs and does not run detection, tracking, pose estimation, identity recognition, or video perception.

## Scope

Retained functionality:

- Train the two-stage Stage 2 cascade:
  - interaction gate: PTI Transformer over paired cow keypoints
  - valence classifier: Transformer over local pair features
- Evaluate the retained Stage 2 checkpoints on the master-index validation split.
- Run Stage 2 inference from existing `tracking_boxes.csv` and `keypoints.csv` folders to produce interaction CSV outputs.

Stage 1 perception outputs and Stage 1 weights are not part of this repository. They should be produced or obtained separately from Sibi's project. The Stage 2 code expects Stage 1 folders to already contain `manifest.json`, `tracking_boxes.csv`, and `keypoints.csv`.

## Repository Layout

- `train_stage2_transformer.py`: training and validation entry point for the Stage 2 Transformer cascade.
- `run_stage2_from_csv.py`: dataset/folder inference from precomputed Stage 1 CSV outputs.
- `stage2_pti_head.py`: PTI Transformer interaction-gate model.
- `stage2_pair_features.py`: local pair-feature encoder used by the valence Transformer.
- `stage2_dynamic_threshold.py`: cascade confidence-threshold calibration and prediction helpers.
- `stage2_index/stage2_master_index.csv`: retained authoritative split and data index.
- `models_new/stage2_interaction_gate_best.pt`: retained interaction-gate checkpoint.
- `models_new/stage2_valence_transformer_best.pt`: retained valence Transformer checkpoint.

## Environment

The intended environment is the lab Linux workstation with the 5090 GPU setup.

```bash
cd ~/DCSNA
source ~/.venvs/dcsna/bin/activate
pip install -r requirements.txt
```

Use GPU 1 for Stage 2 commands:

```bash
CUDA_VISIBLE_DEVICES=1 python train_stage2_transformer.py --help
```

## Data Paths

`stage2_index/stage2_master_index.csv` contains absolute paths to the retained training and validation data. On the workstation, those paths are expected to resolve under:

- `/mnt/data4t/hyw/20250530`
- `/mnt/data4t/hyw/20260604`

If the data is moved, update `stage2_index/stage2_master_index.csv` while preserving its schema. The code intentionally fails when required paths are missing or ambiguous.

## Evaluate Retained Checkpoints

This is the main validation/evaluation entry point. It evaluates the retained gate and valence checkpoints against the validation split from the master index, writes confusion/prediction CSV files, and refreshes dynamic-threshold metadata in the valence checkpoint.

```bash
CUDA_VISIBLE_DEVICES=1 python train_stage2_transformer.py \
  --eval-cascade-only \
  --stage2-index-csv stage2_index/stage2_master_index.csv \
  --interaction-gate-ckpt models_new/stage2_interaction_gate_best.pt \
  --valence-ckpt models_new/stage2_valence_transformer_best.pt \
  --profile 32gb
```

Generated evaluation CSVs are written under `models_new/` and are intentionally not retained as source files.

## Train Stage 2

```bash
CUDA_VISIBLE_DEVICES=1 python train_stage2_transformer.py \
  --task two_stage_cascade \
  --stage2-index-csv stage2_index/stage2_master_index.csv \
  --profile 32gb
```

By default, training writes:

- `models_new/stage2_interaction_gate_best.pt`
- `models_new/stage2_valence_transformer_best.pt`
- generated split/evaluation CSV files under `models_new/`
- run logs under `run_logs/`

For a dataset-construction check without training:

```bash
CUDA_VISIBLE_DEVICES=1 python train_stage2_transformer.py \
  --dry-run \
  --stage2-index-csv stage2_index/stage2_master_index.csv \
  --profile 32gb
```

## Run Stage 2 from CSV Folders

Use this entry point when Stage 1 has already produced folders containing `tracking_boxes.csv`, `keypoints.csv`, and optionally `manifest.json`.

```bash
CUDA_VISIBLE_DEVICES=1 python run_stage2_from_csv.py \
  --input-root /path/to/stage1_csv_outputs \
  --output-dir output_s2 \
  --interaction-gate-ckpt models_new/stage2_interaction_gate_best.pt \
  --valence-ckpt models_new/stage2_valence_transformer_best.pt \
  --profile 40gb
```

`run_stage2_from_csv.py` clears the selected output directory before processing. The output directory must stay under the repository root. Typical outputs include:

- `interactions.csv`
- `no_interactions.csv`
- `adjacency_<class>.csv`
- `proximity_percentiles.csv`
- `manifest.json`

## Generated Files

The repository intentionally excludes regenerated artifacts such as logs, Python bytecode, Stage 2 split CSVs, cascade validation CSVs, and inference output folders. Recreate them by running the commands above.
