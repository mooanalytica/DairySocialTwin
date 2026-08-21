# CowTrack

CowTrack is an offline, bbox-only cattle re-identification pipeline for a fixed
continuous multi-clip recording. It builds conservative micro-tracklets,
extracts local appearance embeddings, calibrates short- and long-gap link
models, constructs provisional global identities, and exports QA artifacts.

The production workflow is intentionally strict. It does not use keypoints or
legacy tracker IDs as identity evidence, does not silently change coordinate
systems, and does not fall back from GPU inference or NVENC rendering to CPU.

## Pipeline

| Stage | Purpose | Main outputs |
| --- | --- | --- |
| S00 | Validate video geometry and normalize bbox CSV rows | `frames.parquet`, `detections.parquet` |
| S01 | Build conservative bbox-only micro-tracklets | `det_to_micro.parquet`, `microtracklets.parquet` |
| S02 | Select crops and extract appearance embeddings | sample embeddings, micro-track prototypes |
| S03 | Calibrate short- and long-gap link scorers | model files, thresholds, pseudo-pair tables |
| S04 | Propose and finalize short-gap links | stable tracklets and stable appearance |
| S05 | Calibrate, propose, and solve long-gap global links | provisional global tracks and mappings |
| S06 | Export identities and visual QA | final CSV, reports, contact sheets, tracked videos |

Every stage validates its upstream artifacts and records deterministic
fingerprints in `_SUCCESS.json`. Stage outputs are immutable inputs to later
stages.

## Repository boundary

This public repository contains source code, tests, packaging metadata, and
documentation. It intentionally excludes:

- commercial farm video and bbox data;
- private manifests and recording indexes;
- full production configuration files;
- trained model checkpoints;
- generated Parquet, NumPy, model, report, image, HTML, and video artifacts;
- logs, PID files, caches, and compiled Python bytecode.

See [DATA_ACCESS.md](DATA_ACCESS.md) for the data-use restriction and
[SCHEMAS.md](SCHEMAS.md) for the public data contracts.

## Workstation environment

The fixed production runner expects the project and existing virtual
environment at these locations:

```text
/home/hyw/re-identification
/home/hyw/.venvs/trackID
```

Activate the environment before using the package:

```bash
cd /home/hyw/re-identification
source /home/hyw/.venvs/trackID/bin/activate
```

The base dependency declarations are in `pyproject.toml`. A minimal editable
installation for CPU-only stage logic and tests is:

```bash
python3 -m pip install -e ".[test]"
```

Appearance stages additionally require the workstation's compatible PyTorch,
CUDA, timm, Pillow, torchvision, and safetensors runtime. The fixed S05 solver
contract requires Python 3.14.4 and SciPy 1.18.0. Use the team-managed
`trackID` environment rather than replacing its PyTorch build.

GPU work must expose physical GPU 1 as the single logical device:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u run_pipeline.py
```

Within that process, the code addresses the selected device as `cuda:0`.
Failure to access CUDA or `h264_nvenc` is fatal.

## Restoring private assets

Private assets are stored internally under:

```text
Yiwen Huang SNA Project Summer 2026/private_assets/re-identification/
```

Restore each item to its original repository-relative location. Do not retain
the `MOVE_IT/re-identification/` prefix when restoring it. The required layout
is:

```text
/home/hyw/re-identification/
|-- configs/
|   |-- production.yaml
|   |-- s01_microtrack.yaml
|   |-- s01_review.yaml
|   |-- s02_appearance.yaml
|   |-- s03_calibration.yaml
|   |-- s04_conflict_review.yaml
|   |-- s04_finalize.yaml
|   |-- s04_proposals.yaml
|   |-- s04_review.yaml
|   |-- s05_finalize.yaml
|   |-- s05_force_appearance.yaml
|   |-- s05_long_calibration.yaml
|   |-- s05_proposals.yaml
|   |-- s05_review.yaml
|   `-- s06_export.yaml
|-- data/
|   `-- manifest.csv
`-- gopro_time_sequence.csv
```

The fixed code also expects the private model and source data at these exact
locations:

```text
/home/hyw/re-identification-models/MegaDescriptor-L-384/pytorch_model.bin
/home/hyw/UPAN_HYW/May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO/
/home/hyw/UPAN_HYW/stage1_output/1/Gopro1/<clip_id>/tracking_boxes.csv
```

The checkpoint directory must also contain the model metadata files validated
by the production runner. No network download or alternate-path fallback is
used. The model weights are not part of this repository and must not be
redistributed without an independently verified license.

## Running the fixed workflow

After restoring all private assets, the complete fixed workflow can be started
with:

```bash
cd /home/hyw/re-identification
source /home/hyw/.venvs/trackID/bin/activate
CUDA_VISIBLE_DEVICES=1 python3 -u run_pipeline.py
```

Individual stages are exposed through the `cowtrack` command. Their arguments
are defined in `src/cowtrack/cli.py`. Each stage is fail-closed: a missing file,
changed fingerprint, unexpected schema, ambiguous mapping, or incompatible
runtime aborts the operation.

## Tests

The test suite imports the private production configs to verify exact schema,
hash, and cross-stage contracts. Restore `configs/` and `data/manifest.csv`
before running the complete suite:

```bash
source /home/hyw/.venvs/trackID/bin/activate
python3 -m pytest
```

The repository preparation process does not execute tests or import project
modules. Test commands above are for an authorized workstation after private
assets have been restored.

## Coordinate and identity rules

- Source frame indices are zero-based.
- Bboxes use the source video's raw encoded landscape coordinates.
- S00 converts input `x, y, w, h` values to `x1, y1, x2, y2` without changing
  the video orientation.
- Video autorotation is disabled throughout the fixed pipeline.
- Invalid source rows are retained for audit but receive no fabricated
  identity.
- Legacy tracker IDs may be preserved for audit but are never identity
  features or training labels.
- Keypoints are outside this pipeline's identity evidence.
- Global IDs are provisional and must not be represented as certified animal
  identities.

## Generated files

The following paths are generated and intentionally excluded from source
control:

- `work/`: all S00-S06 intermediate and final stage artifacts;
- `final/`: published CSV and tracked-video deliverables;
- `logs/`, `*.log`, and `*.pid`: runtime logs and process markers;
- `__pycache__/`, `*.pyc`, and `.pytest_cache/`: Python and pytest caches;
- `*.egg-info/`, `build/`, and `dist/`: package build metadata;
- review videos, contact sheets, embeddings, models, and reports produced
  beneath stage output directories.

Restore the private inputs and rerun the relevant stage or the fixed workflow
to regenerate these files.

