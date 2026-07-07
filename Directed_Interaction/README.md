# Directed Graph Annotator (for Sibi)

This repository contains a first-frame cow bounding-box annotation GUI for interaction clips.
It scans interaction videos, prepares detector-based initial boxes, lets the annotator adjust boxes and assign cow names, and saves annotation text files for each clip.

## Repository Contents

- `bbox_annotator.py`: Tkinter GUI for first-frame annotation.
- `dcsna_local/requirements_windows.txt`: Python package requirements used by the local detector workflow.
- `output/**/bbox.csv`: retained text annotation files.
- `output/**/metadata.json`: retained text metadata files for the annotations.

Large model weights, logs, cache files, generated visualizations, and intermediate export folders are intentionally not included in the cleaned repository.

## Required Local Files

The detector weight is not stored in this repository. Obtain it from Sibi's project and place it at:

```text
dcsna_local/models/Object_Detection_Trained_Model.pt
```

The GUI reads videos recursively from this default root:

```text
F:\DROPBOX\Isolated_Interaction_Clips\Interaction_Type
```

The cow-name filter spreadsheet is read-only and is expected at:

```text
D:\OneDrive\F4_COOP\Directed_Interaction_0516_E.xlsx
```

Both paths can be overridden with command-line arguments.

## Path Configuration

The hard-coded defaults in `bbox_annotator.py` are workstation defaults, not repository requirements.
Keep them when working on the original annotation workstation, or override them from the command line on another machine.

| Purpose | Default path | How to change it |
| --- | --- | --- |
| Video source root | `F:\DROPBOX\Isolated_Interaction_Clips\Interaction_Type` | Pass `--source-root` |
| Annotation output root | `./output` relative to this repository | Pass `--output-root` |
| Detector weight | `dcsna_local/models/Object_Detection_Trained_Model.pt` | Pass `--det-weights`, or place the weight at the default path |
| Cow-name filter spreadsheet | `D:\OneDrive\F4_COOP\Directed_Interaction_0516_E.xlsx` | Pass `--cow-excel` |
| Runtime log file | `logs/bbox_annotator.log` | Pass `--log-file` |

Retained `metadata.json` files may include absolute paths from the workstation where the annotations were originally saved.
Those paths are provenance records for the existing annotations. New annotations will use the paths configured for the current run.

## Environment

On the Windows workstation, activate the existing environment before launching the GUI:

```powershell
conda activate DCSNA
python -u bbox_annotator.py
```

The `-u` flag keeps stdout flushing visible in real time. The program also writes a flushed log file under `logs/` during normal use.

## Usage

Optional arguments:

```powershell
python -u bbox_annotator.py `
  --source-root "F:\DROPBOX\Isolated_Interaction_Clips\Interaction_Type" `
  --output-root ".\output" `
  --det-weights ".\dcsna_local\models\Object_Detection_Trained_Model.pt" `
  --cow-excel "D:\OneDrive\F4_COOP\Directed_Interaction_0516_E.xlsx" `
  --vram-profile 8GB
```

The available cow-name choices are `Bella`, `Daisy`, `Rosie`, `Buttercup`, `Marigold`, and `NO NAME`.
For each video, the dropdown is filtered by the matching row in the Excel sheet when a matching interaction category and number are available.

## Outputs

For each annotated video, the GUI writes:

```text
output/<interaction_class>/<interaction_id>/bbox.csv
output/<interaction_class>/<interaction_id>/metadata.json
output/<interaction_class>/<interaction_id>/vis.png
```

The cleaned repository keeps only the text annotation files (`bbox.csv` and `metadata.json`).
Visualization images (`vis.png`) are generated again when annotations are saved in the GUI.

Coordinates in `bbox.csv` are original input-frame pixel coordinates, not resized visualization coordinates.
The frame index is zero-based and the annotation frame is always frame `0`.

## Detector Provenance

Initial boxes are produced inside `bbox_annotator.py` with Ultralytics YOLO and the local detector weight listed above.
The `source` column in `bbox.csv` records whether each box came directly from the detector, was manually adjusted, or was manually added.

## Cleaned Files

The cleaned GitHub version excludes:

- local model weights (`*.pt`);
- generated visualization images (`vis.png`);
- runtime logs and cache folders;
- old coding-agent instruction files other than `agents_r.md`;
- the `bbox_for_1st_frame/` delivery/export folder;
- dead pipeline code that is not called by the GUI.
