# CSV Segmenter

CSV Segmenter is a research utility for dividing large tracking and keypoint CSV files into frame-aligned shards. It processes only source directories that have a corresponding bounding-box visualization video, preserves original frame numbers, adds a 10-second overlap around internal shard boundaries, and writes machine-readable playback metadata without regenerating video.

## Repository contents

- `split_target_csvs.py`: the complete segmentation program.
- `DATA_ACCESS.md`: restrictions governing the commercial-farm data used by the project.
- `agents_r.md`: repository-maintenance instructions retained unchanged by project policy.

Restricted data, trained models, complete private configurations, and processed results are not part of this public repository.

## Target environment

The supported environment is the research group's Linux workstation. The workstation has NVIDIA RTX 5090 hardware, but this program performs CSV and filesystem processing only and does not use CUDA or PyTorch.

The program requires Python 3.9 or newer and uses only the Python standard library. No package installation is required.

Activate the research environment before running it:

```bash
cd ~/DCSNA
source ~/.venvs/dcsna/bin/activate
cd /path/to/CSV_SEGMENTER
python3 split_target_csvs.py
```

The program flushes status messages immediately and records progress in `output/split_log.txt`, including periodic updates during long CSV scans.

## Data paths

The input locations are intentionally fixed near the top of `split_target_csvs.py`:

```text
SOURCE_ROOT      /mnt/data4t/hyw/20260625
BBOX_VIDEO_ROOT  /mnt/data4t/hyw/20260626V_bbox
OUTPUT_ROOT      <repository>/output
```

On the research workstation, mount or restore the authorized data at these exact paths before running the program. Internal team members who need a restricted asset must retrieve it from the corresponding location under:

```text
Yiwen Huang SNA Project Summer 2026/private_assets/
```

Restore each asset to its original relative directory structure. Do not place `MOVE_IT/` in the restored path, and do not commit restricted assets to this repository. See [DATA_ACCESS.md](DATA_ACCESS.md).

## Expected input layout

Bounding-box visualization videos must be regular files directly inside `BBOX_VIDEO_ROOT` and must follow this exact naming convention:

```text
F{farm_id}_{camera}_{gx_id}_bbox.mp4
```

For example, a video key maps to the following source directory:

```text
BBOX_VIDEO_ROOT/F1_Gopro5_GX100006_bbox.mp4
SOURCE_ROOT/1/Gopro5/GX100006/
```

Names and letter case must match exactly on Linux. Each selected source directory must contain:

```text
tracking_boxes.csv
keypoints.csv
manifest.json
```

Both CSV files must be UTF-8 encoded, have a header containing `frame`, and use integer frame values. They must have identical frame sets and identical row counts for every frame. The current validation compares frame membership and per-frame row counts; it does not compare track identifiers between the two files.

The manifest must be a JSON object containing a positive numeric frame rate at:

```json
{
  "video_manifest": {
    "fps": 29.97002997002997
  }
}
```

Any other files or directories in the selected source directory are copied unchanged into every output shard.

## Segmentation behavior

The program applies the following rules:

1. It clears the entire repository-local `output/` path at startup.
2. It discovers top-level `*_bbox.mp4` files and requires an unambiguous source-directory mapping for each video.
3. It keeps all rows from a frame in the same shard.
4. It uses the same inclusive frame range for `tracking_boxes.csv` and `keypoints.csv`.
5. It preserves original frame numbers.
6. It limits every generated CSV to 500,000 lines, including its header.
7. It balances base shards by data-row count as closely as whole-frame boundaries allow.
8. It adds approximately 10 seconds of frames on both sides of each internal boundary while retaining the line limit.
9. It copies a CSV unchanged when only one shard is necessary.
10. It fails instead of silently accepting missing, duplicated, malformed, ambiguous, or inconsistent inputs.

Because `output/` is deleted before input discovery and validation, move any output that must be retained before starting another run.

## Output layout

Each output directory is named with a one-based shard index:

```text
output/
  {farm_id}/
    {camera}/
      {gx_id}_1/
        tracking_boxes.csv
        keypoints.csv
        playback_segment.json
        ...copied source files...
      {gx_id}_2/
        ...
  split_log.txt
```

`playback_segment.json` records the source visualization-video path, frame rate, base frame range, overlapped segment range, playback times, shard count, and expected CSV row count. The end frame is inclusive; `segment_end_seconds_exclusive` is the time immediately after that frame.

## Data and publication policy

Only source code, documentation, environment metadata, and explicitly approved small public examples belong in this repository. Commercial-farm data and derived records must remain in approved internal storage. Do not commit videos, tracking or keypoint CSV files, private manifests, trained weights, complete private configurations, logs, or generated output.

No open-source license is granted by this repository. Unless the repository owners provide separate written permission, the code remains subject to the default applicable copyright restrictions.
