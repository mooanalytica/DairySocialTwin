# Interaction Annotator

Interaction Annotator is a local review tool for Stage-2 dairy-cow interaction annotation. It builds full-frame temporal review clips from Stage-2 `interactions.csv` intervals, shows the clips in a browser UI, and saves per-clip ROI annotation CSV files.

The repository contains only source code and documentation. Generated clips, logs, indexes, cache files, and annotation outputs are not intended to be committed.

## Repository Layout

- `server.py` - local HTTP server for the annotation UI.
- `web/` - browser UI files.
- `tools/prepare_review_clips.py` - builds the review clip index, clean cache clips, and UI-only visualization clips.
- `tools/cleanup_pending_deletes.py` - retries deletion of files that Windows could not remove while the UI/server had them open.

## Requirements

- Python 3.10 or newer.
- FFmpeg and FFprobe available on `PATH`.
- NVIDIA NVENC support for the default `h264_nvenc` encoder.
- OpenCV for Python (`cv2`) when generating `output/cache_vis` overlays.
- `tqdm` is optional; the exporter falls back to periodic text progress.

## Data Paths

The current default local dataset paths are:

- Stage-2 roots: `F:\V0531_R1_S1_S2`, `F:\V0601_S1_S2`, `F:\V0603_S1_S2`
- Source video root: `F:\FULLDATA\Dairy Farm Videos`

These paths are local workstation defaults. For another workstation, either edit the constants at the top of `tools/prepare_review_clips.py` or pass paths on the command line:

```powershell
python -u -B tools\prepare_review_clips.py `
  --s2-root "F:\V0531_R1_S1_S2" `
  --s2-root "F:\V0601_S1_S2" `
  --s2-root "F:\V0603_S1_S2" `
  --source-root "F:\FULLDATA\Dairy Farm Videos"
```

Use `--tracking-root` only when tracking/keypoint CSVs live under a separate root from the interaction CSVs.

## Generated Output

The exporter and UI create `output/` on demand. This directory is generated data and should not be committed.

Important generated paths:

- `output/index.csv` - clip index consumed by the UI.
- `output/clip_plan.csv` - full planned clip table.
- `output/cached_videos/` - clean review clips used as source material.
- `output/cache_vis/` - UI-only clips with selection-track bounding boxes drawn.
- `output/videos/` - saved positive/negative annotation clips.
- `output/fake_interaction/` - saved clips explicitly labeled `fake_interaction`.
- `output/annotations/` - annotation CSVs created only after a clip is saved in the UI.
- `output/logs/` - runtime logs.
- `output/discarded_clips.txt` and `output/pending_deletes.csv` - UI deletion state.

Regenerate `output/` from the source data with `tools/prepare_review_clips.py`.

## Prepare Review Clips

Check that logs flush correctly before a long run:

```powershell
python -u -B tools\prepare_review_clips.py --check-log-flush
```

Preview the planned clips without exporting videos:

```powershell
python -u -B tools\prepare_review_clips.py --dry-run
```

Export all missing review clips:

```powershell
python -u -B tools\prepare_review_clips.py
```

Export only the next missing batch while keeping the full index:

```powershell
python -u -B tools\prepare_review_clips.py --next-missing 20
```

Candidate intervals are grouped and merged by the `class` column in each `interactions.csv`. The clean cache clips are kept separate from the UI visualization clips, so bbox overlays never modify the clips used for final output.

## Annotation UI

Start the local server:

```powershell
python -u -B server.py
```

Open:

```text
http://127.0.0.1:8765/
```

Annotation CSVs are created only when a clip is saved in the UI. Saving trims the selected frame range from the clean cache clip into `output/videos` or `output/fake_interaction`, then writes ROI rows with frame indices relative to the saved clip.

## Cleanup

If Windows refuses to delete an open clip, the UI removes it from the active dataset and records the physical file in `output/pending_deletes.csv`. After closing the UI/server, retry cleanup with:

```powershell
python -u -B tools\cleanup_pending_deletes.py
```
