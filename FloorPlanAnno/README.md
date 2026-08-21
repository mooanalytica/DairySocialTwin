# FloorPlanAnno

FloorPlanAnno is a local browser-based tool for annotating reference frames from
15 dairy-farm camera views. The interface supports switching between floor-plan
groups, drawing semantic polygons, marking polygons for rectangular 2D
interpretation, and saving machine-readable annotations with SVG previews and
coordinate-convention documentation.

This repository is intended for the research group's Linux-based NVIDIA RTX 5090
workstation. It contains source code and documentation only. Commercial farm
videos, extracted frames, completed annotations, manifests, and other processed
results are restricted and are not distributed through GitHub. See
[DATA_ACCESS.md](DATA_ACCESS.md).

## Repository layout

```text
FloorPlanAnno/
|-- app15.py
|-- static/
|   |-- app.css
|   |-- app.js
|   `-- index.html
|-- tools/
|   `-- setup_output15.py
|-- DATA_ACCESS.md
`-- README.md
```

The following private directories are required for the complete internal
dataset, but are intentionally excluded from the repository:

```text
output15/
outputs/
```

`output15/` contains the completed manual annotations for all 15 `(farm ID,
camera ID)` pairs. `outputs/` is the preserved legacy source for Farm 1, Camera
1. Neither directory is public.

## Workstation environment

Use the research group's existing `dcsna` environment:

```bash
cd ~/DCSNA/FloorPlanAnno
source ~/.venvs/dcsna/bin/activate
export CUDA_VISIBLE_DEVICES=1
```

The application itself uses only the Python standard library and does not invoke
GPU libraries. Python 3.10 or newer is required. `ffmpeg` must be available on
`PATH` when reference frames are extracted.

## Restore restricted assets

Authorized team members can retrieve the private project directory from:

```text
Yiwen Huang SNA Project Summer 2026/private_assets/FloorPlanAnno/
```

Replace `<private-assets-root>` with the Linux path to the internal
`private_assets` directory, then restore both directories to the repository root:

```bash
FLOORPLAN_PRIVATE_ROOT="<private-assets-root>/FloorPlanAnno"
cp -a "$FLOORPLAN_PRIVATE_ROOT/output15" .
cp -a "$FLOORPLAN_PRIVATE_ROOT/outputs" .
```

Restore the original directory structure shown above. Do not copy an enclosing
`MOVE_IT/` directory into the repository. Completed manual annotations cannot be
reconstructed by running the source code.

Each private group directory has this layout:

```text
output15/farm_ID_<farm-id>_camera_ID_<camera-id>/
|-- reference_frame.png
|-- reference_frame_meta.json
|-- floorplan_annotation.json
|-- floorplan_annotation_preview.svg
`-- COORDINATE_CONVENTION.md
```

## Start the annotation server

After restoring `output15/`, start the local service:

```bash
python app15.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/` in a browser. When accessing the workstation over
SSH, create a local tunnel from the client machine:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<5090-workstation>
```

Stop the service with `Ctrl+C` when annotation work is finished.

## Rebuild reference frames

Reference frames may be rebuilt only by researchers who are authorized to
access the underlying videos. Replace `<restricted-video-root>` with the Linux
path containing the three farm video directories:

```bash
python tools/setup_output15.py \
  --source-root "<restricted-video-root>" \
  --video-index 1 \
  --timestamp 00:01:00
```

The setup tool discovers Farm IDs 1 through 3 and Camera IDs 1 through 5. For
each camera, it selects the second MP4 in case-insensitive path order because
`--video-index` is zero-based. It extracts one frame at `00:01:00` without
automatic rotation, resizing, cropping, or perspective correction.

Restore `outputs/` before rebuilding because it supplies the preserved completed
annotation for Farm 1, Camera 1. The setup tool can regenerate frames, metadata,
empty annotation structures, previews, and coordinate documents; it cannot
reproduce the completed manual annotations for the other camera views. Restore
the complete `output15/` directory to use those annotations.

## Annotation format

Coordinates are `[x, y]` pixels in each group's `reference_frame.png`:

- The origin is the top-left corner.
- X increases to the right.
- Y increases downward.
- No homography is applied by the annotation tool.
- `resource` overrides `obstacle`, which overrides `walkable_ground` in overlap
  resolution.
- A polygon with `force2DRectangle: true` must be interpreted using the
  minimum-area rotated bounding rectangle derived from its polygon points.
- Pixels not covered by an explicit polygon remain unlabeled unless
  `defaultUnannotatedClassId` specifies otherwise.

The `image.file` field is relative to the repository root, for example:

```text
output15/farm_ID_1_camera_ID_1/reference_frame.png
```

## Data handling

Do not commit real farm videos, frames, annotations, metadata, manifests,
processed results, or trained models. Keep them in the internal Dropbox location
and restore them only inside an authorized working copy.
