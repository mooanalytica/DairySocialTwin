# SNA Zone Annotator

SNA Zone Annotator is a lightweight local WebUI for drawing semantic
social-network-analysis zones on dairy-cow reference frames. It supports three
farms with five camera views per farm, while keeping each farm-camera zone set
independent.

The annotator is separate from floorplan annotation and Stage2 processing. It
does not read or write `floorplan_annotation.json` and does not change the
reference frames.

## Data access

The reference frames and all real farm annotations are restricted project
data. They are not distributed with this repository and must not be
redistributed. See [DATA_ACCESS.md](DATA_ACCESS.md).

The configured reference-frame root on the group workstation is:

```text
/home/hyw/FloorPlanAnnoEN/output15
```

It must contain:

```text
floorplan_groups.json
farm_ID_1_camera_ID_1/reference_frame.png
farm_ID_1_camera_ID_1/reference_frame_meta.json
...
farm_ID_3_camera_ID_5/reference_frame.png
farm_ID_3_camera_ID_5/reference_frame_meta.json
```

The path is defined by `SOURCE_OUTPUT_ROOT` in `app_sna.py`. Update that
constant if the private data is mounted elsewhere.

Existing private SNA annotations are stored internally under:

```text
Yiwen Huang SNA Project Summer 2026/private_assets/FloorPlanAnnoSR/output15
```

To restore them, copy the contents back to `output15` at the repository root.
Do not include the intermediate `MOVE_IT/FloorPlanAnnoSR` directories in the
restored path.

## Environment

Use the shared DCSNA environment on the 5090 workstation:

```bash
cd ~/DCSNA
source ~/.venvs/dcsna/bin/activate
cd FloorPlanAnnoSR
python -m pip install -r requirements.txt
```

The WebUI itself does not run model inference or require a GPU. Shapely is used
for reliable positive-area polygon-overlap validation.

## Run

From the repository root:

```bash
source ~/.venvs/dcsna/bin/activate
python app_sna.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/` in a browser. When the browser is on another
computer, forward the loopback port over SSH before opening the address:

```bash
ssh -L 8765:127.0.0.1:8765 USER@WORKSTATION
```

An initial farm-camera group can be selected with:

```bash
python app_sna.py --host 127.0.0.1 --port 8765 \
  --group farm_ID_1_camera_ID_1
```

## Annotation workflow

The single group selector lists the 15 expected farm-camera combinations. Each
group has its own zone collection. A zone contains a stable ASCII `zone_id` and
`zone_type`, an arbitrary-language display `label`, a color, and polygon points.

When switching groups with unsaved changes, the UI offers three choices:

- **Save & Switch** saves the current group before loading the next group.
- **Discard** loads the next group without saving the current edits.
- **Cancel** stays in the current group.

Positive-area overlap between zones is rejected. Shared boundaries are
allowed.

## Outputs

Saving a group writes three private files:

```text
output15/farm_ID_1_camera_ID_1/sna_zones.json
output15/farm_ID_1_camera_ID_1/sna_zones_preview.svg
output15/farm_ID_1_camera_ID_1/SNA_ZONES_CONVENTION.md
```

The JSON file contains the manual annotation. The SVG preview and convention
document are regenerated whenever that group is saved. All three files remain
restricted because they encode information derived from commercial farm data.

## Coordinate convention

Zone polygons use `[x, y]` pixels on the clean `reference_frame.png` for the
selected group:

- Coordinate system: `reference_frame_pixel`
- Reference-frame size: 3840 x 2160 pixels
- Origin: top-left corner
- X axis: positive to the right
- Y axis: positive downward
- Unit: pixel
- Resize, crop, perspective correction, or homography: none

Downstream cattle positions must be mapped into the same reference-frame pixel
coordinate system before zone lookup.

## JSON schema

`sna_zones.json` uses schema `sna_zones.v1`. Its main fields are:

```json
{
  "schemaVersion": "sna_zones.v1",
  "groupId": "farm_ID_1_camera_ID_1",
  "farm": "1",
  "camera": "Gopro1",
  "coordinate_system": "reference_frame_pixel",
  "zones": [
    {
      "zone_id": "water_1",
      "zone_type": "water",
      "label": "Water",
      "color": "#1864ab",
      "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]
    }
  ]
}
```

## Tests

The Python tests use the standard-library `unittest` runner:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The optional browser test requires Node.js, Playwright, and an installed
Chromium-compatible browser:

```bash
npm install
PLAYWRIGHT_BROWSER_PATH=/path/to/chromium npm run test:ui
```

## Repository layout

```text
app_sna.py                 SNA annotation server and validation logic
static/index.html          WebUI markup
static/app_sna.js          WebUI interaction logic
static/app.css             WebUI styling
tests/test_app_sna.py      Backend and group-isolation tests
tests/test_app_sna_ui.cjs  Browser interaction test
DATA_ACCESS.md             Restricted-data policy
requirements.txt           Python dependency declaration
package.json               Optional browser-test dependency declaration
```
