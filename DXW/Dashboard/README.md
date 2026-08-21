# Dairy Social Network Dashboard

Standalone, read-only WebUI for the precomputed Farm 1 / Camera 1 social
network analysis result `F1_Gopro1_20250505`.

Restricted data, processed results, and runtime-ready caches are not distributed
through Git. Restore them from the internal Dropbox folder
`Yiwen Huang SNA Project Summer 2026/private_assets/` to the exact paths below.
See the repository-level `README.md` and `DATA_ACCESS.md` before starting the
service.

The Dashboard reads:

- `/home/hyw/DXW/WebUIL/sna_inputs/F1_Gopro1_20250505`
- `/home/hyw/DXW/WebUIL/sna_outputs/F1_Gopro1_20250505`
- `/home/hyw/FloorPlanAnnoEN/output15/farm_ID_1_camera_ID_1`
- `/home/hyw/time_sequence_detector_v2/output/1_1.csv`
- `/home/hyw/DXW/WebUIL/.cache/reidentification_1-1.sqlite3` (photo-cache build only)
- `/mnt/dairycow_sna/FULLDATA/Dairy Farm Videos/May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO` (photo-cache build only)

It does not run Stage1, Stage2, community detection, re-identification, or any
other social network analysis. Net/dominance transformation, community-label
alignment, cattle filtering, and trajectory sampling are plotting operations
copied from the existing WebUIL figure generator.

## Start

```bash
cd /home/hyw/DXW/Dashboard
./run_dashboard.sh
```

From another machine on the LAN, open:

```text
http://172.17.6.39:2299/
```

Before binding the HTTP server, startup validates or creates deterministic
Figure 1, displayed-appearance, and fixed cow-photo caches under
`/home/hyw/DXW/Dashboard/.cache`. The figure and appearance caches follow their
source files and processing contracts. Each cow photo is a checksum-verified
snapshot bound to the current generation and that cow's first listed
appearance; once generated, it deliberately stays the same. The stabilized
WebUIL display trajectory (including frozen display points) remains the
appearance authority. Raw videos and detection-level re-identification data
are needed only if the 62-photo snapshot must be built anew.

## Test

```bash
cd /home/hyw/DXW/Dashboard
source /home/hyw/.venvs/dcsna/bin/activate
MPLCONFIGDIR=/home/hyw/DXW/Dashboard/.cache/matplotlib \
  python3 -m unittest discover -s tests -v
```
