# Dairy Floor Plan Web UI

This web UI draws FloorPlanAnno 2D structure diagrams, overlays Stage1 tracking
boxes as dairy-cattle points, and displays precomputed Stage2 interaction
inference.

Restricted data, full configurations, processed results, and runtime caches are
not distributed through Git. Restore them from the internal Dropbox folder
`Yiwen Huang SNA Project Summer 2026/private_assets/` to the exact paths listed
below. See the repository-level `README.md` and `DATA_ACCESS.md` before running
the service.

Prepare every active Stage1 segmented shard that has a matching current QA track-ID video:

```bash
cd /home/hyw/DXW/WebUIL
source /home/hyw/.venvs/dcsna/bin/activate
set -o pipefail
python3 -u generate_stage2_precomputed.py --verify-log-flush \
  2>&1 | tee logs/stage2_log_flush_check.log
python3 generate_stage2_precomputed.py --all --dry-run
python3 generate_video_cache.py --all --dry-run
CUDA_VISIBLE_DEVICES=1 python3 -u generate_stage2_precomputed.py --all --overwrite \
  --cuda-visible-devices 1 2>&1 | tee logs/stage2_precompute_gpu1_local.log
CUDA_VISIBLE_DEVICES=1 python3 -u generate_video_cache.py --all --overwrite \
  --encoder h264_nvenc --workers 2 2>&1 | tee logs/video_cache_gpu1_local.log
```

Generate the reusable neural-network checkpoint, then run SNA independently.
The checkpoint is the six-file directory under
`sna_inputs/F1_Gopro1_20250505/`; its CSV/JSON/YAML files are the durable
Stage2-to-SNA boundary and do not require pickle. These outputs are not served by
the WebUI; the WebUI remains only the coordinate-checking tool.

```bash
cd /home/hyw/DXW/WebUIL
source /home/hyw/.venvs/dcsna/bin/activate
set -o pipefail

# Full paired refresh: Stage2 neural inference uses CUDA, then SNA analysis runs and
# the current input/output pair is replaced atomically.
CUDA_VISIBLE_DEVICES=1 python3 -u generate_sna_precomputed.py --overwrite \
  2>&1 | tee logs/sna_full_gpu1_local.log

# Re-run only SNA from the existing checkpoint; no CUDA or neural model is loaded.
python3 -u generate_sna_precomputed.py --analysis-only --dry-run --overwrite
python3 -u generate_sna_precomputed.py --analysis-only --overwrite

# Rebuild the 23 PNG figure suite after every successful analysis rerun.
python3 generate_sna_tf_outputs.py --sample F1_Gopro1_20250505 --overwrite
```

To tune SNA without changing the neural checkpoint, copy its full config, edit
only analysis algorithm settings such as `network`, `community`, `isolation`,
`report`, or `events`, and pass it explicitly. Each output stores the resolved
config and hashes for reproducibility.

```bash
cp sna_inputs/F1_Gopro1_20250505/config.yaml sna_config_tuned.yaml
# Edit sna_config_tuned.yaml, then:
python3 -u generate_sna_precomputed.py --analysis-only \
  --analysis-config sna_config_tuned.yaml --overwrite
```

Run for LAN access:

```bash
cd /home/hyw/DXW/WebUIL
source /home/hyw/.venvs/dcsna/bin/activate
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python3 -u app.py --host 0.0.0.0 --port 9922 \
  2>&1 | tee logs/webui_9922.log
```

Open from another machine on the LAN:

```text
http://172.17.6.39:9922/
```

An optional deep link can select a ready Clip entry, load it, seek to one
canonical source-video frame, and start playback:

```text
http://172.17.6.39:9922/?farmID=1&cameraID=1&clipID=F1_Gopro1_GX010006_2&segmentID=GX010006-2-82066&frameID=82066
```

All five query parameters are required. `clipID` is the exact value shown in
the Clip selector. `segmentID` is
`<source_clip_id>-<one_based_shard_index>-<source_clip_frame>`, and `frameID`
must repeat that same 0-based source-clip frame without a segment offset.
Incomplete, malformed, mutually inconsistent, or unavailable targets are
ignored and preserve the normal manual-Load behavior. Ambiguous or internally
inconsistent canonical manifest data is a server error.

Current data rules are intentionally hard-coded:

- Candidate samples: active scope is currently hard-limited to `/mnt/data4t/hyw/Stage1_segmented/1/Gopro1/<GX...>_<shard>`.
- The active sample scope is controlled by `ACTIVE_SAMPLE_FARM_ID` and `ACTIVE_SAMPLE_CAMERA_ID` in `app.py`.
- Segment data: each directory must contain `tracking_boxes.csv`, `keypoints.csv`, `manifest.json`, and `playback_segment.json`.
- Sample keys are `F<farm>_<camera>_<GX...>_<shard>`; the trim range comes from `playback_segment.json`.
- Track-ID visualization sources: `/mnt/drive_bf/BF/re-identification-11B-GOOD/work/dairy_farm_1_gopro1_20250505_all11/06_export_complete_sequence/qa/videos/<GX...>_tracked.mp4`.
- A source is resolved strictly through the authoritative manifest full path, the re-ID clip index, and the QA export clip mapping; GX basename-only fallback is not allowed.
- Floor plans: `/home/hyw/FloorPlanAnnoEN/output15/farm_ID_<farm>_camera_ID_<camera>/floorplan_annotation.json`.
- Resource floorplan zones: `/home/hyw/FloorPlanAnnoSR/output15/farm_ID_1_camera_ID_1/sna_zones.json`.
- Stage2 code and checkpoints: `/home/hyw/DCSNA-ALL`.
- Stage2 WebUI results must be generated under `/home/hyw/DXW/WebUIL/.cache/stage2_precomputed`; external Stage2 outputs are ignored.
- Browser track-ID video cache must be generated under `/home/hyw/DXW/WebUIL/.cache/track_id_videos`; QA source videos and the old bbox cache are not served as fallbacks.
- A sample is selectable only after both its WebUIL cache Stage2 JSON and its low-bitrate 960x540 H.264 video cache are ready.
- The UI initially shows only the sample selectors and an empty workspace. Press `Load` beside `Warp` to load the selected trajectory and track-ID video.
- During `Load`, the status line reports the backend's current step and real progress, for example CSV row counts or frame timeline counts.
- Video display uses the pre-trimmed cache and loops the whole cached shard.
- The bottom timeline controls video time. The floorplan follows video playback time using the sample FPS; there is no manual Sync button.
- Bbox point: center.
- Mapping profiles: landscape tracking `3840x2160` uses `[x, y] -> [x, y]`; portrait tracking `2160x3840` uses `[x, y] -> [y, 2160 - x]`.
- Semantic Warp is an unsaved display-only toggle. It adjusts WebUI endpoints and red-region filtering, not Stage2 neural inference.
- MAP rotation rotates displayed cattle points and is saved under `rotation_settings.json` `map`.
- Plan rotation rotates the floor-plan semantics and is saved under `rotation_settings.json` `plan`.
- UI rotation is toolbar-only CSS display rotation and is saved under `rotation_settings.json` `ui`.
- Forced 2D rectangles in red/green are the base structure; manual blue polygons and non-rectangle red polygons are optional reference areas.
- Resource floorplan is an optional display layer loaded from Farm 1 / Camera 1 `sna_zones.json`.
- Cattle points inside any red/green region are moved to the nearest open 2D space.
- Stage2 is precomputed by `generate_stage2_precomputed.py`; friendly links are green and unfriendly links are red.
- Stage2 links are display-debounced: a link appears after it exists for at least 0.5s within the last 1s, and disappears only after it has been absent for more than 0.5s within the last 1s.
- WebUI cattle points are causally frozen for 0.5s after a bbox disappears. The same frozen rows are retained by the current durable SNA checkpoint and contribute to its downstream metrics.
- Trails display only the most recent 15 seconds and fade out at the tail.
- Interaction links crossing any red region are filtered out before visualization.
- Figure 5 assigns `05A`, `05B`, ... dynamically from the regions present in `edge_level.csv`; its two panels share only the post-dominance edge endpoints and redistribute those cattle evenly around the circle.
- Figure 5 cattle colors remain the sample-seeded VOC mapping built from the complete global-identity set.
- The current production analysis and Figure 3 still require the six reference regions. If that upstream region contract changes later, update their validators separately; Figure 5 numbering itself is count-agnostic.
