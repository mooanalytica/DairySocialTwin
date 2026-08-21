# Data Schemas

This document describes interfaces without publishing farm data or example
rows. The exact Arrow declarations are maintained in
`src/cowtrack/schemas/`.

## Coordinate conventions

- `local_frame`, `global_frame`, `start_frame`, and `end_frame` are zero-based.
- A frame interval is inclusive at both ends unless a field explicitly states
  otherwise.
- Source bbox rows use `x, y, w, h`, with the origin at the upper-left corner.
- Normalized S00 rows use `x1, y1, x2, y2`, where `x2 = x + w` and
  `y2 = y + h` after validation and clamping.
- All bbox coordinates refer to the raw encoded source frame. Autorotation is
  disabled.
- `det_id`, `micro_id`, `stable_id`, and `global_track_id` belong to different
  namespaces and must not be interchanged.

## Private input manifest

`data/manifest.csv` has one row per source clip:

| Field | Type | Meaning |
| --- | --- | --- |
| `sequence_id` | string | Continuous recording identifier |
| `clip_order` | integer | Zero-based order within the sequence |
| `clip_id` | string | Unique clip identifier within the sequence |
| `video_path` | path | Absolute source MP4 path |
| `bbox_csv_path` | path | Absolute source bbox CSV path |
| `frame_index_base` | integer | Required source frame base; fixed to zero |
| `bbox_format` | string | Required bbox encoding; fixed to `xywh` |

The manifest is private because it contains recording identifiers and source
locations.

## Source bbox CSV

The production config maps these required logical fields:

| Logical field | Production column | Type |
| --- | --- | --- |
| video identifier | `video` | string |
| frame index | `frame` | integer |
| bbox left | `x` | number |
| bbox top | `y` | number |
| bbox width | `w` | number |
| bbox height | `h` | number |
| detection confidence | `score` | number or null |
| legacy tracker ID | `track_id` | string or null |

Legacy tracker IDs are audit-only values and are excluded from identity
construction.

## Recording time index

`gopro_time_sequence.csv` records clip order, file name, start and end
timecodes, frame count, duration, inter-clip gap, frame rate, and provenance of
the timing values. It is a private recording index rather than a public sample.

## S00 schemas

`frames.parquet` uses `FRAMES_SCHEMA` from `schemas/frames.py`:

```text
sequence_id, clip_id, clip_order, local_frame, global_frame,
pts_sec, global_time_sec, width, height
```

`detections.parquet` uses `DETECTIONS_SCHEMA` from
`schemas/detections.py`:

```text
det_id, sequence_id, clip_id, local_frame, global_frame, global_time_sec,
x1, y1, x2, y2, cx_norm, cy_norm, w_norm, h_norm, area_norm,
bbox_confidence, legacy_track_id, csv_row_index, valid, qa_flags
```

The `qa_flags` bit mask records clamping, invalid numeric values, non-positive
source size, empty or small boxes, mostly-outside boxes, high-IoU duplicates,
and invalid confidence values.

## S01 schemas

| Artifact | Schema constant | Purpose |
| --- | --- | --- |
| `det_edges.parquet` | `DET_EDGES_SCHEMA` | Candidate and accepted detection links |
| `det_to_micro.parquet` | `DET_TO_MICRO_SCHEMA` | Detection-to-micro mapping and order |
| `microtracklets.parquet` | `MICROTRACKLETS_SCHEMA` | Micro-track time, geometry, motion, purity, and status |

## S02 schemas and arrays

| Artifact | Schema constant or array contract |
| --- | --- |
| `appearance_samples.parquet` | `APPEARANCE_SAMPLES_SCHEMA` |
| `appearance_exclusions.parquet` | `APPEARANCE_EXCLUSIONS_SCHEMA` |
| `micro_appearance.parquet` | `MICRO_APPEARANCE_SCHEMA` |
| `micro_context.parquet` | `MICRO_CONTEXT_SCHEMA` |
| `sample_embeddings.f16.npy` | Row-major float16 embedding matrix |
| `micro_prototypes.f16.npy` | Float16 prototype tensor indexed by micro row |
| `micro_prototype_mask.npy` | Boolean validity mask for prototype slots |

Sample rows contain crop-quality measurements, overlap context, selection
reason, prototype inlier status, and an embedding-row reference. Embedding
arrays do not contain identity labels.

## S03 schemas

Pseudo-pair tables use `PSEUDO_PAIRS_SCHEMA`. They include pair provenance,
split and calibration role, source and target segments, clean-gallery
provenance, appearance features, gap and motion features, endpoint context,
model scores, and decision status.

The fixed appearance feature family is:

```text
prototype_cosine_max, prototype_cosine_top3_mean, medoid_cosine,
mutual_prototype_score, appearance_quality_min, appearance_quality_mean
```

Motion and context fields include temporal gap, predicted geometry, bbox size
ratios, endpoint overlap, boundary distance, and clip-boundary status.

## S04 schemas

| Artifact | Schema constant |
| --- | --- |
| short candidate edges | `SHORT_CANDIDATE_EDGES_SCHEMA` |
| short link proposals | `SHORT_LINK_PROPOSALS_SCHEMA` |
| micro-to-stable mapping | `MICRO_TO_STABLE_SCHEMA` |
| stable tracklets | `STABLE_TRACKLETS_SCHEMA` |
| detection-to-stable mapping | `DET_TO_STABLE_SCHEMA` |
| stable appearance | `STABLE_APPEARANCE_SCHEMA` |

Candidate and proposal rows retain endpoint IDs, time ranges, gallery
provenance, feature values, model evidence, conflict grouping, and review
status. Finalized mapping rows preserve predecessor evidence.

## S05 schemas

| Artifact family | Schema module |
| --- | --- |
| long calibration pairs | `schemas/s05.py` |
| long candidates and proposals | `schemas/s05_proposals.py` |
| global mappings and tracks | `schemas/s05_finalize.py` |
| forced appearance candidates and graded evidence | `schemas/s05_forced.py` |

Global-link tables distinguish calibrated probability from appearance cosine.
Forced appearance links are provisional operator-authorized evidence and do not
claim calibrated identity confidence.

## S06 schemas

| Artifact | Schema constant |
| --- | --- |
| detection export | `DETECTIONS_WITH_GLOBAL_ID_SCHEMA` |
| global-track summary | `GLOBAL_TRACK_SUMMARY_SCHEMA` |
| low-confidence link report | `LOW_CONFIDENCE_LINKS_SCHEMA` |

Invalid S00 rows remain present in the detection export, but identity and
mapping fields are null. Probability fields remain null when only appearance
cosine evidence is available.

## Artifact validation

Parquet files are validated against exact Arrow schemas. NumPy arrays are
validated for shape, dtype, row alignment, finite values, and mask agreement.
Each completed stage records config hashes and input/output fingerprints in
`_SUCCESS.json`; a mismatch is an error rather than a compatibility fallback.

