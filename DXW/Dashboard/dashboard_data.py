from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from appearance_data import AppearanceIndex, DataContractError
from cow_photos import CowPhoto, CowPhotoIndex
import webuil_figure_source as source


ROOT = Path(__file__).resolve().parent
WEBUIL_ROOT = Path("/home/hyw/DXW/WebUIL")
SAMPLE_ID = "F1_Gopro1_20250505"
SNA_INPUT_DIR = WEBUIL_ROOT / "sna_inputs"
SNA_OUTPUT_DIR = WEBUIL_ROOT / "sna_outputs"
CACHE_DIR = ROOT / ".cache"
TRAJECTORY_CACHE = CACHE_DIR / "cow_frame_visual_sample.csv.gz"
TRAJECTORY_CACHE_META = CACHE_DIR / "cow_frame_visual_sample.meta.json"
TRAJECTORY_CACHE_SCHEMA = 1
TRAJECTORY_SAMPLER_CONTRACT = "webuil-deterministic-two-pass-v1"
APPEARANCE_CACHE = CACHE_DIR / "cow_appearance_index.json"
APPEARANCE_CACHE_META = CACHE_DIR / "cow_appearance_index.meta.json"
TIME_SEQUENCE_PATH = Path("/home/hyw/time_sequence_detector_v2/output/1_1.csv")
COW_PHOTO_DIR = CACHE_DIR / "cow_photos"
COW_PHOTO_MANIFEST = CACHE_DIR / "cow_photos.meta.json"
REID_INDEX_PATH = WEBUIL_ROOT / ".cache" / "reidentification_1-1.sqlite3"
VIDEO_ROOT = Path(
    "/mnt/dairycow_sna/FULLDATA/Dairy Farm Videos/"
    "May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO"
)
REQUIRED_DASHBOARD_KEYS = (
    "identities",
    "input_generation",
    "output_generation",
    "zones",
    "summary",
    "cow_frame",
    "time_budget",
    "edge",
    "node",
    "layout",
    "community_windows",
    "community_summary",
)
EXPECTED_ZONES = {
    "cross_zone",
    "food",
    "path",
    "rest",
    "wait_for_water",
    "water",
}
FIGURE_06_EXTREME_CELL_COUNT = 7


FIGURE_COPY = {
    "01": (
        "Figure 1",
        "Floorplan trajectories",
        "Selected cattle trajectories over the annotated farm layout.",
    ),
    "02": (
        "Figure 2",
        "Visibility & zone time budget",
        "Unweighted visible seconds, split across the five observed zones.",
    ),
    "03": (
        "Figure 3",
        "Interaction volume by zone",
        "Dyad-exclusive net interaction volume at dominance ≥ 0.20.",
    ),
    "04A": (
        "Figure 4A",
        "Full-window friendly network",
        "Friendly-dominant interactions on median floorplan positions.",
    ),
    "04B": (
        "Figure 4B",
        "Full-window unfriendly network",
        "Unfriendly-dominant interactions on median floorplan positions.",
    ),
    "06": (
        "Figure 6",
        "Net adjacency heatmaps",
        "Friendly- and unfriendly-dominant induced adjacency matrices; the default "
        "selection includes the endpoints of the 7 highest and 7 lowest positive cells.",
    ),
    "07": (
        "Figure 7",
        "Cow descriptor dashboard",
        "Rate scatter and globally normalized cow-level descriptors.",
    ),
    "08": (
        "Figure 8",
        "Community stability",
        "Existing community memberships across all 92 five-minute windows.",
    ),
    "09": (
        "Figure 9",
        "Isolation decomposition",
        "Weighted components of the existing isolation score.",
    ),
    "10": (
        "Figure 10",
        "Expected vs opportunity",
        "Quality-control view of all precomputed edge rows.",
    ),
}


def _sort_cows(values: Any) -> list[str]:
    return sorted({str(value) for value in values}, key=source.cow_sort_key)


def _read_cache_meta() -> dict[str, Any] | None:
    if not TRAJECTORY_CACHE_META.is_file():
        return None
    with TRAJECTORY_CACHE_META.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DataContractError(f"Trajectory cache metadata is not an object: {TRAJECTORY_CACHE_META}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def re_fullmatch_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef"
        for character in value
    )


def validate_edge_contract(edge: pd.DataFrame, expected_cows: set[str]) -> None:
    required = {
        "cow_i",
        "cow_j",
        "zone",
        "layer",
        "expected_seconds",
        "opportunity_seconds",
    }
    missing = sorted(required - set(edge.columns))
    if missing:
        raise DataContractError(f"edge_level.csv is missing columns: {missing}")
    if len(edge) != 2152:
        raise DataContractError(f"edge_level.csv must contain 2,152 rows, found {len(edge)}")

    cow_i = edge["cow_i"].astype(str)
    cow_j = edge["cow_j"].astype(str)
    unknown = (set(cow_i) | set(cow_j)) - expected_cows
    if unknown:
        raise DataContractError(f"edge_level.csv contains unknown identities: {_sort_cows(unknown)}")
    if (cow_i == cow_j).any():
        raise DataContractError("edge_level.csv contains a self dyad")

    actual_zones = set(edge["zone"].astype(str))
    if actual_zones != EXPECTED_ZONES:
        raise DataContractError(
            "edge_level.csv zones do not match the current six-zone contract: "
            f"missing={sorted(EXPECTED_ZONES - actual_zones)}, "
            f"extra={sorted(actual_zones - EXPECTED_ZONES)}"
        )
    actual_layers = set(edge["layer"].astype(str))
    if actual_layers != {"friendly", "unfriendly"}:
        raise DataContractError(f"Unexpected edge layers: {sorted(actual_layers)}")

    numeric = edge[["expected_seconds", "opportunity_seconds"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise DataContractError("edge_level.csv contains invalid or negative weights")

    validated = edge.copy()
    canonical_pairs = [
        tuple(sorted((left, right), key=source.cow_sort_key))
        for left, right in zip(cow_i, cow_j)
    ]
    validated["_cow_a"] = [pair[0] for pair in canonical_pairs]
    validated["_cow_b"] = [pair[1] for pair in canonical_pairs]
    key_columns = ["_cow_a", "_cow_b", "zone"]
    if validated.duplicated(key_columns + ["layer"]).any():
        raise DataContractError("edge_level.csv contains a duplicate dyad-zone-layer row")
    grouped = validated.groupby(key_columns, sort=True)
    if len(grouped) != 1076:
        raise DataContractError(
            f"edge_level.csv must contain 1,076 dyad-zone keys, found {len(grouped)}"
        )
    if not bool(grouped.size().eq(2).all()):
        raise DataContractError("Every edge dyad-zone key must contain exactly two layers")
    layer_sets = grouped["layer"].agg(lambda values: frozenset(str(value) for value in values))
    if not bool(layer_sets.eq(frozenset({"friendly", "unfriendly"})).all()):
        raise DataContractError("Every edge dyad-zone key must contain friendly and unfriendly")
    ordered_pair_counts = grouped.apply(
        lambda rows: rows[["cow_i", "cow_j"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    if not bool(ordered_pair_counts.eq(1).all()):
        raise DataContractError(
            "The two layers of a dyad-zone do not use one consistent endpoint order"
        )
    opportunity = validated.assign(
        _opportunity=numeric["opportunity_seconds"].to_numpy(dtype=float)
    ).pivot(
        index=key_columns,
        columns="layer",
        values="_opportunity",
    )
    if not np.allclose(
        opportunity["friendly"].to_numpy(dtype=float),
        opportunity["unfriendly"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise DataContractError(
            "Friendly and unfriendly opportunity_seconds differ for a dyad-zone"
        )


class DashboardData:
    def __init__(self) -> None:
        source.SNA_INPUT_DIR = SNA_INPUT_DIR
        source.SNA_OUTPUT_DIR = SNA_OUTPUT_DIR
        self.paths = source.sample_paths(SAMPLE_ID)
        missing = [
            str(self.paths[key])
            for key in REQUIRED_DASHBOARD_KEYS
            if not self.paths[key].is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Required SNA result files are missing: {missing}")

        self.generation_id = source.validate_generation_pair(self.paths)
        self.identity_map = source.load_identity_display_map(self.paths["identities"])
        self.identity_document = source.read_json(self.paths["identities"])
        self._validate_identity_generation()

        self.data: dict[str, Any] = {
            "sample_id": SAMPLE_ID,
            "paths": self.paths,
            "zones": source.read_json(self.paths["zones"]),
            "summary": source.read_json(self.paths["summary"]),
            "cow_frame": pd.DataFrame(columns=["cow_id"]),
            "time_budget": source.read_csv(self.paths["time_budget"], dtype={"cow_id": str}),
            "edge": source.read_csv(
                self.paths["edge"],
                dtype={"cow_i": str, "cow_j": str},
            ),
            "node": source.read_csv(self.paths["node"], dtype={"cow_id": str}),
            "layout": source.read_csv(self.paths["layout"], dtype={"cow_id": str}),
            "community_windows": source.read_csv(
                self.paths["community_windows"],
                dtype={"cow_id": str},
            ),
            "community_summary": source.read_csv(self.paths["community_summary"]),
        }
        self._relabel_loaded_identities()
        self.all_cows = [f"G{index:04d}" for index in range(1, 63)]
        self._validate_core_tables()

        self.appearance_index = AppearanceIndex.load_or_build(
            trajectories_path=self.paths["input_generation"].with_name("trajectories.csv"),
            manifest_path=self.paths["input_generation"],
            identity_path=self.paths["identities"],
            time_sequence_path=TIME_SEQUENCE_PATH,
            cache_path=APPEARANCE_CACHE,
            cache_meta_path=APPEARANCE_CACHE_META,
            generation_id=self.generation_id,
            identity_map=self.identity_map,
        )
        self.cow_photo_index = CowPhotoIndex.load_or_build(
            appearances={
                cow: self.appearance_index.for_cow(cow)
                for cow in self.all_cows
            },
            generation_id=self.generation_id,
            identity_map=self.identity_map,
            trajectories_path=self.paths["input_generation"].with_name("trajectories.csv"),
            reid_index_path=REID_INDEX_PATH,
            video_root=VIDEO_ROOT,
            photo_dir=COW_PHOTO_DIR,
            manifest_path=COW_PHOTO_MANIFEST,
        )

        self.colors = source.cow_color_map(self.all_cows, seed=SAMPLE_ID)
        self.positions = source.layout_floorplan_positions(self.data["layout"])
        if set(self.positions) != set(self.all_cows):
            missing_positions = _sort_cows(set(self.all_cows) - set(self.positions))
            raise DataContractError(f"Layout is missing floorplan positions: {missing_positions}")
        self.communities = source.layout_communities(self.data["layout"])
        self.structures = source.structure_polygons(self.data)

        self.figure_05_zones = source.figure_05_zones(self.data["edge"])
        self.figure_zone: dict[str, str] = {
            f"05{source.alphabetic_figure_suffix(index)}": zone
            for index, zone in enumerate(self.figure_05_zones)
        }
        self.full_networks = source.net_dominance_networks(self.data["edge"])
        self.full_network_max = source.shared_network_max_weight(self.full_networks)
        self.zone_networks = {
            zone: source.net_dominance_networks(self.data["edge"], zone)
            for zone in self.figure_05_zones
        }
        self.zone_network_max = {
            zone: source.shared_network_max_weight(networks)
            for zone, networks in self.zone_networks.items()
        }
        self.zone_candidates = {
            zone: source.drawable_network_cows(networks)
            for zone, networks in self.zone_networks.items()
        }

        ordered_layout = self.data["layout"].copy()
        ordered_layout["community_id_full_friendly"] = pd.to_numeric(
            ordered_layout["community_id_full_friendly"],
            errors="coerce",
        ).fillna(-1)
        ordered_layout = ordered_layout.sort_values(
            ["community_id_full_friendly", "cow_id"],
            kind="stable",
        )
        self.matrix_cow_order = ordered_layout["cow_id"].astype(str).tolist()
        self.full_matrices = source.net_adjacency_matrices(
            self.data["edge"],
            self.matrix_cow_order,
        )
        self.full_matrix_vmax = source.adjacency_heatmap_vmax(self.full_matrices)
        (
            self.figure_06_high_cells,
            self.figure_06_low_cells,
        ) = self._figure_06_extreme_cells(FIGURE_06_EXTREME_CELL_COUNT)
        self.figure_06_default = _sort_cows(
            cow
            for _, _, cow_i, cow_j in (
                self.figure_06_high_cells + self.figure_06_low_cells
            )
            for cow in (cow_i, cow_j)
        )
        self.community_axis, self.community_matrix = source.community_plot_matrix(self.data)

        self.figure_04_default = self._maximally_dispersed_cows(20)
        self._trajectory_lock = threading.Lock()
        self._cow_frame: pd.DataFrame | None = None
        # Build or validate the deterministic plot cache before the HTTP server
        # binds. This keeps a first Figure 1 request from blocking all renders.
        self.cow_frame()

    def _validate_identity_generation(self) -> None:
        identity_generation = str(self.identity_document.get("generation_id", "")).strip()
        if identity_generation != self.generation_id:
            raise DataContractError(
                "Identity mapping generation_id does not match the input/output manifests: "
                f"identity={identity_generation!r}, manifests={self.generation_id!r}"
            )

    def _relabel_loaded_identities(self) -> None:
        columns = {
            "time_budget": ("cow_id",),
            "edge": ("cow_i", "cow_j"),
            "node": ("cow_id",),
            "layout": ("cow_id",),
            "community_windows": ("cow_id",),
        }
        for label, identity_columns in columns.items():
            frame = self.data[label]
            for column in identity_columns:
                source.relabel_identity_column(
                    frame,
                    column,
                    self.identity_map,
                    label,
                )

    def _validate_core_tables(self) -> None:
        expected = set(self.all_cows)
        for label in ("time_budget", "node", "layout"):
            frame = self.data[label]
            actual = set(frame["cow_id"].astype(str))
            if actual != expected or len(frame) != 62:
                raise DataContractError(
                    f"{label} must contain exactly one row for each of G0001..G0062"
                )

        validate_edge_contract(self.data["edge"], expected)

    def _trajectory_cache_signature(self) -> dict[str, Any]:
        stat = self.paths["cow_frame"].stat()
        return {
            "schema_version": TRAJECTORY_CACHE_SCHEMA,
            "generation_id": self.generation_id,
            "source_path": str(self.paths["cow_frame"]),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "max_points_per_cow": int(source.TRAJECTORY_MAX_POINTS_PER_COW),
            "time_gap_factor": float(source.TRAJECTORY_TIME_GAP_FACTOR),
            "sampler_contract": TRAJECTORY_SAMPLER_CONTRACT,
        }

    def _load_or_build_trajectory_cache(self) -> pd.DataFrame:
        signature = self._trajectory_cache_signature()
        cache_exists = TRAJECTORY_CACHE.is_file()
        metadata_exists = TRAJECTORY_CACHE_META.is_file()
        if cache_exists != metadata_exists:
            raise DataContractError(
                "Trajectory cache CSV and metadata must either both exist or both be absent"
            )
        metadata = _read_cache_meta()
        if metadata is not None and all(metadata.get(key) == value for key, value in signature.items()):
            expected_checksum = str(metadata.get("cache_sha256", "")).strip()
            if not re_fullmatch_sha256(expected_checksum):
                raise DataContractError("Trajectory cache metadata has an invalid SHA-256")
            actual_checksum = _file_sha256(TRAJECTORY_CACHE)
            if actual_checksum != expected_checksum:
                raise DataContractError(
                    "Trajectory cache checksum does not match its metadata"
                )
            frame = pd.read_csv(TRAJECTORY_CACHE, dtype={"cow_id": str})
            expected_row_count = metadata.get("row_count")
            if (
                not isinstance(expected_row_count, int)
                or expected_row_count < 1
                or len(frame) != expected_row_count
            ):
                raise DataContractError(
                    "Trajectory cache row count does not match its metadata: "
                    f"actual={len(frame)}, expected={expected_row_count!r}"
                )
        else:
            print(
                "Building deterministic Figure 1 visual cache from "
                f"{self.paths['cow_frame']} ...",
                flush=True,
            )
            frame = source.read_cow_frame_visual_sample(self.paths["cow_frame"])
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_tmp = CACHE_DIR / "cow_frame_visual_sample.csv.gz.tmp"
            meta_tmp = CACHE_DIR / "cow_frame_visual_sample.meta.json.tmp"
            frame.to_csv(cache_tmp, index=False, compression="gzip")
            signature["row_count"] = int(len(frame))
            signature["cache_sha256"] = _file_sha256(cache_tmp)
            with meta_tmp.open("w", encoding="utf-8") as handle:
                json.dump(signature, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(cache_tmp, TRAJECTORY_CACHE)
            os.replace(meta_tmp, TRAJECTORY_CACHE_META)
            print(
                f"Figure 1 visual cache ready: rows={len(frame):,}, path={TRAJECTORY_CACHE}",
                flush=True,
            )

        self._validate_trajectory_frame(frame)
        source.relabel_identity_column(frame, "cow_id", self.identity_map, "cow_frame")
        return frame

    def _validate_trajectory_frame(self, frame: pd.DataFrame) -> None:
        required = {
            "frame",
            "time_s",
            "dt_s",
            "cow_id",
            "anchor_x",
            "anchor_y",
            source.TRAJECTORY_SEGMENT_COLUMN,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise DataContractError(f"Trajectory cache is missing columns: {missing}")
        raw_cows = frame["cow_id"].astype(str)
        unknown = set(raw_cows) - set(self.identity_map)
        if unknown:
            raise DataContractError(
                f"Trajectory cache contains unknown raw identities: {sorted(unknown)}"
            )
        counts = raw_cows.value_counts()
        if set(counts.index) != set(self.identity_map):
            raise DataContractError("Trajectory cache does not contain all 62 cattle")
        if bool((counts > source.TRAJECTORY_MAX_POINTS_PER_COW).any()):
            raise DataContractError("Trajectory cache exceeds the per-cow point limit")

        numeric_columns = ["frame", "time_s", "dt_s", "anchor_x", "anchor_y"]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataContractError("Trajectory cache contains a non-finite numeric value")
        if bool((numeric["dt_s"] <= 0.0).any()):
            raise DataContractError("Trajectory cache contains a non-positive dt_s")
        frames = numeric["frame"].to_numpy(dtype=float)
        if not np.array_equal(frames, np.rint(frames)):
            raise DataContractError("Trajectory cache contains a non-integer frame")
        segments = pd.to_numeric(
            frame[source.TRAJECTORY_SEGMENT_COLUMN],
            errors="coerce",
        )
        if (
            segments.isna().any()
            or bool((segments < 0).any())
            or not np.array_equal(
                segments.to_numpy(dtype=float),
                np.rint(segments.to_numpy(dtype=float)),
            )
        ):
            raise DataContractError("Trajectory cache contains an invalid segment index")
    def cow_frame(self) -> pd.DataFrame:
        if self._cow_frame is None:
            with self._trajectory_lock:
                if self._cow_frame is None:
                    self._cow_frame = self._load_or_build_trajectory_cache()
                    self.data["cow_frame"] = self._cow_frame
        return self._cow_frame

    def appearances_for_cow(self, cow_id: str) -> list[dict[str, Any]]:
        if cow_id not in self.all_cows:
            raise ValueError(f"Unknown cattle identity: {cow_id}")
        return self.appearance_index.for_cow(cow_id)

    def cow_photo_for_cow(self, cow_id: str) -> CowPhoto:
        if cow_id not in self.all_cows:
            raise ValueError(f"Unknown cattle identity: {cow_id}")
        return self.cow_photo_index.for_cow(cow_id)

    def _maximally_dispersed_cows(self, count: int) -> list[str]:
        if count <= 0 or count > len(self.all_cows):
            raise ValueError(f"Invalid dispersed selection count: {count}")
        pair_candidates: list[tuple[float, str, str]] = []
        for left_index, left in enumerate(self.all_cows):
            left_x, left_y = self.positions[left]
            for right in self.all_cows[left_index + 1 :]:
                right_x, right_y = self.positions[right]
                distance_sq = (left_x - right_x) ** 2 + (left_y - right_y) ** 2
                pair_candidates.append((distance_sq, left, right))
        if not pair_candidates:
            raise DataContractError("Cannot derive a dispersed selection without position pairs")
        maximum = max(item[0] for item in pair_candidates)
        tied_pairs = sorted(
            (left, right)
            for distance_sq, left, right in pair_candidates
            if math.isclose(distance_sq, maximum, rel_tol=1e-12, abs_tol=1e-9)
        )
        selected = [tied_pairs[0][0], tied_pairs[0][1]]
        remaining = set(self.all_cows) - set(selected)
        while len(selected) < count:
            best_distance = -1.0
            best_cow: str | None = None
            for cow in sorted(remaining, key=source.cow_sort_key):
                x, y = self.positions[cow]
                distance = min(
                    (x - self.positions[chosen][0]) ** 2
                    + (y - self.positions[chosen][1]) ** 2
                    for chosen in selected
                )
                if distance > best_distance + 1e-9:
                    best_distance = distance
                    best_cow = cow
            if best_cow is None:
                raise DataContractError("Unable to complete the dispersed Figure 4 selection")
            selected.append(best_cow)
            remaining.remove(best_cow)
        return _sort_cows(selected)

    def _ranked_cows(self, frame: pd.DataFrame, column: str) -> list[str]:
        ranked = frame.loc[:, ["cow_id", column]].copy()
        ranked[column] = pd.to_numeric(ranked[column], errors="raise")
        ranked = ranked.sort_values(
            [column, "cow_id"],
            ascending=[False, True],
            kind="stable",
        )
        return ranked["cow_id"].astype(str).tolist()

    def _figure_06_extreme_cells(
        self,
        count: int,
    ) -> tuple[
        list[tuple[float, str, str, str]],
        list[tuple[float, str, str, str]],
    ]:
        if count <= 0:
            raise ValueError(f"Invalid Figure 6 extreme-cell count: {count}")

        cells: list[tuple[float, str, str, str]] = []
        for layer in ("friendly", "unfriendly"):
            matrix = self.full_matrices[layer]
            for row_index, cow_i in enumerate(self.matrix_cow_order):
                for cow_j in self.matrix_cow_order[row_index + 1 :]:
                    score = float(matrix.loc[cow_i, cow_j])
                    reverse_score = float(matrix.loc[cow_j, cow_i])
                    if (
                        not math.isfinite(score)
                        or score < 0.0
                        or not math.isclose(
                            score,
                            reverse_score,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    ):
                        raise DataContractError(
                            f"Figure 6 {layer} matrix is invalid or asymmetric for "
                            f"{cow_i}/{cow_j}"
                        )
                    if score <= 0.0:
                        continue
                    cow_a, cow_b = sorted(
                        (cow_i, cow_j),
                        key=source.cow_sort_key,
                    )
                    cells.append((score, layer, cow_a, cow_b))

        if len(cells) < count * 2:
            raise DataContractError(
                "Figure 6 does not contain enough distinct positive cells for "
                f"{count} highest and {count} lowest cells: found {len(cells)}"
            )

        layer_order = {"friendly": 0, "unfriendly": 1}

        def ascending_key(
            cell: tuple[float, str, str, str],
        ) -> tuple[float, tuple[int, int | str], tuple[int, int | str], int]:
            score, layer, cow_i, cow_j = cell
            return (
                score,
                source.cow_sort_key(cow_i),
                source.cow_sort_key(cow_j),
                layer_order[layer],
            )

        def descending_key(
            cell: tuple[float, str, str, str],
        ) -> tuple[float, tuple[int, int | str], tuple[int, int | str], int]:
            score, layer, cow_i, cow_j = cell
            return (
                -score,
                source.cow_sort_key(cow_i),
                source.cow_sort_key(cow_j),
                layer_order[layer],
            )

        highest = sorted(cells, key=descending_key)[:count]
        lowest = sorted(cells, key=ascending_key)[:count]
        return highest, lowest

    def _random_from_remaining(
        self,
        rng: Any,
        count: int,
        excluded: set[str] | None = None,
    ) -> list[str]:
        excluded = excluded or set()
        population = [cow for cow in self.all_cows if cow not in excluded]
        if count > len(population):
            raise DataContractError(
                f"Cannot select {count} cattle from a population of {len(population)}"
            )
        return list(rng.sample(population, count))

    def initial_selections(self, rng: Any | None = None) -> dict[str, list[str]]:
        rng = random.SystemRandom() if rng is None else rng
        figure_02_rank = self._ranked_cows(self.data["time_budget"], "visible_time_s")
        figure_02_fixed = set(figure_02_rank[:5] + figure_02_rank[-5:])
        figure_02 = figure_02_fixed | set(
            self._random_from_remaining(rng, 5, figure_02_fixed)
        )

        figure_09_rank = self._ranked_cows(self.data["node"], "isolation_score")
        figure_09_fixed = set(figure_09_rank[:5] + figure_09_rank[-5:])
        figure_09 = figure_09_fixed | set(
            self._random_from_remaining(rng, 5, figure_09_fixed)
        )

        selections: dict[str, list[str]] = {
            "01": self.all_cows[:5],
            "02": _sort_cows(figure_02),
            "03": [],
            "04A": list(self.figure_04_default),
            "04B": list(self.figure_04_default),
            "06": list(self.figure_06_default),
            "07": _sort_cows(self._random_from_remaining(rng, 10)),
            "08": _sort_cows(self._random_from_remaining(rng, 20)),
            "09": _sort_cows(figure_09),
            "10": [],
        }
        for figure_key, zone in self.figure_zone.items():
            candidates = list(self.zone_candidates[zone])
            selected = candidates if len(candidates) <= 20 else list(rng.sample(candidates, 20))
            selections[figure_key] = _sort_cows(selected)
        return selections

    def figure_definitions(self, rng: Any | None = None) -> list[dict[str, Any]]:
        initial = self.initial_selections(rng)
        definitions: list[dict[str, Any]] = []
        ordered_keys = ["01", "02", "03", "04A", "04B", *self.figure_zone, "06", "07", "08", "09", "10"]
        for key in ordered_keys:
            if key in self.figure_zone:
                zone = self.figure_zone[key]
                suffix = key.removeprefix("05")
                title = f"Figure 5{suffix}"
                heading = f"{zone.replace('_', ' ').title()} network"
                description = (
                    "Friendly- and unfriendly-dominant interactions for this precomputed zone."
                )
            else:
                title, heading, description = FIGURE_COPY[key]
                zone = None
            item = {
                "key": key,
                "label": title,
                "heading": heading,
                "description": description,
                "cattleMode": "none" if key in {"03", "10"} else "filter",
                "initialSelection": initial[key],
            }
            if zone is not None:
                item["zone"] = zone
            definitions.append(item)
        return definitions

    def cattle_catalog(self) -> list[dict[str, str]]:
        raw_identities = self.identity_document.get("identities")
        if not isinstance(raw_identities, list):
            raise DataContractError("Identity mapping does not contain an identities list")
        by_display = {
            str(item["display_global_id"]): str(item["global_track_uuid"])
            for item in raw_identities
        }
        if set(by_display) != set(self.all_cows):
            raise DataContractError("Identity catalog is not exactly G0001..G0062")
        return [
            {
                "id": cow,
                "label": source.figure_identity_label(cow),
                "uuid": by_display[cow],
                "color": self.colors[cow],
            }
            for cow in self.all_cows
        ]

    def bootstrap_payload(self, rng: Any | None = None) -> dict[str, Any]:
        return {
            "ok": True,
            "dataset": {
                "farm": "1",
                "camera": "1",
                "segment": "1",
                "sampleId": SAMPLE_ID,
                "generationId": self.generation_id,
            },
            "cattle": self.cattle_catalog(),
            "figures": self.figure_definitions(rng),
            "initialFigure": "01",
        }
