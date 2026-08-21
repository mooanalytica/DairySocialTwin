from __future__ import annotations

import importlib
import math
import sys
import threading
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_stage2_module(stage2_code_dir: Path):
    stage2_code_dir = Path(stage2_code_dir)
    if not stage2_code_dir.is_dir():
        raise FileNotFoundError(f"Stage2 code directory is missing: {stage2_code_dir}")
    code_dir = str(stage2_code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    return importlib.import_module("run_stage2_from_csv")


def make_stage2_args(
    stage2_code_dir: Path,
    device: str = "cuda",
    profile: str = "auto",
    inference_batch_size: int | None = None,
) -> SimpleNamespace:
    stage2_code_dir = Path(stage2_code_dir)
    return SimpleNamespace(
        interaction_gate_ckpt=str(stage2_code_dir / "models_new" / "stage2_interaction_gate_best.pt"),
        valence_ckpt=str(stage2_code_dir / "models_new" / "stage2_valence_transformer_best.pt"),
        device=device,
        profile=profile,
        inference_batch_size=inference_batch_size,
        fps=60.0,
        no_amp=False,
    )


def load_stage2_resources(
    stage2_code_dir: Path,
    device: str = "cuda",
    profile: str = "auto",
    inference_batch_size: int | None = None,
) -> SimpleNamespace:
    stage2_code_dir = Path(stage2_code_dir)
    s2 = load_stage2_module(stage2_code_dir)
    args = make_stage2_args(stage2_code_dir, device, profile, inference_batch_size)
    defaults = s2.profile_defaults(args.profile)
    if args.inference_batch_size is None:
        args.inference_batch_size = int(defaults["inference_batch_size"])
    resolved_device = s2.resolve_device(args.device)
    models = s2.load_models(args, resolved_device)
    return SimpleNamespace(
        stage2_code_dir=stage2_code_dir,
        s2=s2,
        args=args,
        device=resolved_device,
        models=models,
    )


class Stage2Runtime:
    def __init__(
        self,
        stage2_code_dir: Path,
        source_dir: Path,
        device: str = "cuda",
        profile: str = "auto",
        inference_batch_size: int | None = None,
        resources: SimpleNamespace | None = None,
    ):
        self.stage2_code_dir = Path(stage2_code_dir)
        self.source_dir = Path(source_dir)
        self.lock = threading.Lock()
        if resources is None:
            resources = load_stage2_resources(
                self.stage2_code_dir,
                device=device,
                profile=profile,
                inference_batch_size=inference_batch_size,
            )
        self.s2 = resources.s2
        self.args = resources.args
        self.device = resources.device
        self.models = resources.models

        track_csv = self.source_dir / "tracking_boxes.csv"
        kp_csv = self.source_dir / "keypoints.csv"
        manifest_path = self.source_dir / "manifest.json"
        for path in (track_csv, kp_csv, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required Stage2 input is missing: {path}")

        manifest = self.s2.normalize_legacy_paths(self.s2.load_json_if_present(manifest_path))
        video_manifest = dict(manifest.get("video_manifest", {})) if manifest else {}
        self.fps = float(video_manifest.get("fps", 0.0) or self.args.fps)
        if self.fps <= 0:
            self.fps = float(self.args.fps)
        self.video_name = str(video_manifest.get("relative_path", "")).replace("/", "\\").split("\\")[-1]

        self.boxes_by_frame, self.track_rows, track_video_name = self.s2.load_boxes(track_csv)
        num_kpts = len(self.models.required_keypoints)
        self.kpts_by_key, self.kpt_rows = self.s2.load_keypoints(kp_csv, num_kpts)
        if track_video_name:
            self.video_name = track_video_name
        if not self.video_name:
            self.video_name = self.source_dir.name

        all_frames = sorted(self.boxes_by_frame)
        if not all_frames:
            raise RuntimeError("no tracking frames available for Stage2")
        self.track_last_frame: dict[int, int] = {}
        for frame_idx in all_frames:
            for track_id in self.boxes_by_frame[frame_idx]:
                self.track_last_frame[int(track_id)] = int(frame_idx)
        self.frame_min = min(all_frames)
        max_frame = max(all_frames)
        manifest_frames = int(video_manifest.get("frame_count", 0) or 0)
        self.frame_total = max(max_frame + 1, manifest_frames)

        self.num_kpts = num_kpts
        self.buf_sec = max(self.s2.INTERACT_MIN_SEC, self.s2.BUF_SEC_MIN)
        self.buf_len = int(math.ceil(self.buf_sec * self.fps))
        self.min_frames = int(math.ceil(self.s2.INTERACT_MIN_SEC * self.fps))
        self.gap_tol_fr = int(round(self.s2.GAP_TOL_SEC * self.fps))
        self.classify_every = max(1, int(round(self.fps / self.s2.CLASSIFY_EVERY_HZ)))
        self.cooldown_frames = int(round(self.s2.COOLDOWN_SEC * self.fps))
        self.line_window_frames = max(1, int(round(self.fps * 1.0)))
        self.line_threshold_frames = max(1, int(math.ceil(self.fps * 0.5)))
        self.reset()

    def _make_pair_geom(self) -> dict[str, Any]:
        return {"suspect_frames": 0, "miss": 0, "prox": None, "last_ok_frame": -1}

    def _make_pair_prox_hist(self):
        return {
            w: deque(maxlen=max(1, int(w * self.fps)))
            for w in self.s2.Q_WINDOWS_SEC
        }

    def _make_pair_buf(self):
        return self.s2._make_pair_buf(self.buf_len)

    def reset(self) -> None:
        with self.lock:
            self.current_frame = int(self.frame_min) - 1
            self.pair_geom = defaultdict(self._make_pair_geom)
            self.pair_prox_hist = defaultdict(self._make_pair_prox_hist)
            self.pair_buf = defaultdict(self._make_pair_buf)
            self.pair_evt = defaultdict(self.s2._make_pair_evt)
            self.live_pair_geom_keys: set[tuple[int, int]] = set()
            self.expired_pair_geom_keys: set[tuple[int, int]] = set()
            self.recent_event_keys: set[tuple[int, int]] = set()
            self.event_expiry_frame: dict[tuple[int, int], int] = {}
            self.event_expiry_keys: dict[int, set[tuple[int, int]]] = defaultdict(set)
            self.pair_terminal_expiry_frame: dict[tuple[int, int], int] = {}
            self.pair_terminal_expiry_keys: dict[int, set[tuple[int, int]]] = defaultdict(set)
            self.pair_cooldown_until: dict[tuple[int, int], int] = {}
            self.line_history: dict[tuple[int, int, str], deque] = {}
            self.line_visible: set[tuple[int, int, str]] = set()
            self.line_last_item: dict[tuple[int, int, str], dict[str, Any]] = {}
            self.raw_frame_interactions: dict[int, list[dict[str, Any]]] = {}
            self.raw_frame_geometry_candidates: dict[int, set[tuple[int, int]]] = {}

    def meta(self) -> dict[str, Any]:
        return {
            "video": self.video_name,
            "fps": self.fps,
            "frameMin": int(self.frame_min),
            "frameTotal": self.frame_total,
            "currentFrame": self.current_frame,
            "trackRows": self.track_rows,
            "keypointRows": self.kpt_rows,
            "requiredKeypoints": len(self.models.required_keypoints),
            "interactionGateThreshold": float(self.models.interaction_gate_thresh),
            "inferenceBatchSize": int(self.args.inference_batch_size),
            "device": str(self.device),
            "lineDebounceWindowFrames": int(self.line_window_frames),
            "lineDebounceThresholdFrames": int(self.line_threshold_frames),
            "lineDebounceRule": "show after >=0.5s present in the last 1s; hide after >0.5s absent in the last 1s",
        }

    def interactions_for_frame(self, frame: int) -> list[dict[str, Any]]:
        target = max(int(self.frame_min), min(int(frame), int(self.frame_total - 1)))
        with self.lock:
            if target < self.current_frame:
                self._reset_unlocked()
            while self.current_frame < target:
                self._step_unlocked(self.current_frame + 1)
            return self._display_interactions_unlocked()

    def raw_interactions_for_frame(self, frame: int) -> list[dict[str, Any]]:
        target = max(int(self.frame_min), min(int(frame), int(self.frame_total - 1)))
        with self.lock:
            if target < self.current_frame:
                self._reset_unlocked()
            while self.current_frame < target:
                self._step_unlocked(self.current_frame + 1)
            return [dict(item) for item in self.raw_frame_interactions.get(target, [])]

    def geometry_candidate_pairs_for_frame(self, frame: int) -> set[tuple[int, int]]:
        target = max(int(self.frame_min), min(int(frame), int(self.frame_total - 1)))
        with self.lock:
            if target < self.current_frame:
                self._reset_unlocked()
            while self.current_frame < target:
                self._step_unlocked(self.current_frame + 1)
            return set(self.raw_frame_geometry_candidates.get(target, set()))

    def _reset_unlocked(self) -> None:
        self.current_frame = int(self.frame_min) - 1
        self.pair_geom = defaultdict(self._make_pair_geom)
        self.pair_prox_hist = defaultdict(self._make_pair_prox_hist)
        self.pair_buf = defaultdict(self._make_pair_buf)
        self.pair_evt = defaultdict(self.s2._make_pair_evt)
        self.live_pair_geom_keys = set()
        self.expired_pair_geom_keys = set()
        self.recent_event_keys = set()
        self.event_expiry_frame = {}
        self.event_expiry_keys = defaultdict(set)
        self.pair_terminal_expiry_frame = {}
        self.pair_terminal_expiry_keys = defaultdict(set)
        self.pair_cooldown_until = {}
        self.line_history = {}
        self.line_visible = set()
        self.line_last_item = {}
        self.raw_frame_interactions = {}
        self.raw_frame_geometry_candidates = {}

    def _step_unlocked(self, frame_idx: int) -> None:
        s2 = self.s2
        np = s2.np
        self._expire_terminal_pair_states_unlocked(frame_idx)

        frame_boxes = self.boxes_by_frame.get(frame_idx, {})
        tracks = [frame_boxes[tid] for tid in sorted(frame_boxes)]
        all_pairs: list[tuple[float, tuple[int, int]]] = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a = tracks[i]
                b = tracks[j]
                overlap, _iou = s2.boxes_overlap(a, b)
                center_dist_norm, d1, d2 = s2.normalized_center_distance(a, b)
                sim = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
                diag_ok = sim >= s2.INTERACT_DIAG_SIM_RATIO
                key = tuple(sorted((int(a.tid), int(b.tid))))
                self._index_terminal_pair_expiry_unlocked(key)

                prox_hist = self.pair_prox_hist[key]
                evt = self.pair_evt[key]
                hard_prox_ok = center_dist_norm <= s2.PROX_HARD_CAP
                quantile_ok = False

                if overlap and diag_ok:
                    for w in s2.Q_WINDOWS_SEC:
                        hist = prox_hist[w]
                        hist.append(center_dist_norm)
                        if len(hist) >= max(3, int(0.5 * w * self.fps)):
                            qv = float(np.quantile(np.asarray(hist, dtype=np.float32), s2.Q_PROX))
                            if qv <= s2.PROX_HARD_CAP:
                                quantile_ok = True
                else:
                    for hist in prox_hist.values():
                        hist.clear()

                gate_pass = bool(overlap and diag_ok and (hard_prox_ok or quantile_ok))
                evt["gate_hist"].append(1 if gate_pass else 0)
                stable_gate = sum(evt["gate_hist"]) >= s2.GATE_STABLE_K
                self._resolve_expired_geometry_reappearance_unlocked(key, stable_gate)
                if stable_gate:
                    g = self.pair_geom[key]
                    g["suspect_frames"] += 1
                    g["miss"] = 0
                    g["last_ok_frame"] = frame_idx
                    g["prox"] = float(center_dist_norm)
                    all_pairs.append((float(center_dist_norm), key))

        all_pairs.sort(key=lambda x: x[0])
        geometry_candidate_pairs = {p[1] for p in all_pairs}
        suspect_pairs = {p[1] for p in all_pairs[: s2.MAX_SUSPECT_PAIRS]}
        self.raw_frame_geometry_candidates[frame_idx] = set(geometry_candidate_pairs)

        self.live_pair_geom_keys.update(geometry_candidate_pairs)
        self._age_pair_geometry_unlocked(suspect_pairs)

        for a_id, b_id in suspect_pairs:
            box_a = frame_boxes.get(int(a_id))
            box_b = frame_boxes.get(int(b_id))
            k_a = self.kpts_by_key.get((frame_idx, int(a_id)))
            k_b = self.kpts_by_key.get((frame_idx, int(b_id)))
            st = self.pair_buf[(a_id, b_id)]
            st["frames"].append(frame_idx)
            st["kptsA"].append(k_a)
            st["kptsB"].append(k_b)
            st["centerA"].append(box_a.center if box_a is not None else None)
            st["centerB"].append(box_b.center if box_b is not None else None)
            st["scaleA"].append(box_a.area_scale if box_a is not None else float("nan"))
            st["scaleB"].append(box_b.area_scale if box_b is not None else float("nan"))

        if (frame_idx % self.classify_every) == 0:
            self._classify_unlocked(frame_idx, suspect_pairs)
        else:
            for key in suspect_pairs:
                evt = self.pair_evt[key]
                if evt["active"] and (evt["last_pos_frame"] >= 0) and (frame_idx - evt["last_pos_frame"] > self.gap_tol_fr):
                    end_f = int(evt["last_pos_frame"])
                    self._finish_event_unlocked(key, evt)
                    self.pair_cooldown_until[key] = end_f + self.cooldown_frames

        self._expire_recent_events_unlocked(frame_idx)
        raw_interactions = self._raw_active_interactions_unlocked(frame_idx)
        self.raw_frame_interactions[frame_idx] = [dict(item) for item in raw_interactions]
        self._update_line_display_unlocked(raw_interactions)
        self.current_frame = frame_idx

    def _classify_unlocked(self, frame_idx: int, suspect_pairs: set[tuple[int, int]]) -> None:
        s2 = self.s2
        eligible_keys: list[tuple[int, int]] = []
        interaction_gate_pti_map: dict[tuple[int, int], tuple[Any, Any]] = {}
        valence_xfull_map: dict[tuple[int, int], Any] = {}

        for key in suspect_pairs:
            if key in self.pair_cooldown_until and frame_idx < self.pair_cooldown_until[key]:
                continue
            stbuf = self.pair_buf[key]
            if len(stbuf["frames"]) < int(min(s2.Q_WINDOWS_SEC) * self.fps):
                continue
            valence_xfull = s2.valence_seq_from_pairbuf(stbuf, self.num_kpts)
            interaction_gate_pti = s2.pti_seq_from_pairbuf(stbuf, self.num_kpts)
            if interaction_gate_pti is None or valence_xfull is None or valence_xfull.shape[0] < 2:
                continue
            if interaction_gate_pti[0].shape[0] < 2:
                continue
            eligible_keys.append(key)
            interaction_gate_pti_map[key] = interaction_gate_pti
            valence_xfull_map[key] = valence_xfull

        if not eligible_keys:
            self._close_stale_events_unlocked(frame_idx, suspect_pairs)
            return

        p1_max = {k: 0.0 for k in eligible_keys}
        for w in s2.MI_CROPS_SEC:
            keys_w: list[tuple[int, int]] = []
            list_pti: list[tuple[Any, Any]] = []
            for key in eligible_keys:
                kfull, mfull = interaction_gate_pti_map[key]
                kseq, mseq = s2.crop_pti_sequence(kfull, mfull, self.fps, w, self.models.interaction_gate_max_frames)
                list_pti.append((kseq, mseq))
                keys_w.append(key)
                if s2.BIDIRECTIONAL_PAIR_INFERENCE:
                    list_pti.append(s2.swap_pti_pair_sequence(kseq, mseq))
                    keys_w.append(key)
            probs = s2.pti_gate_batch_probs(
                self.models.interaction_gate_model,
                list_pti,
                device=self.device,
                batch_size=self.args.inference_batch_size,
                amp=not self.args.no_amp,
            )
            for key, prob in zip(keys_w, probs):
                if prob > p1_max[key]:
                    p1_max[key] = float(prob)

        stable_interaction_gate_keys: list[tuple[int, int]] = []
        for key in eligible_keys:
            evt = self.pair_evt[key]
            p1 = float(p1_max[key])
            hit1 = p1 >= self.models.interaction_gate_thresh
            evt["conf_stage1_max"] = max(evt.get("conf_stage1_max", 0.0), p1)
            evt["stage1_hist"].append(1 if hit1 else 0)
            stable_interaction_gate = sum(evt["stage1_hist"]) >= s2.INTERACTION_GATE_STABLE_K
            if stable_interaction_gate:
                evt["last_pos_frame"] = frame_idx
                evt["last_pred_frame"] = frame_idx
                self._index_recent_event_unlocked(key, evt)
                stable_interaction_gate_keys.append(key)

        if stable_interaction_gate_keys:
            self._classify_valence_unlocked(frame_idx, stable_interaction_gate_keys, valence_xfull_map, p1_max)

        self._close_stale_events_unlocked(frame_idx, eligible_keys)

    def _classify_valence_unlocked(
        self,
        frame_idx: int,
        stable_keys: list[tuple[int, int]],
        valence_xfull_map: dict[tuple[int, int], Any],
        p1_max: dict[tuple[int, int], float],
    ) -> None:
        s2 = self.s2
        logits_sum: dict[tuple[int, int], Any | None] = {k: None for k in stable_keys}
        for w in s2.MI_CROPS_SEC:
            list_x = []
            keys_w: list[tuple[tuple[int, int], bool]] = []
            for key in stable_keys:
                xfull = valence_xfull_map[key]
                if w == "full":
                    x = xfull
                else:
                    n = int(float(w) * self.fps)
                    x = xfull[-n:] if xfull.shape[0] > n else xfull
                list_x.append(s2.prepare_seq_for_model(x, self.fps, self.models.valence_train_fps, self.models.valence_max_frames))
                keys_w.append((key, False))
                if s2.BIDIRECTIONAL_PAIR_INFERENCE:
                    list_x.append(
                        s2.prepare_seq_for_model(
                            s2.swap_valence_pair_sequence(x),
                            self.fps,
                            self.models.valence_train_fps,
                            self.models.valence_max_frames,
                        )
                    )
                    keys_w.append((key, True))

            logits_batch = s2.valence_batch_logits(
                self.models.valence_model,
                list_x,
                device=self.device,
                batch_size=self.args.inference_batch_size,
                amp=not self.args.no_amp,
            )

            best_logits: dict[tuple[int, int], Any] = {}
            best_conf: dict[tuple[int, int], float] = {}
            for (key, _swapped), logits in zip(keys_w, logits_batch):
                _, conf, _ = s2.decode_valence_logits(logits)
                if (key not in best_conf) or (conf > best_conf[key]):
                    best_conf[key] = conf
                    best_logits[key] = logits.astype(s2.np.float32)

            for key, logits in best_logits.items():
                logits_sum[key] = logits if logits_sum[key] is None else logits_sum[key] + logits

        num_windows = float(len(s2.MI_CROPS_SEC))
        for key in stable_keys:
            evt = self.pair_evt[key]
            p1_now = float(p1_max.get(key, 0.0))
            if logits_sum[key] is None:
                continue
            avg_logits = logits_sum[key] / max(1.0, num_windows)
            cid, cconf, probs = s2.decode_valence_logits(avg_logits)
            stage2_label = self.models.valence_id_to_label.get(cid, str(cid))
            valence_label, valence_conf, friendly_score, unfriendly_score = s2.decode_valence_scores(
                probs,
                self.models.valence_id_to_label,
            )
            interaction_score = float(friendly_score + unfriendly_score)
            valence_conf_thresh = s2.dynamic_valence_conf_threshold(
                p1_now,
                self.models.dynamic_valence_threshold,
                s2.VALENCE_MIN_CONF,
            )
            evt["last_pred_frame"] = frame_idx

            if (
                stage2_label.lower() == s2.NO_INTERACTION_LABEL
                or interaction_score < float(self.models.valence_interaction_thresh)
                or valence_conf < valence_conf_thresh
            ):
                if not evt["active"] and evt["stage2_votes"]:
                    evt["stage2_votes"].clear()
                continue

            evt["stage2_votes"].append(valence_label)
            voted_label = s2.vote_mode(evt["stage2_votes"]) or valence_label

            if (not evt["active"]) and (len(evt["stage2_votes"]) < s2.VALENCE_MIN_START_VOTES):
                self._update_evt_conf(evt, cconf, valence_conf, valence_conf_thresh, friendly_score, unfriendly_score, stage2_label)
                continue

            if evt["active"] and evt["cur_label"] != voted_label:
                self._finish_event_unlocked(key, evt)
                evt["conf_stage1_max"] = p1_now
                evt["stage2_votes"].append(valence_label)
                voted_label = valence_label

            if not evt["active"]:
                evt["active"] = True
                evt["cur_label"] = voted_label
                evt["start_f"] = frame_idx

            self._update_evt_conf(evt, cconf, valence_conf, valence_conf_thresh, friendly_score, unfriendly_score, stage2_label)
            evt["last_pos_frame"] = frame_idx
            evt["last_pred_frame"] = frame_idx
            self._index_recent_event_unlocked(key, evt)

    def _update_evt_conf(
        self,
        evt: dict,
        cconf: float,
        valence_conf: float,
        valence_thresh: float,
        friendly_score: float,
        unfriendly_score: float,
        stage2_label: str,
    ) -> None:
        if cconf >= evt.get("conf_stage2_max", 0.0):
            evt["cur_stage2_label"] = stage2_label
        evt["conf_stage2_max"] = max(evt.get("conf_stage2_max", 0.0), float(cconf))
        evt["conf_valence_max"] = max(evt.get("conf_valence_max", 0.0), float(valence_conf))
        evt["valence_thresh_min"] = min(evt.get("valence_thresh_min", 1.0), float(valence_thresh))
        evt["conf_friendly_max"] = max(evt.get("conf_friendly_max", 0.0), float(friendly_score))
        evt["conf_unfriendly_max"] = max(evt.get("conf_unfriendly_max", 0.0), float(unfriendly_score))

    def _age_pair_geometry_unlocked(self, suspect_pairs: set[tuple[int, int]]) -> None:
        """Age only geometry states that have not already crossed the gap."""
        for key in self.live_pair_geom_keys.difference(suspect_pairs):
            st = self.pair_geom.get(key)
            if st is None:
                self.live_pair_geom_keys.discard(key)
                self.pair_prox_hist.pop(key, None)
                self.pair_buf.pop(key, None)
                continue
            st["miss"] += 1
            if st["miss"] > self.gap_tol_fr:
                self.pair_geom.pop(key, None)
                self.pair_prox_hist.pop(key, None)
                self.pair_buf.pop(key, None)
                self.live_pair_geom_keys.discard(key)
                self.expired_pair_geom_keys.add(key)

    def _resolve_expired_geometry_reappearance_unlocked(
        self,
        key: tuple[int, int],
        stable_gate: bool,
    ) -> None:
        """Preserve legacy proximity resets while an expired pair restabilizes."""
        if stable_gate:
            self.expired_pair_geom_keys.discard(key)
        elif key in self.expired_pair_geom_keys:
            self.pair_prox_hist.pop(key, None)

    def _index_recent_event_unlocked(self, key: tuple[int, int], evt: dict) -> None:
        """Index an active event until the first frame outside its gap tolerance."""
        if not evt.get("active", False):
            return
        last_pos_frame = int(evt.get("last_pos_frame", -1))
        if last_pos_frame < 0:
            return
        expiry_frame = last_pos_frame + self.gap_tol_fr + 1
        self.recent_event_keys.add(key)
        self.event_expiry_frame[key] = expiry_frame
        self.event_expiry_keys[expiry_frame].add(key)

    def _index_terminal_pair_expiry_unlocked(self, key: tuple[int, int]) -> None:
        """Schedule full cleanup once either track can no longer reappear."""
        if key in self.pair_terminal_expiry_frame:
            return
        last_a = self.track_last_frame[int(key[0])]
        last_b = self.track_last_frame[int(key[1])]
        expiry_frame = min(last_a, last_b) + self.gap_tol_fr + 1
        self.pair_terminal_expiry_frame[key] = expiry_frame
        self.pair_terminal_expiry_keys[expiry_frame].add(key)

    def _expire_terminal_pair_states_unlocked(self, frame_idx: int) -> None:
        """Discard pair history only when the pair is provably unable to reappear."""
        for key in self.pair_terminal_expiry_keys.pop(frame_idx, set()):
            if self.pair_terminal_expiry_frame.get(key) != frame_idx:
                continue
            self.pair_terminal_expiry_frame.pop(key, None)
            self.live_pair_geom_keys.discard(key)
            self.expired_pair_geom_keys.discard(key)
            self.recent_event_keys.discard(key)
            self.event_expiry_frame.pop(key, None)
            self.pair_geom.pop(key, None)
            self.pair_prox_hist.pop(key, None)
            self.pair_buf.pop(key, None)
            self.pair_evt.pop(key, None)
            self.pair_cooldown_until.pop(key, None)

    def _expire_recent_events_unlocked(self, frame_idx: int) -> None:
        """Remove stale events from the output index without discarding event history."""
        for key in self.event_expiry_keys.pop(frame_idx, set()):
            if self.event_expiry_frame.get(key) != frame_idx:
                continue
            evt = self.pair_evt.get(key)
            last_pos_frame = int(evt.get("last_pos_frame", -1)) if evt is not None else -1
            if evt is None or not evt.get("active", False) or frame_idx - last_pos_frame > self.gap_tol_fr:
                self.recent_event_keys.discard(key)
                self.event_expiry_frame.pop(key, None)

    def _finish_event_unlocked(self, key: tuple[int, int], evt: dict) -> None:
        self.recent_event_keys.discard(key)
        self.event_expiry_frame.pop(key, None)
        self.s2.reset_stage2_event(evt)

    def _close_stale_events_unlocked(self, frame_idx: int, keys) -> None:
        for key in keys:
            evt = self.pair_evt[key]
            if evt["active"] and (evt["last_pos_frame"] >= 0) and (frame_idx - evt["last_pos_frame"] > self.gap_tol_fr):
                end_f = int(evt["last_pos_frame"])
                self._finish_event_unlocked(key, evt)
                self.pair_cooldown_until[key] = end_f + self.cooldown_frames

    def _line_key(self, item: dict[str, Any]) -> tuple[int, int, str]:
        tid_a = int(item["tidA"])
        tid_b = int(item["tidB"])
        label = str(item.get("class") or "")
        return min(tid_a, tid_b), max(tid_a, tid_b), label

    def _update_line_display_unlocked(self, raw_interactions: list[dict[str, Any]]) -> None:
        present: dict[tuple[int, int, str], dict[str, Any]] = {}
        for item in raw_interactions:
            key = self._line_key(item)
            present[key] = item
            self.line_last_item[key] = item

        keys = set(self.line_history) | set(self.line_visible) | set(present)
        for key in keys:
            history = self.line_history.get(key)
            if history is None:
                history = deque(maxlen=self.line_window_frames)
                self.line_history[key] = history

            history.append(key in present)
            present_frames = sum(1 for value in history if value)
            absent_frames = len(history) - present_frames

            if key not in self.line_visible:
                if present_frames >= self.line_threshold_frames:
                    self.line_visible.add(key)
            elif absent_frames > self.line_threshold_frames:
                self.line_visible.remove(key)

            if key not in self.line_visible and key not in present and present_frames == 0:
                self.line_history.pop(key, None)
                self.line_last_item.pop(key, None)

    def _display_interactions_unlocked(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in sorted(self.line_visible, key=lambda item: (item[2], item[0], item[1])):
            item = self.line_last_item.get(key)
            if item is None:
                continue
            copied = dict(item)
            history = self.line_history.get(key, [])
            present_frames = sum(1 for value in history if value)
            copied["displayWindowPresentFrames"] = int(present_frames)
            copied["displayWindowMissingFrames"] = int(len(history) - present_frames)
            copied["displayWindowFrames"] = int(self.line_window_frames)
            copied["displayThresholdFrames"] = int(self.line_threshold_frames)
            out.append(copied)
        return out

    def _raw_active_interactions_unlocked(self, frame: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in self.recent_event_keys:
            evt = self.pair_evt[key]
            if not evt.get("active", False):
                continue
            start_f = evt.get("start_f")
            if start_f is None or frame < int(start_f):
                continue
            last_pos = int(evt.get("last_pos_frame", -1))
            if last_pos >= 0 and frame - last_pos > self.gap_tol_fr:
                continue
            label = str(evt.get("cur_label") or "")
            if label not in {"friendly", "unfriendly"}:
                continue
            out.append(
                {
                    "video": self.video_name,
                    "class": label,
                    "tidA": int(key[0]),
                    "tidB": int(key[1]),
                    "startFrame": int(start_f),
                    "lastPositiveFrame": int(last_pos),
                    "stage2Class": str(evt.get("cur_stage2_label") or ""),
                    "stage1Prob": round(float(evt.get("conf_stage1_max", 0.0)), 6),
                    "stage2Conf": round(float(evt.get("conf_stage2_max", 0.0)), 6),
                    "valenceScore": round(float(evt.get("conf_valence_max", 0.0)), 6),
                    "friendlyScore": round(float(evt.get("conf_friendly_max", 0.0)), 6),
                    "unfriendlyScore": round(float(evt.get("conf_unfriendly_max", 0.0)), 6),
                }
            )
        out.sort(key=lambda item: (item["class"], item["tidA"], item["tidB"]))
        return out
