"""
tracker.py

Maintains persistent per-person identity and rolling history buffers.

Two layers of tracking are supported:
  1. Native Ultralytics tracking (BoT-SORT / ByteTrack), used automatically
     when `PoseDetector.infer(..., use_tracking=True)` succeeds and returns
     track IDs.
  2. A fallback greedy IoU + centroid tracker (SimpleIoUTracker), used when
     native tracking is unavailable (e.g. missing `lap` dependency) or when
     a detection has no track_id for some other reason.

Either way, downstream code only ever sees a stable integer track_id and a
PersonTrack object with history it can reason about temporally.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.pose_detector import Detection, NUM_KEYPOINTS
from src.activity_classifier import compute_torso_angle


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleIoUTracker:

    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._next_id = 1
        self._last_bbox: Dict[int, np.ndarray] = {}
        self._missing: Dict[int, int] = {}

    def assign(self, detections: List[Detection]) -> None:
        unmatched_dets = [d for d in detections if d.track_id is None]
        if not unmatched_dets:
            return

        available_ids = list(self._last_bbox.keys())
        used_ids = set()

        # Greedy matching by descending IoU.
        pairs: List[Tuple[float, int, Detection]] = []
        for tid in available_ids:
            for det in unmatched_dets:
                iou = _bbox_iou(self._last_bbox[tid], det.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, det))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_dets = set()
        for iou, tid, det in pairs:
            if tid in used_ids or id(det) in matched_dets:
                continue
            det.track_id = tid
            used_ids.add(tid)
            matched_dets.add(id(det))

        for det in unmatched_dets:
            if id(det) in matched_dets:
                continue
            det.track_id = self._next_id
            self._next_id += 1

        seen_ids = {d.track_id for d in detections}
        for det in detections:
            self._last_bbox[det.track_id] = det.bbox
            self._missing[det.track_id] = 0
        for tid in list(self._missing.keys()):
            if tid not in seen_ids:
                self._missing[tid] += 1
                if self._missing[tid] > self.max_missing_frames:
                    self._last_bbox.pop(tid, None)
                    self._missing.pop(tid, None)


@dataclass
class PersonTrack:

    track_id: int
    history_len: int = 30

    bbox_history: Deque[np.ndarray] = field(default_factory=deque)
    keypoints_history: Deque[np.ndarray] = field(default_factory=deque)
    center_history: Deque[Tuple[float, float]] = field(default_factory=deque)
    angle_history: Deque[Optional[float]] = field(default_factory=deque)
    timestamp_history: Deque[float] = field(default_factory=deque)

    current_state: str = "UNKNOWN"
    state_confidence: float = 0.0
    consecutive_state_frames: int = 0
    fall_score: float = 0.0
    fall_state: str = "NORMAL"
    last_transition: Optional[str] = None
    fall_event_emitted: bool = False

    frames_since_seen: int = 0
    first_seen_frame: int = -1
    last_seen_frame: int = -1

    def __post_init__(self):
        maxlen = self.history_len
        self.bbox_history = deque(maxlen=maxlen)
        self.keypoints_history = deque(maxlen=maxlen)
        self.center_history = deque(maxlen=maxlen)
        self.angle_history = deque(maxlen=maxlen)
        self.timestamp_history = deque(maxlen=maxlen)

    def update(self, det: Detection, timestamp: float, frame_idx: int, torso_angle: Optional[float]):
        self.bbox_history.append(det.bbox)
        self.keypoints_history.append(det.keypoints)
        cx = float((det.bbox[0] + det.bbox[2]) / 2.0)
        cy = float((det.bbox[1] + det.bbox[3]) / 2.0)
        self.center_history.append((cx, cy))
        self.angle_history.append(torso_angle)
        self.timestamp_history.append(timestamp)
        self.frames_since_seen = 0
        if self.first_seen_frame < 0:
            self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx

    @property
    def latest_bbox(self) -> Optional[np.ndarray]:
        return self.bbox_history[-1] if self.bbox_history else None

    @property
    def latest_keypoints(self) -> Optional[np.ndarray]:
        return self.keypoints_history[-1] if self.keypoints_history else None

    def bbox_diagonal(self) -> float:
        if not self.bbox_history:
            return 1.0
        x1, y1, x2, y2 = self.latest_bbox
        return max(1.0, float(np.hypot(x2 - x1, y2 - y1)))


class TrackManager:

    def __init__(
        self,
        history_len: int = 30,
        max_missing_frames: int = 30,
        iou_threshold: float = 0.3,
        fps: float = 30.0,
        relink_max_gap_seconds: float = 3.0,
        relink_max_distance_ratio: float = 2.0,
    ):
        self.history_len = history_len
        self.fps = max(1e-3, fps)
        self.relink_max_gap_frames = max(1, round(relink_max_gap_seconds * self.fps))
        self.relink_max_distance_ratio = relink_max_distance_ratio
        self.max_missing_frames = max(max_missing_frames, self.relink_max_gap_frames)
        self.tracks: Dict[int, PersonTrack] = {}
        self._fallback_tracker = SimpleIoUTracker(
            iou_threshold=iou_threshold, max_missing_frames=self.max_missing_frames
        )
        self._raw_to_stable: Dict[object, int] = {}
        self._next_stable_id = 1

    def _resolve_stable_ids(self, detections: List[Detection]) -> None:
        claimed_this_frame: set = set()
        needs_resolution: List[Detection] = []

        for det in detections:
            if det.track_id is not None and det.track_id in self._raw_to_stable:
                stable_id = self._raw_to_stable[det.track_id]
                det.track_id = stable_id
                claimed_this_frame.add(stable_id)
            else:
                needs_resolution.append(det)

        if needs_resolution:
            candidates = [
                sid
                for sid, track in self.tracks.items()
                if sid not in claimed_this_frame
                and 0 < track.frames_since_seen <= self.relink_max_gap_frames
                and track.latest_bbox is not None
            ]
            pairs: List[Tuple[float, Detection, int]] = []
            for det in needs_resolution:
                raw_id = det.track_id
                det_cx = float((det.bbox[0] + det.bbox[2]) / 2.0)
                det_cy = float((det.bbox[1] + det.bbox[3]) / 2.0)
                for sid in candidates:
                    track = self.tracks[sid]
                    lx1, ly1, lx2, ly2 = track.latest_bbox
                    lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
                    dist = float(np.hypot(det_cx - lcx, det_cy - lcy)) / track.bbox_diagonal()
                    if dist <= self.relink_max_distance_ratio:
                        pairs.append((dist, det, sid))
            pairs.sort(key=lambda p: p[0])

            matched_dets = set()
            for dist, det, sid in pairs:
                if id(det) in matched_dets or sid in claimed_this_frame:
                    continue
                raw_id = det.track_id
                det.track_id = sid
                if raw_id is not None:
                    self._raw_to_stable[raw_id] = sid
                claimed_this_frame.add(sid)
                matched_dets.add(id(det))

            for det in needs_resolution:
                if id(det) in matched_dets:
                    continue
                raw_id = det.track_id
                stable_id = self._next_stable_id
                self._next_stable_id += 1
                det.track_id = stable_id
                if raw_id is not None:
                    self._raw_to_stable[raw_id] = stable_id
                claimed_this_frame.add(stable_id)

    @staticmethod
    def _suppress_duplicate_detections(detections: List[Detection], iou_threshold: float = 0.6) -> List[Detection]:
        if len(detections) < 2:
            return detections
        order = sorted(range(len(detections)), key=lambda i: detections[i].bbox_conf, reverse=True)
        kept_indices: List[int] = []
        for i in order:
            det = detections[i]
            if any(_bbox_iou(det.bbox, detections[j].bbox) >= iou_threshold for j in kept_indices):
                continue
            kept_indices.append(i)
        kept_set = set(kept_indices)
        return [d for idx, d in enumerate(detections) if idx in kept_set]

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        timestamp: float,
        native_tracking_available: bool,
        torso_angles: Optional[List[Optional[float]]] = None,
    ) -> Dict[int, PersonTrack]:
        original_detections = detections
        detections = self._suppress_duplicate_detections(detections)
        if torso_angles is not None and len(detections) != len(original_detections):
            kept_ids = {id(d) for d in detections}
            torso_angles = [a for d, a in zip(original_detections, torso_angles) if id(d) in kept_ids]

        if not native_tracking_available or any(d.track_id is None for d in detections):
            self._fallback_tracker.assign(detections)

        self._resolve_stable_ids(detections)

        seen_ids = set()
        for i, det in enumerate(detections):
            if det.track_id is None:
                continue
            tid = det.track_id
            seen_ids.add(tid)
            if tid not in self.tracks:
                self.tracks[tid] = PersonTrack(track_id=tid, history_len=self.history_len)
            angle = torso_angles[i] if torso_angles is not None else None
            self.tracks[tid].update(det, timestamp, frame_idx, angle)

        
        stale_ids = []
        for tid, track in self.tracks.items():
            if tid not in seen_ids:
                track.frames_since_seen += 1
                if track.frames_since_seen > self.max_missing_frames:
                    stale_ids.append(tid)
        for tid in stale_ids:
            del self.tracks[tid]

        return self.tracks

    def active_tracks(self) -> Dict[int, PersonTrack]:
        return {tid: t for tid, t in self.tracks.items() if t.frames_since_seen == 0}

    def try_attach_low_conf(
        self,
        candidate_track_ids: List[int],
        low_conf_detections: List[Detection],
        frame_idx: int,
        timestamp: float,
        conf_thresh: float = 0.3,
        max_distance_ratio: float = 1.5,
    ) -> List[int]:
        candidates = [
            tid
            for tid in candidate_track_ids
            if tid in self.tracks and self.tracks[tid].frames_since_seen > 0 and self.tracks[tid].latest_bbox is not None
        ]
        if not candidates or not low_conf_detections:
            return []

        pairs: List[Tuple[float, int, Detection]] = []
        for tid in candidates:
            track = self.tracks[tid]
            lx1, ly1, lx2, ly2 = track.latest_bbox
            lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
            diag = track.bbox_diagonal()
            for det in low_conf_detections:
                dcx = float((det.bbox[0] + det.bbox[2]) / 2.0)
                dcy = float((det.bbox[1] + det.bbox[3]) / 2.0)
                dist = float(np.hypot(dcx - lcx, dcy - lcy)) / diag
                if dist <= max_distance_ratio:
                    pairs.append((dist, tid, det))
        pairs.sort(key=lambda p: p[0])

        attached: List[int] = []
        claimed_tids: set = set()
        claimed_dets: set = set()
        for dist, tid, det in pairs:
            if tid in claimed_tids or id(det) in claimed_dets:
                continue
            det.track_id = tid
            angle = compute_torso_angle(det.keypoints, conf_thresh)
            self.tracks[tid].update(det, timestamp, frame_idx, angle)
            claimed_tids.add(tid)
            claimed_dets.add(id(det))
            attached.append(tid)

        return attached
