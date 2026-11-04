"""
activity_classifier.py

Pure, testable geometry + temporal-smoothing logic for classifying a
person's posture/activity as STANDING, SITTING, WALKING, CROUCHING (a
transient bend, never shown directly) or LYING (a candidate the
fall_detector module turns into FALLING/FALLEN after temporal persistence
checks).

The functions here operate on plain (17, 3) keypoint arrays and (4,) bbox
arrays so they can be unit-tested with synthetic data, independent of the
video pipeline, tracker, or Ultralytics.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.pose_detector import KPT


TORSO_UPRIGHT_MAX_DEG = 35.0      # torso angle <= this => upright (stand/sit)
TORSO_HORIZONTAL_MIN_DEG = 55.0   # torso angle >= this => lying/fallen candidate
KNEE_BENT_MAX_DEG = 150.0         # knee angle < this => "bent" (sitting-like)
BBOX_ASPECT_LYING_MIN = 1.15      # width/height >= this supports a lying posture

WALK_SPEED_THRESHOLD = 0.10       # bbox-diagonals of center movement, per second
WALK_LEG_VAR_THRESHOLD = 3.5      # px stddev of ankle separation over the window
MIN_MOTION_FRAMES = 6             # min frames of history before judging motion


def get_point(keypoints: np.ndarray, idx: int, conf_thresh: float = 0.3) -> Optional[Tuple[float, float]]:
    if keypoints is None or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    if c < conf_thresh:
        return None
    return (float(x), float(y))


def midpoint(p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def compute_angle_at_point(
    a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]], c: Optional[Tuple[float, float]]
) -> Optional[float]:
    if a is None or b is None or c is None:
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def compute_torso_angle(keypoints: np.ndarray, conf_thresh: float = 0.3) -> Optional[float]:
    shoulder_mid = midpoint(
        get_point(keypoints, KPT["left_shoulder"], conf_thresh),
        get_point(keypoints, KPT["right_shoulder"], conf_thresh),
    )
    hip_mid = midpoint(
        get_point(keypoints, KPT["left_hip"], conf_thresh),
        get_point(keypoints, KPT["right_hip"], conf_thresh),
    )
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def compute_knee_angles(keypoints: np.ndarray, conf_thresh: float = 0.3) -> List[float]:
    angles = []
    for side in ("left", "right"):
        hip = get_point(keypoints, KPT[f"{side}_hip"], conf_thresh)
        knee = get_point(keypoints, KPT[f"{side}_knee"], conf_thresh)
        ankle = get_point(keypoints, KPT[f"{side}_ankle"], conf_thresh)
        angle = compute_angle_at_point(hip, knee, ankle)
        if angle is not None:
            angles.append(angle)
    return angles


def compute_bbox_aspect(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2) - float(x1))
    h = max(1.0, float(y2) - float(y1))
    return w / h


def count_valid_keypoints(keypoints: np.ndarray, conf_thresh: float = 0.3) -> int:
    if keypoints is None:
        return 0
    return int(np.sum(keypoints[:, 2] >= conf_thresh))


@dataclass
class PostureFeatures:
    torso_angle: Optional[float]
    knee_angle: Optional[float]  
    bbox_aspect: float
    valid_kpt_count: int
    avg_conf: float


POSTURE_RELEVANT_KEYPOINTS = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
_POSTURE_RELEVANT_IDX = [KPT[name] for name in POSTURE_RELEVANT_KEYPOINTS]


def compute_posture_features(keypoints: np.ndarray, bbox: np.ndarray, conf_thresh: float = 0.3) -> PostureFeatures:
    torso_angle = compute_torso_angle(keypoints, conf_thresh)
    knee_angles = compute_knee_angles(keypoints, conf_thresh)
    knee_angle = float(np.mean(knee_angles)) if knee_angles else None
    bbox_aspect = compute_bbox_aspect(bbox)
    valid_count = count_valid_keypoints(keypoints, conf_thresh)
    if keypoints is not None and len(keypoints):
        avg_conf = float(np.mean(keypoints[_POSTURE_RELEVANT_IDX, 2]))
    else:
        avg_conf = 0.0
    return PostureFeatures(
        torso_angle=torso_angle,
        knee_angle=knee_angle,
        bbox_aspect=bbox_aspect,
        valid_kpt_count=valid_count,
        avg_conf=avg_conf,
    )


POSTURE_STANDING = "STANDING"
POSTURE_SITTING = "SITTING"
POSTURE_CROUCHING = "CROUCHING"
POSTURE_LYING = "LYING"
POSTURE_UNKNOWN = "UNKNOWN"


def classify_posture(features: PostureFeatures) -> Tuple[str, float]:
    if features.valid_kpt_count < 4 or features.torso_angle is None:
        return POSTURE_UNKNOWN, 0.0

    score = max(0.3, min(1.0, features.avg_conf))

    if features.torso_angle >= TORSO_HORIZONTAL_MIN_DEG or features.bbox_aspect >= BBOX_ASPECT_LYING_MIN:
        return POSTURE_LYING, score

    if features.torso_angle <= TORSO_UPRIGHT_MAX_DEG:
        if features.knee_angle is None:
            return POSTURE_UNKNOWN, 0.0
        if features.knee_angle < KNEE_BENT_MAX_DEG:
            return POSTURE_SITTING, score
        return POSTURE_STANDING, score

    return POSTURE_CROUCHING, score * 0.8


def compute_center_speed(
    center_history: List[Tuple[float, float]], timestamp_history: List[float], bbox_diagonal: float
) -> float:
    if len(center_history) < 2:
        return 0.0
    p0, p1 = center_history[0], center_history[-1]
    t0, t1 = timestamp_history[0], timestamp_history[-1]
    dt = max(1e-3, t1 - t0)
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    diag = max(1.0, bbox_diagonal)
    return (dist / diag) / dt


def compute_leg_motion_variance(keypoints_history: List[np.ndarray], conf_thresh: float = 0.3) -> float:
    dists = []
    for kp in keypoints_history:
        la = get_point(kp, KPT["left_ankle"], conf_thresh)
        ra = get_point(kp, KPT["right_ankle"], conf_thresh)
        if la is not None and ra is not None:
            dists.append(abs(la[0] - ra[0]))
    if len(dists) < 3:
        return 0.0
    return float(np.std(dists))


class StateSmoother:

    def __init__(self, window: int = 15, min_consecutive: int = 5):
        self.window: Deque[str] = deque(maxlen=window)
        self.min_consecutive = min_consecutive
        self.displayed_state = "UNKNOWN"
        self._pending_state: Optional[str] = None
        self._pending_count = 0
        self.consecutive_displayed = 0

    def update(self, raw_state: str) -> Tuple[str, float, int]:
        self.window.append(raw_state)
        counts = Counter(self.window)
        _, majority_count = counts.most_common(1)[0]
        confidence = majority_count / len(self.window)

        if raw_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_state
            self._pending_count = 1

        if raw_state == self.displayed_state:
            self.consecutive_displayed += 1
        elif self._pending_count >= self.min_consecutive:
            self.displayed_state = raw_state
            self.consecutive_displayed = self._pending_count

        return self.displayed_state, confidence, self.consecutive_displayed


@dataclass
class ClassificationResult:
    raw_state: str
    smoothed_state: str
    confidence: float
    consecutive_frames: int
    is_moving: bool
    features: PostureFeatures
    geometric_posture: str = POSTURE_UNKNOWN


class ActivityClassifier:

    def __init__(
        self,
        conf_thresh: float = 0.3,
        min_consecutive_frames: int = 5,
        smoothing_window: int = 15,
        walk_speed_threshold: float = WALK_SPEED_THRESHOLD,
        walk_leg_var_threshold: float = WALK_LEG_VAR_THRESHOLD,
        min_motion_frames: int = MIN_MOTION_FRAMES,
    ):
        self.conf_thresh = conf_thresh
        self.min_consecutive_frames = min_consecutive_frames
        self.smoothing_window = smoothing_window
        self.walk_speed_threshold = walk_speed_threshold
        self.walk_leg_var_threshold = walk_leg_var_threshold
        self.min_motion_frames = min_motion_frames
        self._smoothers: Dict[int, StateSmoother] = {}

    def _smoother_for(self, track_id: int) -> StateSmoother:
        if track_id not in self._smoothers:
            self._smoothers[track_id] = StateSmoother(
                window=self.smoothing_window, min_consecutive=self.min_consecutive_frames
            )
        return self._smoothers[track_id]

    def classify(self, track) -> ClassificationResult:
        keypoints = track.latest_keypoints
        bbox = track.latest_bbox
        features = compute_posture_features(keypoints, bbox, self.conf_thresh)
        posture_label, posture_score = classify_posture(features)

        raw_state = posture_label
        is_moving = False

        if posture_label in (POSTURE_STANDING, POSTURE_CROUCHING):
            hist_len = len(track.center_history)
            if hist_len >= self.min_motion_frames:
                window = min(hist_len, self.smoothing_window)
                centers = list(track.center_history)[-window:]
                times = list(track.timestamp_history)[-window:]
                kpt_hist = list(track.keypoints_history)[-window:]
                speed = compute_center_speed(centers, times, track.bbox_diagonal())
                leg_var = compute_leg_motion_variance(kpt_hist, self.conf_thresh)
                is_moving = speed >= self.walk_speed_threshold or leg_var >= self.walk_leg_var_threshold
                if is_moving and posture_label == POSTURE_STANDING:
                    raw_state = "WALKING"

        if posture_label == POSTURE_CROUCHING and raw_state == POSTURE_CROUCHING:
            raw_state = POSTURE_STANDING

        if posture_label == POSTURE_LYING:
            raw_state = "LYING"

        smoother = self._smoother_for(track.track_id)
        smoothed_state, confidence, consecutive = smoother.update(raw_state)
        final_confidence = confidence * (posture_score if posture_score > 0 else 0.5)

        return ClassificationResult(
            raw_state=raw_state,
            smoothed_state=smoothed_state,
            confidence=final_confidence,
            consecutive_frames=consecutive,
            is_moving=is_moving,
            features=features,
            geometric_posture=posture_label,
        )

    def remove_track(self, track_id: int) -> None:
        self._smoothers.pop(track_id, None)
