"""
fall_detector.py

Highest-priority module: temporal fall detection.

A person is never marked FALLEN from a single frame. Instead each track
runs its own small state machine:

    NORMAL -> POSSIBLE_FALL -> FALLEN -> RECOVERING -> NORMAL

Every frame we compute a continuous `fall_score` in [0, 1] from six signals
(body orientation, bounding-box aspect ratio, downward velocity, angular
velocity of the torso, head drop, and bbox height-collapse), then feed that
score through the state machine so a fall is only confirmed once
"horizontal + falling-like motion" evidence has persisted for
`fall_threshold` seconds -- filtering out single-frame bbox flukes, quick
crouches/bends, and normal sitting.

Note on camera angle: torso-angle and bbox-aspect are strong signals when
the camera views a fall roughly side-on, but they can both fail to read
"horizontal" when a person falls *away* from an elevated/angled camera
(along its depth axis) -- the 2D shoulder-hip vector foreshortens instead
of rotating toward horizontal, and the bbox doesn't necessarily widen. The
bbox height-collapse signal (current height vs. that person's own recent
standing-height baseline) is included specifically because it stays
informative in that camera geometry too: a person who is no longer
standing/sitting at their usual height, and stays that way, is informative
regardless of which direction they fell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.activity_classifier import (
    TORSO_UPRIGHT_MAX_DEG,
    TORSO_HORIZONTAL_MIN_DEG,
    compute_bbox_aspect,
    compute_torso_angle,
    get_point,
    midpoint,
)
from src.pose_detector import KPT

ENTER_THRESHOLD = 0.40          # fall_score to move NORMAL -> POSSIBLE_FALL
SUSTAIN_THRESHOLD = 0.30        # fall_score needed to keep accumulating persistence credit
RECOVER_EXIT_THRESHOLD = 0.22   # fall_score below this counts as "recovered"
DEFAULT_RECOVERY_SECONDS = 1.0  # how long upright evidence must persist to clear FALLEN

DOWNWARD_VELOCITY_THRESHOLD = 0.8   # bbox-diagonals/sec of downward center motion
ANGLE_VELOCITY_THRESHOLD = 60.0     # degrees/sec of torso-angle change
HEADDROP_VELOCITY_THRESHOLD = 0.8   # bbox-diagonals/sec of downward head motion
VELOCITY_WINDOW_SECONDS = 0.5       # window used for the two velocity signals

HEIGHT_COLLAPSE_HIGH_RATIO = 0.80
HEIGHT_COLLAPSE_LOW_RATIO = 0.45
REFERENCE_HEIGHT_EMA_ALPHA = 0.05

W_ANGLE = 0.25
W_ANGLE_VEL = 0.20
W_ASPECT = 0.10
W_VELOCITY = 0.15
W_HEADDROP = 0.10
W_HEIGHT_COLLAPSE = 0.20

STATE_NORMAL = "NORMAL"
STATE_POSSIBLE_FALL = "POSSIBLE_FALL"
STATE_FALLEN = "FALLEN"
STATE_RECOVERING = "RECOVERING"
STATE_SUSPECTED_FALLEN_OCCLUDED = "SUSPECTED_FALLEN_OCCLUDED"
OCCLUSION_GRACE_SECONDS = 1.5
LEAKY_DECAY_STEP = 0.5


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))



def _angle_score(torso_angle: Optional[float]) -> Tuple[float, bool]:
    if torso_angle is None:
        return 0.0, False
    if torso_angle <= TORSO_UPRIGHT_MAX_DEG:
        return 0.0, True
    if torso_angle >= TORSO_HORIZONTAL_MIN_DEG:
        return 1.0, True
    span = TORSO_HORIZONTAL_MIN_DEG - TORSO_UPRIGHT_MAX_DEG
    return (torso_angle - TORSO_UPRIGHT_MAX_DEG) / span, True


def _aspect_score(bbox_aspect: float) -> Tuple[float, bool]:
    return _clip01((bbox_aspect - 0.8) / (1.6 - 0.8)), True


def _height_collapse_score(current_height: float, reference_height: float) -> Tuple[float, bool]:
    if reference_height <= 1.0:
        return 0.0, False
    ratio = current_height / reference_height
    if ratio >= HEIGHT_COLLAPSE_HIGH_RATIO:
        return 0.0, True
    if ratio <= HEIGHT_COLLAPSE_LOW_RATIO:
        return 1.0, True
    span = HEIGHT_COLLAPSE_HIGH_RATIO - HEIGHT_COLLAPSE_LOW_RATIO
    return (HEIGHT_COLLAPSE_HIGH_RATIO - ratio) / span, True


def _windowed_points(
    values: List[Optional[Tuple[float, float]]], timestamps: List[float], window_seconds: float
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float]:
    if not values:
        return None, None, 0.0
    t_end = timestamps[-1]
    first_pt, first_t = None, None
    last_pt, last_t = None, None
    for pt, t in zip(values, timestamps):
        if t_end - t > window_seconds:
            continue
        if pt is None:
            continue
        if first_pt is None:
            first_pt, first_t = pt, t
        last_pt, last_t = pt, t
    if first_pt is None or last_pt is None or first_t == last_t:
        return None, None, 0.0
    return first_pt, last_pt, max(1e-3, last_t - first_t)


def _vertical_velocity_score(
    point_history: List[Optional[Tuple[float, float]]],
    timestamps: List[float],
    bbox_diagonal: float,
    threshold: float,
) -> Tuple[float, bool]:
    p0, p1, dt = _windowed_points(point_history, timestamps, VELOCITY_WINDOW_SECONDS)
    if p0 is None:
        return 0.0, False
    vy = (p1[1] - p0[1]) / max(1.0, bbox_diagonal) / dt
    if vy <= 0:
        return 0.0, True
    return _clip01(vy / threshold), True


def _angle_velocity_score(
    angle_history: List[Optional[float]], timestamps: List[float], threshold: float
) -> Tuple[float, bool]:
    t_end = timestamps[-1] if timestamps else 0.0
    first_a, first_t = None, None
    last_a, last_t = None, None
    for a, t in zip(angle_history, timestamps):
        if t_end - t > VELOCITY_WINDOW_SECONDS:
            continue
        if a is None:
            continue
        if first_a is None:
            first_a, first_t = a, t
        last_a, last_t = a, t
    if first_a is None or last_a is None or first_t == last_t:
        return 0.0, False
    dt = max(1e-3, last_t - first_t)
    d_angle = (last_a - first_a) / dt
    if d_angle <= 0:
        return 0.0, True
    return _clip01(d_angle / threshold), True


def compute_fall_score(
    keypoints: np.ndarray,
    bbox: np.ndarray,
    center_history: List[Tuple[float, float]],
    timestamp_history: List[float],
    angle_history: List[Optional[float]],
    keypoints_history: List[np.ndarray],
    bbox_diagonal: float,
    reference_height: float = 0.0,
    conf_thresh: float = 0.3,
) -> Tuple[float, dict]:
    torso_angle = compute_torso_angle(keypoints, conf_thresh)
    bbox_aspect = compute_bbox_aspect(bbox)
    current_height = max(1.0, float(bbox[3]) - float(bbox[1]))

    angle_sc, angle_avail = _angle_score(torso_angle)
    aspect_sc, aspect_avail = _aspect_score(bbox_aspect)
    vel_sc, vel_avail = _vertical_velocity_score(
        list(center_history), list(timestamp_history), bbox_diagonal, DOWNWARD_VELOCITY_THRESHOLD
    )
    angle_vel_sc, angle_vel_avail = _angle_velocity_score(
        list(angle_history), list(timestamp_history), ANGLE_VELOCITY_THRESHOLD
    )
    height_sc, height_avail = _height_collapse_score(current_height, reference_height)

    head_points = []
    for kp in keypoints_history:
        nose = get_point(kp, KPT["nose"], conf_thresh)
        if nose is None:
            sh = midpoint(
                get_point(kp, KPT["left_shoulder"], conf_thresh),
                get_point(kp, KPT["right_shoulder"], conf_thresh),
            )
            nose = sh
        head_points.append(nose)
    headdrop_sc, headdrop_avail = _vertical_velocity_score(
        head_points, list(timestamp_history), bbox_diagonal, HEADDROP_VELOCITY_THRESHOLD
    )

    components = (
        (W_ANGLE, angle_sc, angle_avail),
        (W_ANGLE_VEL, angle_vel_sc, angle_vel_avail),
        (W_ASPECT, aspect_sc, aspect_avail),
        (W_VELOCITY, vel_sc, vel_avail),
        (W_HEADDROP, headdrop_sc, headdrop_avail),
        (W_HEIGHT_COLLAPSE, height_sc, height_avail),
    )
    weight_sum = sum(w for w, _, avail in components if avail)
    if weight_sum <= 0:
        score = 0.0
    else:
        score = sum(w * sc for w, sc, avail in components if avail) / weight_sum

    details = {
        "torso_angle": torso_angle,
        "bbox_aspect": bbox_aspect,
        "angle_score": angle_sc,
        "aspect_score": aspect_sc,
        "velocity_score": vel_sc,
        "angle_velocity_score": angle_vel_sc,
        "headdrop_score": headdrop_sc,
        "height_score": height_sc,
        "current_height": current_height,
        "reference_height": reference_height,
        "available_weight": weight_sum,
    }
    return _clip01(score), details


@dataclass
class FallEvent:
    timestamp: float
    track_id: int
    event: str
    confidence: float
    previous_state: str
    current_state: str


@dataclass
class FallResult:
    state: str
    fall_score: float
    details: dict
    event: Optional[FallEvent]
    transition: Optional[str]


_UPRIGHT_ACTIVITY_STATES = {"STANDING", "WALKING"}


@dataclass
class _FallTrackState:
    state: str = STATE_NORMAL
    possible_count: float = 0.0
    recovery_count: int = 0
    fall_event_emitted: bool = False
    pending_previous_state: str = "UNKNOWN"
    reference_height: float = 0.0
    last_score: float = 0.0
    last_details: dict = field(default_factory=dict)
    occlusion_event_emitted: bool = False


class FallDetector:

    def __init__(
        self,
        conf_thresh: float = 0.3,
        fall_threshold_seconds: float = 0.8,
        recovery_seconds: float = DEFAULT_RECOVERY_SECONDS,
    ):
        self.conf_thresh = conf_thresh
        self.fall_threshold_seconds = fall_threshold_seconds
        self.recovery_seconds = recovery_seconds
        self._states: Dict[int, _FallTrackState] = {}

    def _state_for(self, track_id: int) -> _FallTrackState:
        if track_id not in self._states:
            self._states[track_id] = _FallTrackState()
        return self._states[track_id]

    def update(
        self,
        track,
        fps: float,
        current_activity_state: str,
        raw_posture: Optional[str] = None,
    ) -> FallResult:
        st = self._state_for(track.track_id)

        if st.state == STATE_SUSPECTED_FALLEN_OCCLUDED:
            st.state = STATE_POSSIBLE_FALL

        keypoints = track.latest_keypoints
        bbox = track.latest_bbox
        timestamps = list(track.timestamp_history)
        if raw_posture is not None:
            is_upright_evidence = raw_posture == "STANDING"
        else:
            is_upright_evidence = current_activity_state in _UPRIGHT_ACTIVITY_STATES
        if st.state == STATE_NORMAL and is_upright_evidence and bbox is not None:
            current_h = max(1.0, float(bbox[3]) - float(bbox[1]))
            if st.reference_height <= 0:
                st.reference_height = current_h
            else:
                st.reference_height = (
                    1 - REFERENCE_HEIGHT_EMA_ALPHA
                ) * st.reference_height + REFERENCE_HEIGHT_EMA_ALPHA * current_h

        score, details = compute_fall_score(
            keypoints=keypoints,
            bbox=bbox,
            center_history=list(track.center_history),
            timestamp_history=timestamps,
            angle_history=list(track.angle_history),
            keypoints_history=list(track.keypoints_history),
            bbox_diagonal=track.bbox_diagonal(),
            reference_height=st.reference_height,
            conf_thresh=self.conf_thresh,
        )
        st.last_score = score
        st.last_details = details

        fps = max(1e-3, fps)
        required_frames = max(2, round(self.fall_threshold_seconds * fps))
        recovery_required_frames = max(2, round(self.recovery_seconds * fps))

        transition = None
        event = None
        now = timestamps[-1] if timestamps else 0.0

        if st.state == STATE_NORMAL:
            if score >= ENTER_THRESHOLD:
                st.state = STATE_POSSIBLE_FALL
                st.possible_count = 1
                st.pending_previous_state = current_activity_state
                transition = "NORMAL -> POSSIBLE_FALL"

        elif st.state == STATE_POSSIBLE_FALL:
            if score >= SUSTAIN_THRESHOLD:
                st.possible_count = min(required_frames, st.possible_count + 1)
                if st.possible_count >= required_frames:
                    st.state = STATE_FALLEN
                    st.possible_count = 0
                    transition = "POSSIBLE_FALL -> FALLEN"
                    if not st.fall_event_emitted:
                        event = FallEvent(
                            timestamp=now,
                            track_id=track.track_id,
                            event="FALLEN",
                            confidence=score,
                            previous_state=st.pending_previous_state,
                            current_state="FALLEN",
                        )
                        st.fall_event_emitted = True
            else:
                st.possible_count -= LEAKY_DECAY_STEP
                if st.possible_count <= 0:
                    st.state = STATE_NORMAL
                    st.possible_count = 0.0
                    transition = "POSSIBLE_FALL -> NORMAL"

        elif st.state == STATE_FALLEN:
            if score <= RECOVER_EXIT_THRESHOLD:
                st.state = STATE_RECOVERING
                st.recovery_count = 1
                transition = "FALLEN -> RECOVERING"
            else:
                st.recovery_count = 0

        elif st.state == STATE_RECOVERING:
            if score <= RECOVER_EXIT_THRESHOLD:
                st.recovery_count = min(recovery_required_frames, st.recovery_count + 1)
                if st.recovery_count >= recovery_required_frames:
                    st.state = STATE_NORMAL
                    st.recovery_count = 0
                    st.fall_event_emitted = False
                    st.occlusion_event_emitted = False
                    transition = "RECOVERING -> NORMAL"
            else:
                st.recovery_count -= 1
                if st.recovery_count <= 0:
                    st.state = STATE_FALLEN
                    st.recovery_count = 0
                    transition = "RECOVERING -> FALLEN"

        return FallResult(state=st.state, fall_score=score, details=details, event=event, transition=transition)

    def check_occlusion(self, track, fps: float) -> Optional[FallResult]:
        if track.track_id not in self._states:
            return None
        st = self._states[track.track_id]
        if st.state not in (STATE_POSSIBLE_FALL, STATE_FALLEN, STATE_SUSPECTED_FALLEN_OCCLUDED):
            return None

        fps = max(1e-3, fps)
        missing_seconds = track.frames_since_seen / fps
        transition = None
        event = None

        if missing_seconds <= OCCLUSION_GRACE_SECONDS:
            if st.state != STATE_SUSPECTED_FALLEN_OCCLUDED:
                prev_state = st.state
                st.state = STATE_SUSPECTED_FALLEN_OCCLUDED
                transition = f"{prev_state} -> SUSPECTED_FALLEN_OCCLUDED"
        else:
            if not st.occlusion_event_emitted:
                event = FallEvent(
                    timestamp=track.timestamp_history[-1] if track.timestamp_history else 0.0,
                    track_id=track.track_id,
                    event="SUSPECTED_FALLEN_OCCLUDED",
                    confidence=st.last_score,
                    previous_state=st.pending_previous_state,
                    current_state="SUSPECTED_FALLEN_OCCLUDED",
                )
                st.occlusion_event_emitted = True
                transition = "SUSPECTED_FALLEN_OCCLUDED (grace period exceeded, no re-detection)"

        return FallResult(state=st.state, fall_score=st.last_score, details=st.last_details, event=event, transition=transition)

    def remove_track(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    @staticmethod
    def display_state(fall_state: str) -> Optional[str]:
        if fall_state == STATE_POSSIBLE_FALL:
            return "FALLING"
        if fall_state == STATE_FALLEN:
            return "FALLEN"
        if fall_state == STATE_SUSPECTED_FALLEN_OCCLUDED:
            return "FALLEN? (occluded)"
        return None
