"""
visualization.py

Drawing helpers: bounding boxes, pose skeleton, person ID, activity state,
and an optional debug overlay with the raw measurements used to tune the
fall-detection algorithm.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.pose_detector import SKELETON
from src.activity_classifier import (
    TORSO_UPRIGHT_MAX_DEG,
    TORSO_HORIZONTAL_MIN_DEG,
    KNEE_BENT_MAX_DEG,
    BBOX_ASPECT_LYING_MIN,
    WALK_SPEED_THRESHOLD,
)
from src.fall_detector import ENTER_THRESHOLD, SUSTAIN_THRESHOLD, RECOVER_EXIT_THRESHOLD

STATE_COLORS = {
    "STANDING": (60, 200, 60),      # green
    "WALKING": (230, 160, 30),      # blue-ish
    "SITTING": (30, 200, 230),      # yellow
    "FALLING": (0, 140, 255),       # orange
    "FALLEN": (0, 0, 255),          # red
    "UNKNOWN": (150, 150, 150),     # gray
}
KPT_COLOR = (0, 255, 255)
KPT_LOW_CONF_COLOR = (0, 0, 150)
SKELETON_COLOR = (255, 255, 255)


def color_for_state(state: str):
    return STATE_COLORS.get(state, STATE_COLORS["UNKNOWN"])


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf_thresh: float = 0.3) -> None:
    for a, b in SKELETON:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca >= conf_thresh and cb >= conf_thresh:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), SKELETON_COLOR, 2, cv2.LINE_AA)
    for x, y, c in keypoints:
        if c >= conf_thresh:
            cv2.circle(frame, (int(x), int(y)), 3, KPT_COLOR, -1, cv2.LINE_AA)
        elif c > 0:
            cv2.circle(frame, (int(x), int(y)), 2, KPT_LOW_CONF_COLOR, -1, cv2.LINE_AA)


def draw_person(
    frame: np.ndarray,
    track_id: int,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    display_state: str,
    state_confidence: float,
    bbox_conf: float,
    conf_thresh: float = 0.3,
    debug: bool = False,
    debug_info: Optional[dict] = None,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = color_for_state(display_state)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_skeleton(frame, keypoints, conf_thresh)

    label_top = f"Person {track_id}"
    label_bottom = f"{display_state}"
    _draw_label_block(frame, x1, y1, [label_top, label_bottom], color, frame.shape)

    if debug:
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        debug_line_count = 1 + sum(
            key in (debug_info or {}) and (debug_info or {}).get(key) is not None
            for key in ("torso_angle", "bbox_aspect", "speed", "fall_score", "fall_state", "transition")
        )
        debug_height = debug_line_count * 16 + 4
        if y2 + debug_height < frame_h:
            debug_y = y2
        else:
            debug_y = y1 + 48
        _draw_debug_overlay(frame, x1, debug_y, debug_info or {}, state_confidence, bbox_conf, frame.shape)


def _clamp_x(x: int, block_w: int, frame_w: int) -> int:
    return max(0, min(x, frame_w - block_w))


def _draw_label_block(frame: np.ndarray, x: int, y: int, lines, color, frame_shape) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    padding = 4
    line_height = 22
    frame_h, frame_w = frame_shape[0], frame_shape[1]

    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, w)

    block_w = max_width + padding * 2
    x = _clamp_x(x, block_w, frame_w)
    box_h = line_height * len(lines) + padding * 2
    box_top = max(0, min(y - box_h, frame_h - box_h))
    cv2.rectangle(frame, (x, box_top), (x + block_w, box_top + box_h), color, -1)

    for i, line in enumerate(lines):
        ty = box_top + padding + (i + 1) * line_height - 6
        cv2.putText(frame, line, (x + padding, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_debug_overlay(
    frame: np.ndarray, x: int, y: int, info: dict, state_conf: float, bbox_conf: float, frame_shape
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    color = (255, 255, 0)
    line_height = 16
    frame_h, frame_w = frame_shape[0], frame_shape[1]

    lines = [f"det_conf={bbox_conf:.2f}  state_conf={state_conf:.2f}"]
    if "torso_angle" in info and info["torso_angle"] is not None:
        lines.append(f"torso_angle={info['torso_angle']:.1f}deg")
    if "bbox_aspect" in info:
        lines.append(f"aspect={info['bbox_aspect']:.2f}")
    if "speed" in info:
        lines.append(f"speed={info['speed']:.2f} diag/s")
    if "fall_score" in info:
        lines.append(f"fall_score={info['fall_score']:.2f}")
    if "fall_state" in info:
        lines.append(f"fall_state={info['fall_state']}")
    if "transition" in info and info["transition"]:
        lines.append(f"transition: {info['transition']}")

    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, w)
    x = _clamp_x(x, max_width, frame_w)
    total_h = len(lines) * line_height
    y = max(0, min(y, frame_h - total_h - line_height))

    ty = y + line_height
    for line in lines:
        cv2.putText(frame, line, (x, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x, ty), font, scale, color, thickness, cv2.LINE_AA)
        ty += line_height


def draw_frame_header(frame: np.ndarray, frame_idx: int, fps: float, num_people: int, tracking_mode: str) -> int:
    text = f"frame={frame_idx}  fps={fps:.1f}  people={num_people}  tracker={tracking_mode}"
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return 28

_THRESHOLD_LEGEND_LINES = [
    f"THRESHOLDS  upright<={TORSO_UPRIGHT_MAX_DEG:.0f}deg  horizontal>={TORSO_HORIZONTAL_MIN_DEG:.0f}deg  "
    f"kneeBent<{KNEE_BENT_MAX_DEG:.0f}deg  aspectLying>={BBOX_ASPECT_LYING_MIN:.2f}  walkSpeed>={WALK_SPEED_THRESHOLD:.2f}",
    f"FALL SCORE  enter>={ENTER_THRESHOLD:.2f}  sustain>={SUSTAIN_THRESHOLD:.2f}  recover<={RECOVER_EXIT_THRESHOLD:.2f}",
]


def draw_threshold_panel(frame: np.ndarray, top_y: int, per_person_lines: list) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    line_height = 15
    color = (0, 255, 255)

    lines = list(_THRESHOLD_LEGEND_LINES) + list(per_person_lines)
    frame_w = frame.shape[1]

    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, w)
    panel_w = min(frame_w - 4, max_width + 8)
    panel_h = len(lines) * line_height + 6

    overlay = frame.copy()
    cv2.rectangle(overlay, (2, top_y), (2 + panel_w, top_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    ty = top_y + line_height
    for line in lines:
        cv2.putText(frame, line, (6, ty), font, scale, color, thickness, cv2.LINE_AA)
        ty += line_height
