"""
pose_detector.py

Thin wrapper around an Ultralytics YOLO pose model.

The keypoint layout below was NOT assumed. It was determined by:
  1. Loading the user's custom model (models/best.pt) and inspecting
     `model.model.yaml` -> kpt_shape = [17, 3]  (17 keypoints, x/y/visibility)
  2. Inspecting the checkpoint's train_args, which shows the model was
     fine-tuned from Ultralytics' pretrained `yolo26s-pose.pt` (COCO-pose
     architecture) without changing kpt_shape.
  3. Running a real inference pass on a photo containing upright people and
     confirming the 17 output keypoints are monotonically ordered from head
     to feet in exactly the COCO-pose arrangement (nose/eyes/ears, then
     shoulders, elbows, wrists, hips, knees, ankles), with correct left/right
     pairing.

Class list: model.names = {0: 'person'} (single class, as expected for a
pose model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Keypoint layout, confirmed empirically (see module docstring).
# --------------------------------------------------------------------------
KEYPOINT_NAMES = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
]
KPT = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# Skeleton edges as (a, b) index pairs, 0-indexed version of Ultralytics'
# standard COCO-pose skeleton.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


@dataclass
class Detection:

    track_id: Optional[int]
    bbox: np.ndarray  # (x1, y1, x2, y2) in pixel coords
    bbox_conf: float
    keypoints: np.ndarray  # (17, 3) -> x, y, confidence (pixel coords)
    class_id: int = 0


def resolve_device(requested: str) -> str:
    requested = (requested or "cpu").strip().lower()
    if requested in ("cpu", ""):
        return "cpu"
    if not torch.cuda.is_available():
        print(
            f"[pose_detector] WARNING: device='{requested}' requested but "
            f"CUDA is not available on this machine. Falling back to CPU."
        )
        return "cpu"
    return requested


class PoseDetector:

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf: float = 0.4,
        iou: float = 0.5,
        imgsz: int = 640,
        tracker: str = "botsort.yaml",
    ):
        self.model = YOLO(model_path)
        if self.model.task != "pose":
            raise ValueError(
                f"Loaded model task is '{self.model.task}', expected 'pose'. "
                f"Make sure --model points to your pose model (best.pt)."
            )
        self.device = resolve_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker_cfg = tracker
        self.class_names = self.model.names

        self.tracking_available = True
        self._tracking_checked = False

    def infer(self, frame: np.ndarray, use_tracking: bool = True) -> List[Detection]:
        try:
            if use_tracking and self.tracking_available:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    imgsz=self.imgsz,
                    tracker=self.tracker_cfg,
                    verbose=False,
                )
                self._tracking_checked = True
            else:
                raise RuntimeError("tracking disabled")
        except Exception as exc:
            if use_tracking and self.tracking_available:
                print(
                    f"[pose_detector] WARNING: native tracking failed "
                    f"({exc}); falling back to per-frame detection + "
                    f"external IoU tracker."
                )
                self.tracking_available = False
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                imgsz=self.imgsz,
                verbose=False,
            )

        return self._parse_results(results)

    def infer_low_conf(self, frame: np.ndarray, low_conf: float) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=low_conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        return self._parse_results(results)

    def _parse_results(self, results) -> List[Detection]:
        if not results:
            return []
        r = results[0]
        detections: List[Detection] = []

        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        boxes_conf = r.boxes.conf.cpu().numpy()
        boxes_cls = r.boxes.cls.cpu().numpy().astype(int)
        track_ids = None
        if r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)

        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts_all = r.keypoints.data.cpu().numpy()  # (N, 17, 3)
        else:
            kpts_all = np.zeros((len(boxes_xyxy), NUM_KEYPOINTS, 3), dtype=np.float32)

        for i in range(len(boxes_xyxy)):
            detections.append(
                Detection(
                    track_id=int(track_ids[i]) if track_ids is not None else None,
                    bbox=boxes_xyxy[i],
                    bbox_conf=float(boxes_conf[i]),
                    keypoints=kpts_all[i] if i < len(kpts_all) else np.zeros((NUM_KEYPOINTS, 3)),
                    class_id=int(boxes_cls[i]),
                )
            )
        return detections
