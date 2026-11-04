"""
Unit tests for the activity classification / fall detection logic.

These tests use synthetic (hand-built) keypoint arrays instead of real
video, so they run instantly and don't depend on the model or GPU/CPU
inference. Run from the project root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import unittest

import numpy as np

from src.activity_classifier import (
    ActivityClassifier,
    StateSmoother,
    classify_posture,
    compute_posture_features,
)
from src.fall_detector import (
    FallDetector,
    STATE_FALLEN,
    STATE_NORMAL,
    STATE_POSSIBLE_FALL,
    STATE_SUSPECTED_FALLEN_OCCLUDED,
    OCCLUSION_GRACE_SECONDS,
    compute_fall_score,
)
from src.pose_detector import KPT, NUM_KEYPOINTS, Detection
from src.tracker import PersonTrack


def make_keypoints(points: dict, conf: float = 0.9) -> np.ndarray:
    """points: {keypoint_name: (x, y)}. Unlisted keypoints get conf=0."""
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for name, (x, y) in points.items():
        kp[KPT[name]] = [x, y, conf]
    return kp


def bbox_from_keypoints(kp: np.ndarray, pad: float = 10.0) -> np.ndarray:
    valid = kp[kp[:, 2] > 0]
    if len(valid) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float32)
    x1, y1 = valid[:, 0].min() - pad, valid[:, 1].min() - pad
    x2, y2 = valid[:, 0].max() + pad, valid[:, 1].max() + pad
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def standing_keypoints() -> np.ndarray:
    return make_keypoints(
        {
            "nose": (100, 50),
            "left_shoulder": (90, 100),
            "right_shoulder": (110, 100),
            "left_hip": (92, 220),
            "right_hip": (108, 220),
            "left_knee": (92, 320),
            "right_knee": (108, 320),
            "left_ankle": (92, 420),
            "right_ankle": (108, 420),
        }
    )


def sitting_keypoints() -> np.ndarray:
    return make_keypoints(
        {
            "nose": (100, 50),
            "left_shoulder": (90, 100),
            "right_shoulder": (110, 100),
            "left_hip": (92, 180),
            "right_hip": (108, 180),
            "left_knee": (150, 185),
            "right_knee": (150, 185),
            "left_ankle": (150, 280),
            "right_ankle": (150, 280),
        }
    )


def lying_keypoints() -> np.ndarray:
    return make_keypoints(
        {
            "nose": (50, 150),
            "left_shoulder": (120, 145),
            "right_shoulder": (120, 155),
            "left_hip": (250, 148),
            "right_hip": (250, 152),
            "left_knee": (350, 150),
            "right_knee": (350, 150),
            "left_ankle": (430, 150),
            "right_ankle": (430, 150),
        }
    )


def make_track(track_id: int, history_len: int = 30) -> PersonTrack:
    return PersonTrack(track_id=track_id, history_len=history_len)


def push_frame(track: PersonTrack, kp: np.ndarray, t: float, frame_idx: int, conf_thresh: float = 0.3):
    from src.activity_classifier import compute_torso_angle

    bbox = bbox_from_keypoints(kp)
    det = Detection(track_id=track.track_id, bbox=bbox, bbox_conf=0.9, keypoints=kp)
    angle = compute_torso_angle(kp, conf_thresh)
    track.update(det, t, frame_idx, angle)
    return bbox


class TestPostureGeometry(unittest.TestCase):
    def test_standing_posture(self):
        kp = standing_keypoints()
        bbox = bbox_from_keypoints(kp)
        features = compute_posture_features(kp, bbox)
        label, score = classify_posture(features)
        self.assertEqual(label, "STANDING")
        self.assertGreater(score, 0)

    def test_sitting_posture(self):
        kp = sitting_keypoints()
        bbox = bbox_from_keypoints(kp)
        features = compute_posture_features(kp, bbox)
        label, _ = classify_posture(features)
        self.assertEqual(label, "SITTING")

    def test_lying_posture(self):
        kp = lying_keypoints()
        bbox = bbox_from_keypoints(kp)
        features = compute_posture_features(kp, bbox)
        label, _ = classify_posture(features)
        self.assertEqual(label, "LYING")

    def test_missing_keypoints_is_unknown(self):
        kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)  # all zero confidence
        bbox = np.array([0, 0, 100, 200], dtype=np.float32)
        features = compute_posture_features(kp, bbox)
        label, score = classify_posture(features)
        self.assertEqual(label, "UNKNOWN")
        self.assertEqual(score, 0.0)

    def test_upright_torso_with_no_leg_evidence_is_unknown_not_standing(self):
        kp = standing_keypoints()
        for name in ("left_knee", "right_knee", "left_ankle", "right_ankle"):
            kp[KPT[name]] = [0, 0, 0.0]
        bbox = bbox_from_keypoints(kp)
        features = compute_posture_features(kp, bbox)
        label, score = classify_posture(features)
        self.assertEqual(label, "UNKNOWN")
        self.assertEqual(score, 0.0)

    def test_partial_missing_keypoints_still_classifies(self):
        kp = standing_keypoints()
        kp[KPT["right_hip"]] = [0, 0, 0.0]
        bbox = bbox_from_keypoints(kp)
        features = compute_posture_features(kp, bbox)
        label, _ = classify_posture(features)
        self.assertIn(label, ("STANDING", "SITTING"))


class TestStateSmoother(unittest.TestCase):
    def test_debounces_alternating_noise(self):
        smoother = StateSmoother(window=15, min_consecutive=5)
        states = ["STANDING", "WALKING"] * 10
        displayed_states = [smoother.update(s)[0] for s in states]
        self.assertTrue(all(s in ("UNKNOWN", "STANDING") for s in displayed_states))

    def test_switches_after_sustained_evidence(self):
        smoother = StateSmoother(window=15, min_consecutive=5)
        for _ in range(5):
            smoother.update("STANDING")
        displayed, _, _ = smoother.update("STANDING")
        self.assertEqual(displayed, "STANDING")

        result = None
        for _ in range(5):
            result = smoother.update("WALKING")
        self.assertEqual(result[0], "WALKING")


class TestActivityClassifierIntegration(unittest.TestCase):
    def test_stationary_standing_person_not_classified_as_walking(self):
        clf = ActivityClassifier(min_consecutive_frames=3, smoothing_window=10)
        track = make_track(track_id=1)
        kp = standing_keypoints()
        result = None
        for i in range(20):
            jittered = kp.copy()
            jittered[:, 0] += (0.3 if i % 2 == 0 else -0.3)
            push_frame(track, jittered, t=i / 30.0, frame_idx=i)
            result = clf.classify(track)
        self.assertEqual(result.smoothed_state, "STANDING")

    def test_walking_person_detected_via_motion(self):
        clf = ActivityClassifier(min_consecutive_frames=3, smoothing_window=10)
        track = make_track(track_id=2)
        kp = standing_keypoints()
        result = None
        for i in range(20):
            shifted = kp.copy()
            shifted[:, 0] += i * 6.0  # steady horizontal walking motion
            push_frame(track, shifted, t=i / 30.0, frame_idx=i)
            result = clf.classify(track)
        self.assertEqual(result.smoothed_state, "WALKING")

    def test_multiple_people_independent_state(self):
        clf = ActivityClassifier(min_consecutive_frames=3, smoothing_window=10)
        standing_track = make_track(track_id=10)
        sitting_track = make_track(track_id=11)
        r1 = r2 = None
        for i in range(10):
            push_frame(standing_track, standing_keypoints(), t=i / 30.0, frame_idx=i)
            push_frame(sitting_track, sitting_keypoints(), t=i / 30.0, frame_idx=i)
            r1 = clf.classify(standing_track)
            r2 = clf.classify(sitting_track)
        self.assertEqual(r1.smoothed_state, "STANDING")
        self.assertEqual(r2.smoothed_state, "SITTING")


class TestFallDetector(unittest.TestCase):
    def test_single_frame_horizontal_blip_does_not_trigger_fallen(self):
        fd = FallDetector(fall_threshold_seconds=0.5)
        track = make_track(track_id=20)
        fps = 30.0
        for i in range(10):
            push_frame(track, standing_keypoints(), t=i / fps, frame_idx=i)
            fd.update(track, fps, "STANDING")
        push_frame(track, lying_keypoints(), t=10 / fps, frame_idx=10)
        fd.update(track, fps, "STANDING")
        for i in range(11, 20):
            push_frame(track, standing_keypoints(), t=i / fps, frame_idx=i)
            result = fd.update(track, fps, "STANDING")
        self.assertNotEqual(result.state, STATE_FALLEN)

    def test_sustained_lying_posture_triggers_fallen_event(self):
        fd = FallDetector(fall_threshold_seconds=0.3)
        track = make_track(track_id=21)
        fps = 30.0
        frame_idx = 0
        for i in range(10):
            push_frame(track, standing_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
            fd.update(track, fps, "STANDING")
            frame_idx += 1

        event_seen = False
        for i in range(30):
            push_frame(track, lying_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
            result = fd.update(track, fps, "STANDING")
            if result.event is not None:
                event_seen = True
                self.assertEqual(result.event.event, "FALLEN")
                self.assertEqual(result.event.previous_state, "STANDING")
            frame_idx += 1

        self.assertTrue(event_seen)
        self.assertEqual(result.state, STATE_FALLEN)

    def test_fall_event_not_repeated_every_frame(self):
        fd = FallDetector(fall_threshold_seconds=0.3)
        track = make_track(track_id=22)
        fps = 30.0
        frame_idx = 0
        for i in range(10):
            push_frame(track, standing_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
            fd.update(track, fps, "STANDING")
            frame_idx += 1

        events = []
        for i in range(60):  # 2 seconds of sustained lying posture
            push_frame(track, lying_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
            result = fd.update(track, fps, "STANDING")
            if result.event is not None:
                events.append(result.event)
            frame_idx += 1

        self.assertEqual(len(events), 1, "Exactly one FALLEN event should be emitted, not one per frame")


    def test_missing_torso_angle_does_not_dilute_fall_score(self):
        fps = 25.0
        n = 13
        timestamps = [i / fps for i in range(n)]

        def kp_with_nose_only(nose_y):
            kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
            kp[KPT["nose"]] = [100.0, nose_y, 0.9]
            return kp

        keypoints_history = [kp_with_nose_only(40.0 + i * 30.0) for i in range(n)]
        center_history = [(100.0, 40.0 + i * 30.0) for i in range(n)]
        angle_history = [None] * n
        bbox = np.array([0.0, 300.0, 220.0, 380.0], dtype=np.float32)
        bbox_diagonal = float(np.hypot(220.0, 80.0))

        score, details = compute_fall_score(
            keypoints=keypoints_history[-1],
            bbox=bbox,
            center_history=center_history,
            timestamp_history=timestamps,
            angle_history=angle_history,
            keypoints_history=keypoints_history,
            bbox_diagonal=bbox_diagonal,
            reference_height=200.0,
            conf_thresh=0.3,
        )

        self.assertIsNone(details["torso_angle"])
        self.assertAlmostEqual(details["available_weight"], 0.55, places=2)
        self.assertGreater(score, 0.9, "renormalized score should not be diluted by the unavailable angle signals")



    def test_reference_height_ignores_crouching_relabeled_as_standing(self):
        fd = FallDetector(fall_threshold_seconds=0.8)
        track = make_track(track_id=40)
        fps = 25.0
        kp = standing_keypoints()

        for i in range(10):
            push_frame(track, kp, t=i / fps, frame_idx=i)
            fd.update(track, fps, current_activity_state="STANDING", raw_posture="CROUCHING")

        self.assertEqual(
            fd._states[40].reference_height,
            0.0,
            "reference_height must stay unset from CROUCHING frames, even though the display state says STANDING",
        )

        for i in range(10, 20):
            push_frame(track, kp, t=i / fps, frame_idx=i)
            fd.update(track, fps, current_activity_state="STANDING", raw_posture="STANDING")

        self.assertGreater(
            fd._states[40].reference_height,
            0.0,
            "reference_height should be established once raw_posture is genuinely STANDING",
        )

    def test_reference_height_legacy_fallback_without_raw_posture(self):
        fd = FallDetector(fall_threshold_seconds=0.8)
        track = make_track(track_id=41)
        fps = 25.0
        kp = standing_keypoints()
        for i in range(10):
            push_frame(track, kp, t=i / fps, frame_idx=i)
            fd.update(track, fps, "STANDING")
        self.assertGreater(fd._states[41].reference_height, 0.0)


    def test_leaky_bucket_decays_by_half_step_not_full_reset(self):
        fd = FallDetector(fall_threshold_seconds=1.0)
        track = make_track(track_id=42)
        fps = 30.0
        frame_idx = 0
        for _ in range(5):
            push_frame(track, standing_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
            fd.update(track, fps, "STANDING")
            frame_idx += 1

        for _ in range(4):
            push_frame(track, lying_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
            fd.update(track, fps, "STANDING")
            frame_idx += 1

        count_before = fd._states[42].possible_count
        self.assertGreater(count_before, 0)

        push_frame(track, standing_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
        result = fd.update(track, fps, "STANDING")

        self.assertEqual(result.state, STATE_POSSIBLE_FALL, "a single noisy frame should not fully reset persistence")
        self.assertAlmostEqual(fd._states[42].possible_count, count_before - 0.5, places=3)

    def test_occlusion_check_inert_for_normal_track(self):
        fd = FallDetector(fall_threshold_seconds=0.8)
        track = make_track(track_id=51)
        fps = 25.0
        push_frame(track, standing_keypoints(), t=0.0, frame_idx=0)
        fd.update(track, fps, "STANDING")
        track.frames_since_seen = 100
        occ = fd.check_occlusion(track, fps)
        self.assertIsNone(occ, "an ordinary NORMAL-state track leaving frame must not engage the occlusion path")

    def test_occlusion_hold_then_grace_period_exceeded_emits_event(self):
        fd = FallDetector(fall_threshold_seconds=0.3)
        track = make_track(track_id=52)
        fps = 25.0
        frame_idx = 0
        for _ in range(10):
            push_frame(track, standing_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
            fd.update(track, fps, "STANDING")
            frame_idx += 1

        result = None
        for _ in range(15):
            push_frame(track, lying_keypoints(), t=frame_idx / fps, frame_idx=frame_idx)
            result = fd.update(track, fps, "STANDING")
            frame_idx += 1
        self.assertEqual(result.state, STATE_FALLEN)
        missing_frames_needed = int(OCCLUSION_GRACE_SECONDS * fps) + 5
        event_seen = False
        for i in range(1, missing_frames_needed + 1):
            track.frames_since_seen = i
            occ = fd.check_occlusion(track, fps)
            self.assertIsNotNone(occ)
            if i / fps <= OCCLUSION_GRACE_SECONDS:
                self.assertEqual(occ.state, STATE_SUSPECTED_FALLEN_OCCLUDED)
            if occ.event is not None:
                event_seen = True
                self.assertEqual(occ.event.event, "SUSPECTED_FALLEN_OCCLUDED")
                break

        self.assertTrue(event_seen, "an event must fire once the occlusion grace period elapses with no re-detection")


if __name__ == "__main__":
    unittest.main()
