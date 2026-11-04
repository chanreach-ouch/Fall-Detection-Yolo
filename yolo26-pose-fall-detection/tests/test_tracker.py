"""
Unit tests for src/tracker.py, focused on the spatial-proximity re-linking
layer that keeps a person's identity stable across a raw tracker ID switch
-- the failure mode observed on real low-FPS CCTV footage where a person
can move a large distance between consecutive 1-second-apart frames,
breaking IoU/motion-based association in the underlying tracker.

Run from the project root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import unittest

import numpy as np

from src.pose_detector import Detection, NUM_KEYPOINTS
from src.tracker import TrackManager


def make_det(track_id, cx, cy, w=60, h=180):
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
    return Detection(track_id=track_id, bbox=bbox, bbox_conf=0.9, keypoints=kp)


class TestTrackManagerRelinking(unittest.TestCase):
    def test_stable_id_persists_across_raw_id_switch_nearby(self):
        # fps=1 with a generous relink window, mimicking the real low-FPS
        # CCTV footage this was tuned against.
        tm = TrackManager(history_len=10, max_missing_frames=5, fps=1.0, relink_max_gap_seconds=3.0)

        tm.update([make_det(101, 100, 100)], frame_idx=0, timestamp=0.0, native_tracking_available=True)
        first_stable_id = list(tm.active_tracks().keys())[0]

        # Detection missing for one frame (person briefly not detected).
        tm.update([], frame_idx=1, timestamp=1.0, native_tracking_available=True)

        # Reappears nearby, but the underlying tracker assigns a brand new
        # raw ID (simulating an ID switch) -- should re-link to the same
        # stable ID rather than creating a new person.
        tm.update([make_det(202, 130, 110)], frame_idx=2, timestamp=2.0, native_tracking_available=True)
        second_stable_id = list(tm.active_tracks().keys())[0]

        self.assertEqual(first_stable_id, second_stable_id)
        self.assertEqual(len(tm.tracks), 1)

    def test_far_away_new_detection_gets_new_id(self):
        tm = TrackManager(history_len=10, max_missing_frames=5, fps=1.0, relink_max_gap_seconds=3.0)
        tm.update([make_det(101, 100, 100)], frame_idx=0, timestamp=0.0, native_tracking_available=True)
        first_id = list(tm.active_tracks().keys())[0]

        tm.update([], frame_idx=1, timestamp=1.0, native_tracking_available=True)
        # Reappears far away (e.g. a different person entering) -> new ID.
        tm.update([make_det(202, 900, 900)], frame_idx=2, timestamp=2.0, native_tracking_available=True)
        second_id = list(tm.active_tracks().keys())[0]

        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(tm.tracks), 2)

    def test_two_people_do_not_collide_onto_the_same_stable_id(self):
        tm = TrackManager(history_len=10, max_missing_frames=5, fps=1.0, relink_max_gap_seconds=3.0)
        tm.update(
            [make_det(1, 100, 100), make_det(2, 400, 100)],
            frame_idx=0,
            timestamp=0.0,
            native_tracking_available=True,
        )
        ids_frame0 = set(tm.active_tracks().keys())
        self.assertEqual(len(ids_frame0), 2)

        tm.update([], frame_idx=1, timestamp=1.0, native_tracking_available=True)

        # Both reappear nearby their old spots with new raw IDs in the same
        # frame -- each should re-link to its own prior identity, not both
        # collapsing onto one.
        tm.update(
            [make_det(11, 110, 105), make_det(12, 410, 105)],
            frame_idx=2,
            timestamp=2.0,
            native_tracking_available=True,
        )
        ids_frame2 = set(tm.active_tracks().keys())
        self.assertEqual(len(ids_frame2), 2)
        self.assertEqual(ids_frame0, ids_frame2)


class TestDuplicateDetectionSuppression(unittest.TestCase):
    """Regression tests for the same-frame IoU duplicate-suppression pass,
    added after real footage (video_5) showed the model -- an NMS-free
    end-to-end architecture -- occasionally emitting two near-identical
    overlapping boxes for one real person in a single frame (observed
    IoU ~0.95), which the tracker previously handed two separate track
    IDs, each independently able to accumulate its own fall state machine
    for what was really one fall."""

    def test_two_heavily_overlapping_detections_become_one_track(self):
        tm = TrackManager(history_len=10, max_missing_frames=5, fps=25.0)
        # Two near-duplicate boxes for the same person, different raw IDs
        # (as native tracking would assign), one slightly lower confidence.
        det_a = make_det(101, 150, 200, w=80, h=180)
        det_a.bbox_conf = 0.88
        det_b = make_det(202, 151, 201, w=80, h=180)
        det_b.bbox_conf = 0.67

        tracks = tm.update([det_a, det_b], frame_idx=0, timestamp=0.0, native_tracking_available=True)
        self.assertEqual(len(tracks), 1, "two heavily-overlapping boxes for one person must not become two tracks")

    def test_torso_angles_stay_aligned_after_duplicate_suppression(self):
        # The surviving (higher-confidence) detection's torso angle must be
        # the one actually recorded -- not the suppressed detection's,
        # and not misaligned by index after the list is filtered.
        tm = TrackManager(history_len=10, max_missing_frames=5, fps=25.0)
        det_a = make_det(101, 150, 200, w=80, h=180)
        det_a.bbox_conf = 0.88
        det_b = make_det(202, 151, 201, w=80, h=180)
        det_b.bbox_conf = 0.67

        tracks = tm.update(
            [det_a, det_b], frame_idx=0, timestamp=0.0, native_tracking_available=True, torso_angles=[12.5, 99.0]
        )
        (track,) = tracks.values()
        self.assertEqual(track.angle_history[-1], 12.5, "kept detection's own torso angle must be recorded, not the suppressed one's")

    def test_two_distinct_nearby_but_non_overlapping_people_both_survive(self):
        tm = TrackManager(history_len=10, max_missing_frames=5, fps=25.0)
        det_a = make_det(101, 100, 200, w=60, h=180)
        det_b = make_det(202, 400, 200, w=60, h=180)  # far apart, no overlap

        tracks = tm.update([det_a, det_b], frame_idx=0, timestamp=0.0, native_tracking_available=True)
        self.assertEqual(len(tracks), 2, "distinct, non-overlapping people must not be merged")


class TestTrackManagerRelinkingWindow(unittest.TestCase):
    def test_gap_longer_than_relink_window_gets_new_id(self):
        tm = TrackManager(history_len=10, max_missing_frames=10, fps=1.0, relink_max_gap_seconds=2.0)
        tm.update([make_det(101, 100, 100)], frame_idx=0, timestamp=0.0, native_tracking_available=True)
        first_id = list(tm.active_tracks().keys())[0]

        for i in range(1, 5):  # 4 seconds of absence, beyond the 2s relink window
            tm.update([], frame_idx=i, timestamp=float(i), native_tracking_available=True)

        tm.update([make_det(202, 105, 102)], frame_idx=5, timestamp=5.0, native_tracking_available=True)
        second_id = list(tm.active_tracks().keys())[0]
        self.assertNotEqual(first_id, second_id)


if __name__ == "__main__":
    unittest.main()
