"""
fall_detection.py

Command-line pose-based human activity / fall detection over a video file,
using a custom-trained YOLO26-pose model.

Example (Windows CMD or any shell):

    python fall_detection.py --model models/best.pt --input input/test.mp4 --output output/result.mp4 --conf 0.4 --device 0 --show

Run `python fall_detection.py --help` for the full list of options.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict

_root_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".ultralytics"))
os.environ.setdefault("YOLO_CONFIG_DIR", _root_config)

import cv2

from src.pose_detector import PoseDetector
from src.tracker import TrackManager
from src.activity_classifier import ActivityClassifier, compute_torso_angle, compute_center_speed
from src.fall_detector import FallDetector, FallEvent
from src.visualization import draw_person, draw_frame_header, draw_threshold_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pose-based human activity and fall detection over a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Windows CMD example:\n"
            "  python fall_detection.py --model models\\best.pt --input input\\test.mp4 "
            "--output output\\result.mp4 --conf 0.4 --device 0 --show\n\n"
            "POSIX example:\n"
            "  python fall_detection.py \\\n"
            "      --model models/best.pt \\\n"
            "      --input input/video.mp4 \\\n"
            "      --output output/result.mp4"
        ),
    )
    parser.add_argument(
        "--model",
        default="models/yolo26s-pose.pt",
        help="Path to the trained YOLO pose model (default: models/yolo26s-pose.pt)"
    )
    parser.add_argument("--input", required=True, help="Path to the input video (mp4/avi/mov)")
    parser.add_argument("--output", default="output/result.mp4", help="Path to write the annotated output video")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection/keypoint confidence threshold (default 0.4)")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (default 0.5)")
    parser.add_argument("--device", default="cpu", help="Inference device: 'cpu', '0', '0,1', etc. (default cpu)")
    parser.add_argument("--show", action="store_true", help="Show a live preview window while processing")
    parser.add_argument("--save", dest="save", action="store_true", default=True, help="Save the annotated output video (default: on)")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Disable writing the output video")
    parser.add_argument("--history", type=int, default=30, help="Number of frames of per-person history to retain (default 30)")
    parser.add_argument("--fall-threshold", type=float, default=0.8, help="Seconds a horizontal/falling posture must persist before FALLEN is confirmed (default 0.8)")
    parser.add_argument("--debug", action="store_true", help="Overlay debug measurements (angles, scores, speed, fall_score, transitions)")
    parser.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"], help="Ultralytics tracker config (default botsort.yaml)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640)")
    parser.add_argument(
        "--low-conf",
        type=float,
        default=0.15,
        help="Secondary confidence floor used only to re-acquire a person who was already flagged as a possible/confirmed fall and then went undetected (default 0.15). Never used for ordinary detection.",
    )
    return parser.parse_args()


class EventLogger:

    def __init__(self, csv_path: str, json_path: str):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self.csv_path = csv_path
        self.json_path = json_path
        self.events = []
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["timestamp", "person_id", "event", "confidence", "previous_state", "current_state"])

    def log(self, event: FallEvent) -> None:
        self._writer.writerow(
            [f"{event.timestamp:.2f}", event.track_id, event.event, f"{event.confidence:.2f}", event.previous_state, event.current_state]
        )
        self._csv_file.flush()
        self.events.append(asdict(event))
        print(
            f"[FALL EVENT] t={event.timestamp:.2f}s person={event.track_id} "
            f"conf={event.confidence:.2f} ({event.previous_state} -> {event.current_state})"
        )

    def close(self) -> None:
        self._csv_file.close()
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)


def resolve_fourcc(output_path: str) -> str:
    ext = os.path.splitext(output_path)[1].lower()
    return {".avi": "XVID", ".mp4": "mp4v", ".mov": "mp4v"}.get(ext, "mp4v")


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.model):
        alt_paths = [
            os.path.join(os.path.dirname(__file__), args.model),
            os.path.join(os.path.dirname(__file__), "..", "models", os.path.basename(args.model)),
            os.path.join(os.path.dirname(__file__), "models", "yolo26s-pose.pt"),
            os.path.join(os.path.dirname(__file__), "..", "models", "yolo26s-pose.pt"),
            os.path.join(os.path.dirname(__file__), "..", "models", "yolov8m-pose.pt"),
        ]
        found = False
        for p in alt_paths:
            if os.path.isfile(p):
                args.model = p
                found = True
                break
        if not found:
            print(f"ERROR: model file not found: {args.model}")
            return 1
    if not os.path.isfile(args.input):
        print(f"ERROR: input video not found: {args.input}")
        return 1

    print(f"[fall_detection] Loading model: {args.model}")
    detector = PoseDetector(
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        tracker=args.tracker,
    )
    print(f"[fall_detection] Model task={detector.model.task} classes={detector.class_names}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: could not open input video: {args.input}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("[fall_detection] WARNING: video reports invalid FPS, assuming 25.0")
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0.0
    print(f"[fall_detection] Video: {width}x{height} @ {fps:.2f} FPS, ~{total_frames} frames (~{duration:.1f}s)")

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*resolve_fourcc(args.output))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"ERROR: could not open output video for writing: {args.output}")
            return 1

    out_base = os.path.splitext(args.output)[0]
    logger = EventLogger(csv_path=out_base + "_events.csv", json_path=out_base + "_events.json")

    min_consecutive_frames = max(3, round(0.3 * fps))
    track_manager = TrackManager(
        history_len=args.history,
        max_missing_frames=max(5, round(fps * 2)),
        fps=fps,
    )
    activity_classifier = ActivityClassifier(
        conf_thresh=args.conf,
        min_consecutive_frames=min_consecutive_frames,
        smoothing_window=args.history,
    )
    fall_detector = FallDetector(conf_thresh=args.conf, fall_threshold_seconds=args.fall_threshold)

    frame_idx = 0
    start_time = time.time()
    prev_ids = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            detections = detector.infer(frame, use_tracking=True)
            torso_angles = [compute_torso_angle(d.keypoints, args.conf) for d in detections]

            tracks = track_manager.update(
                detections, frame_idx, timestamp, detector.tracking_available, torso_angles
            )

            fall_relevant_states = ("POSSIBLE_FALL", "FALLEN", "SUSPECTED_FALLEN_OCCLUDED")
            missing_fall_relevant_ids = [
                tid
                for tid, tr in tracks.items()
                if tr.frames_since_seen > 0 and tr.fall_state in fall_relevant_states
            ]
            if missing_fall_relevant_ids:
                low_conf_detections = detector.infer_low_conf(frame, args.low_conf)
                track_manager.try_attach_low_conf(
                    missing_fall_relevant_ids, low_conf_detections, frame_idx, timestamp, conf_thresh=args.conf
                )

            current_ids = set(tracks.keys())
            for removed_id in prev_ids - current_ids:
                activity_classifier.remove_track(removed_id)
                fall_detector.remove_track(removed_id)
            prev_ids = current_ids

            det_by_id = {d.track_id: d for d in detections if d.track_id is not None}
            per_person_debug_lines = []

            for tid, track in tracks.items():
                if track.frames_since_seen > 0:
                    occlusion_result = fall_detector.check_occlusion(track, fps)
                    if occlusion_result is not None:
                        track.fall_score = occlusion_result.fall_score
                        track.fall_state = occlusion_result.state
                        if occlusion_result.transition:
                            track.last_transition = occlusion_result.transition
                            print(
                                f"[fall_detection] t={timestamp:.2f}s person={tid}: {occlusion_result.transition}"
                            )
                        if occlusion_result.event:
                            logger.log(occlusion_result.event)
                    continue

                cls_result = activity_classifier.classify(track)
                track.current_state = cls_result.smoothed_state
                track.state_confidence = cls_result.confidence
                track.consecutive_state_frames = cls_result.consecutive_frames

                fall_result = fall_detector.update(
                    track, fps, track.current_state, raw_posture=cls_result.geometric_posture
                )
                track.fall_score = fall_result.fall_score
                track.fall_state = fall_result.state
                if fall_result.transition:
                    track.last_transition = fall_result.transition

                if fall_result.event:
                    logger.log(fall_result.event)

                display_state = FallDetector.display_state(fall_result.state) or track.current_state

                det = det_by_id.get(tid)
                bbox_conf = det.bbox_conf if det is not None else 0.0

                debug_info = None
                if args.debug:
                    speed = 0.0
                    if len(track.center_history) >= 2:
                        speed = compute_center_speed(
                            list(track.center_history), list(track.timestamp_history), track.bbox_diagonal()
                        )
                    debug_info = {
                        "torso_angle": fall_result.details.get("torso_angle"),
                        "bbox_aspect": fall_result.details.get("bbox_aspect"),
                        "speed": speed,
                        "fall_score": fall_result.fall_score,
                        "fall_state": fall_result.state,
                        "transition": track.last_transition,
                    }
                    angle_str = (
                        f"{fall_result.details.get('torso_angle'):.0f}"
                        if fall_result.details.get("torso_angle") is not None
                        else "N/A"
                    )
                    ref_h = fall_result.details.get("reference_height", 0.0)
                    cur_h = fall_result.details.get("current_height", 0.0)
                    height_ratio = (cur_h / ref_h) if ref_h and ref_h > 0 else 1.0
                    per_person_debug_lines.append(
                        f"P{tid} [{display_state}] angle={angle_str} aspect={fall_result.details.get('bbox_aspect', 0):.2f} "
                        f"height_ratio={height_ratio:.2f} speed={speed:.2f} fall={fall_result.fall_score:.2f}/{fall_result.state}"
                    )

                draw_person(
                    frame,
                    track_id=tid,
                    bbox=track.latest_bbox,
                    keypoints=track.latest_keypoints,
                    display_state=display_state,
                    state_confidence=track.state_confidence,
                    bbox_conf=bbox_conf,
                    conf_thresh=args.conf,
                    debug=args.debug,
                    debug_info=debug_info,
                )


            if args.debug:
                tracking_mode = "native" if detector.tracking_available else "fallback-iou"
                header_bottom_y = draw_frame_header(frame, frame_idx, fps, len(tracks), tracking_mode)
                draw_threshold_panel(frame, header_bottom_y, per_person_debug_lines)

            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("fall_detection", frame)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    print("[fall_detection] 'q' pressed, stopping early.")
                    break

            frame_idx += 1
            if frame_idx % max(1, round(fps)) == 0:
                elapsed = time.time() - start_time
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                progress = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
                print(f"[fall_detection] frame {progress}  ({proc_fps:.1f} fps processing)", end="\r")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.close()

    elapsed = time.time() - start_time
    print()
    print(f"[fall_detection] Done. Processed {frame_idx} frames in {elapsed:.1f}s.")
    if args.save:
        print(f"[fall_detection] Annotated video: {args.output}")
    print(f"[fall_detection] Fall events: {logger.csv_path} / {logger.json_path} ({len(logger.events)} event(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
