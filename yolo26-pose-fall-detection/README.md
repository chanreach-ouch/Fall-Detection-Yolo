# Fall Detection & Human Activity Recognition (YOLO26-Pose)

Pose-based human activity recognition and fall detection over video, built
on a **custom-trained YOLO26-pose model**. Detects and tracks people across
frames and classifies each one as **STANDING**, **SITTING**, **WALKING**,
**FALLING**, or **FALLEN**, with temporal smoothing so the state doesn't
flicker between frames.

## 1. About the model (inspected, not assumed)

Before writing any code, `models/Yolo26s-pose.pt` was loaded and inspected directly:

| Property | Value |
|---|---|
| Architecture | YOLO26s-pose (`yolo26s-pose.yaml`), end-to-end / NMS-free |
| Task | `pose` |
| Classes | `{0: 'person'}` (single class) |
| Keypoint shape | `[17, 3]` (17 keypoints, x/y/visibility) |
| Trained with | Ultralytics `8.4.21`, fine-tuned from Ultralytics' pretrained `yolo26s-pose.pt` |
| Tracker (from train_args) | `botsort.yaml` |

The 17 keypoints were **not assumed to be COCO** -- this was verified by
running the actual model on a real photo of upright people and checking
that the 17 output points are monotonically ordered head-to-feet with
correct left/right pairing. They matched the standard COCO-pose layout
exactly:

```
0 nose            6 right_shoulder   12 right_hip
1 left_eye        7 left_elbow       13 left_knee
2 right_eye       8 right_elbow      14 right_knee
3 left_ear        9 left_wrist       15 left_ankle
4 right_ear      10 right_wrist      16 right_ankle
5 left_shoulder  11 left_hip
```

This layout lives in one place, `src/pose_detector.py` (`KEYPOINT_NAMES`,
`KPT`, `SKELETON`), so if you retrain with a different keypoint set later,
you only need to update it there.

If you swap in a different model with a different `kpt_shape` or keypoint
order, re-run the inspection (see "Inspecting a new model" below) and
update `src/pose_detector.py` accordingly -- nothing here is hardcoded to
COCO indices, everything reads from that one table.

## 2. Project structure

```
project/
├── models/                    # put your .pt model here
│   └── yolo26s-pose.pt
├── input/                     # put your input videos here
├── output/                    # annotated videos + event logs are written here
├── src/
│   ├── pose_detector.py       # Ultralytics model wrapper, keypoint layout
│   ├── tracker.py             # PersonTrack history + ID tracking (native + IoU fallback + dedup)
│   ├── activity_classifier.py # STANDING / SITTING / WALKING geometry + smoothing
│   ├── fall_detector.py       # temporal FALLING / FALLEN state machine
│   └── visualization.py       # drawing: bbox, skeleton, labels, debug overlay
├── tests/
│   ├── test_activity_classifier.py
│   └── test_tracker.py
├── tools/
│   └── coco_keypoints_to_yolo_pose.py  # CVAT export -> YOLO-pose label converter
├── docs/
│   ├── ANNOTATION_GUIDE.md    # CVAT annotation workflow (visibility flags, export, convert)
│   └── TRAINING_GUIDE.md      # training/fine-tuning this project's model
├── fall_detection.py          # CLI entrypoint
├── requirements.txt
└── README.md
```

## 3. Installation

**Recommended Python version: 3.10 - 3.13** (tested on 3.13.14).


```bash
cd path/to/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Required packages

See `requirements.txt`: `ultralytics`, `opencv-python`, `numpy`, `torch`,
`torchvision`, `lap` (the last one is required for Ultralytics' BoT-SORT /
ByteTrack tracking; `pip install ultralytics` will auto-install it on first
use if missing, but it's pinned explicitly here for reproducibility).

## 4. Placing your model and video

1. Copy your trained pose model to `models/yolo26s-pose.pt` (or any path/name --
   just point `--model` at it).
2. Copy your input video to `input/your_video.mp4`.
3. Output goes to `output/` by default (created automatically).

## 5. Running inference


```bash
python fall_detection.py \
    --model models/yolo26s-pose.pt \
    --input input/video.mp4 \
    --output output/result.mp4 \
    --conf 0.4 \
    --device 0 \
    --show
```

### CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--model` | *(required)* | Path to the trained YOLO pose model |
| `--input` | *(required)* | Path to input video (mp4 / avi / mov) |
| `--output` | `output/result.mp4` | Path to write the annotated output video |
| `--conf` | `0.4` | Detection / keypoint confidence threshold |
| `--iou` | `0.5` | NMS IoU threshold. **Note:** this model is NMS-free/end-to-end (§1); this flag may have little or no effect on this checkpoint's inference path. Duplicate-box suppression for this model is handled separately (§12), not via `--iou`. Verify empirically before relying on this flag to change duplicate-detection behavior.
| `--device` | `cpu` | `cpu`, `0`, `0,1`, ... (falls back to CPU with a warning if CUDA isn't available) |
| `--show` | off | Show a live preview window (press `q` to stop early) |
| `--save` / `--no-save` | save on | Whether to write the annotated video file |
| `--history` | `30` | Frames of per-person history retained for smoothing/motion analysis |
| `--fall-threshold` | `0.8` | Seconds a horizontal/falling posture must persist before FALLEN is confirmed |
| `--debug` | off | Overlay torso angle, aspect ratio, speed, fall score, state confidence, transitions |
| `--tracker` | `botsort.yaml` | `botsort.yaml` or `bytetrack.yaml` |
| `--imgsz` | `640` | Inference image size |
| `--low-conf` | `0.15` | Secondary confidence floor used only to re-acquire a person already flagged as a possible/confirmed fall who then went undetected. **TODO: the "Occlusion handling" section this originally pointed to is missing from this doc — write it, or verify this flag is actually wired into the reacquisition path described in this row before publishing.** Never used for ordinary detection. |

Outputs, per run:
- `output/result.mp4` -- annotated video (bbox, skeleton, ID, state)
- `output/result_events.csv` -- one row per confirmed fall event
- `output/result_events.json` -- same events as JSON

## 6. GPU / CUDA setup

This app runs fine on CPU (the default). To use an NVIDIA GPU:

1. Confirm you have an NVIDIA GPU and a recent driver (`nvidia-smi` should
   work in a terminal).
2. Install a CUDA-enabled torch build matching your CUDA version, e.g.:
   ```cmd
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   (see https://pytorch.org/get-started/locally/ for the exact command for
   your CUDA version).
3. Run with `--device 0` (or `0,1` for multi-GPU).

If `--device 0` is passed but CUDA isn't actually available, the app prints
a warning and automatically falls back to CPU rather than crashing.


## 7. Why bbox height-collapse, and a real camera-angle limitation

Torso angle and aspect ratio are strong signals when the camera views a
fall roughly **side-on**. But when a person falls **away from an
elevated/angled camera** (along its depth axis, e.g. a ceiling-mounted
camera looking down a hallway), the 2D shoulder-hip vector can foreshorten
instead of rotating toward horizontal, and the bbox doesn't necessarily
widen -- both primary signals can read "still upright" for the entire
duration the person is actually lying down. This was found and confirmed
while tuning against real fall footage (see "Validation against real
footage" below).

The height-collapse signal -- a person's current bbox height compared to
an EMA baseline of their own recent standing height (only updated while
confidently `STANDING`/`WALKING`, never while `SITTING`, so an
already-collapsed posture can't drag its own baseline down) -- stays
informative in that camera geometry too, since a person who is no longer
at their usual standing height, and stays that way, is meaningful
regardless of which direction they fell. It is not a complete fix: a
seated person's height ratio is often not far from a lying person's in
this camera geometry, so it's weighted as one contributing signal, not a
standalone trigger.


## 8. Low-FPS video (e.g. CCTV-style ~1 FPS clips)

<!-- NOTE: sections 9-10 are unaccounted for. "Occlusion handling" and the
duplicate-detection-suppression writeup (currently squeezed into §12
"Tracking" below) likely belong here as their own numbered sections.
Renumber once that content is confirmed/written. -->

If `ffprobe`/OpenCV reports a very low source FPS (this was tested against
a real 1 FPS, 320x240 clip), keep two things in mind:

- **`--fall-threshold` is in seconds, converted to a frame count using the
  video's actual FPS** (`required_frames = max(2, round(fall_threshold *
  fps))`), and is always floored at 2 frames -- a fall is never confirmed
  from a single frame, even at 1 FPS. But at 1 FPS the default `0.8`
  rounds down to the floor of 2 frames (2 real seconds), which is a fairly
  thin persistence check. For low-FPS footage, pass a larger
  `--fall-threshold` (e.g. `2.0`-`4.0`) so a real fall still needs several
  actual seconds/samples of horizontal-posture evidence, not just two.
- **Walking/motion-based signals degrade** at very low FPS: there's only
  one position sample per second (or less), so periodic leg motion can't
  really be observed, and the velocity-based fall signals (downward
  velocity, angular velocity, head-drop) get much less history to work
  with inside their 0.5s windows. Posture-based signals (torso angle,
  knee angle, bbox aspect ratio, bbox aspect) remain fully reliable
  regardless of FPS since they don't depend on motion between frames --
  expect the fall score to lean more heavily on those at low FPS.
- Tracker ID stability also gets harder at low FPS, since a person can
  move a large distance between consecutive frames. `TrackManager` (in
  `src/tracker.py`) includes a proximity-based re-linking layer for
  exactly this: if a new track ID appears close to where a recently-lost
  track last was (within a few seconds and a couple of body-diagonals),
  it's treated as the same person rather than a new one. This is on by
  default (`relink_max_gap_seconds=3.0`, `relink_max_distance_ratio=2.0`
  in `TrackManager.__init__`); adjust those constants directly if your
  footage needs a wider or tighter window.


## 11. Changing input/output paths

Input and output paths are entirely CLI-driven, no code changes needed:

```cmd
python fall_detection.py --model models\yolo26s-pose.pt --input "C:\videos\clip1.mp4" --output "D:\results\clip1_out.mp4"
```

Any extension in `{.mp4, .avi, .mov}` is supported for input; output
container/codec is chosen automatically from the `--output` extension
(`mp4`/`mov` -> `mp4v`, `avi` -> `XVID`).

## 12. Tracking

Native Ultralytics tracking (`model.track()`, BoT-SORT by default, or
ByteTrack via `--tracker bytetrack.yaml`) is used when available -- this
requires the `lap` package (in `requirements.txt`; auto-installed by
Ultralytics on first use if missing). If tracking fails to initialize for
any reason, the app automatically falls back to a simple greedy IoU +
centroid tracker (`src/tracker.py: SimpleIoUTracker`) so person IDs stay
stable either way. The frame header shows which mode is active
(`tracker=native` or `tracker=fallback-iou`).

**Duplicate-detection suppression**: this NMS-free end-to-end model
occasionally emits two heavily-overlapping boxes for the same real person
in one frame (observed IoU ~0.95 on real footage). `TrackManager` now
runs a same-frame IoU suppression pass (`_suppress_duplicate_detections`,
threshold 0.6) before assigning track identities, keeping only the
higher-confidence box -- otherwise the tracker could hand the duplicate
its own track ID and, combined with the occlusion-hold/leaky-bucket
robustness above, that phantom track could accumulate its own separate
fall confirmation for a fall already correctly reported once. This is
distinct from the temporal proximity re-linking described above (same
frame vs. across frames) and, at threshold 0.6, is unlikely to merge two
genuinely distinct people unless they are standing very close together;
lower the threshold with caution if you see two real people collapsed
into one track.

## 13. Fall event log

In addition to the annotated video, every run writes:
- `output/<name>_events.csv`
- `output/<name>_events.json`

with columns/fields `timestamp, person_id, event, confidence,
previous_state, current_state`.

## 14. Testing

`tests/test_activity_classifier.py` unit-tests the classification/fall
logic with synthetic keypoints (no model or video needed): standing,
sitting, lying, missing keypoints, multiple people, walking-via-motion, a
stationary person *not* being misread as walking, state-smoothing
debounce, a single-frame horizontal blip *not* triggering FALLEN, a
sustained fall correctly triggering exactly one FALLEN event, and no
repeated events while the person stays down.


Run from the project root:

```cmd
python -m unittest discover -s tests -t . -v
```

## 15. Inspecting a new/different model

If you retrain or swap models, re-verify assumptions instead of trusting
old ones:

```python
from ultralytics import YOLO
m = YOLO("models/model.pt")
print(m.task, m.names, m.model.yaml["kpt_shape"])
```

Then run inference on a real image of a person and check the keypoint
y-coordinates are ordered head-to-feet as expected before trusting
`src/pose_detector.py`'s `KEYPOINT_NAMES`/`SKELETON` for that model.

## 16. Full pipeline: data annotation & training

This repo is the **inference/application layer** -- it consumes an
already-trained YOLO-pose model (`models/yolo26s-pose.pt`). If you want to
collect more data, re-annotate, or retrain/fine-tune that model, see:

- **[docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md)** -- extracting
  frames, setting up a 17-point skeleton template in CVAT, annotating with
  per-keypoint visibility flags, exporting, and converting the export to
  Ultralytics YOLO-pose `.txt` format (includes a ready-to-run converter
  script, `tools/coco_keypoints_to_yolo_pose.py`).
- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** -- setting up
  `data.yaml`, training/fine-tuning with the Ultralytics CLI, and the
  exact hyperparameters `models/yolo26s-pose.pt` was actually trained with
  (recovered from the checkpoint's own embedded `train_args`, not
  guessed).

The high-level pipeline, end to end:

```
Raw video/images
      │
      ▼
Frame extraction  ─────────────────────────►  docs/ANNOTATION_GUIDE.md
      │
      ▼
CVAT annotation (bbox + 17 keypoints, visibility flags)
      │
      ▼
Export (CVAT for COCO Keypoints 1.0)
      │
      ▼
Convert to YOLO-pose format (tools/coco_keypoints_to_yolo_pose.py)
      │
      ▼
Train / fine-tune (yolo26s-pose.pt → best.pt)  ────────►  docs/TRAINING_GUIDE.md
      │
      ▼
models/best.pt
      │
      ▼
This repo: fall_detection.py  (pose → track → classify → fall-detect → annotate video)
      │
      ▼
Annotated video + fall event CSV/JSON
```
