# Data Annotation Guide (CVAT → YOLO-Pose)

How to annotate new fall/activity footage for this project's keypoint
layout, using [CVAT](https://www.cvat.ai/) (open-source, self-hostable or
[cvat.ai](https://app.cvat.ai) free tier), and convert the export into the
format `models/yolo26s-pose.pt` was trained on.

This project's model uses the **standard COCO-17 keypoint layout**
(verified empirically, not assumed -- see the main README section 1):

```
0 nose            6 right_shoulder   12 right_hip
1 left_eye        7 left_elbow       13 left_knee
2 right_eye       8 right_elbow      14 right_knee
3 left_ear        9 left_wrist       15 left_ankle
4 right_ear      10 right_wrist      16 right_ankle
5 left_shoulder  11 left_hip
```

If you're annotating for a *different* model with a different keypoint
set, re-derive the order first (README section 15) before following this
guide with your own layout.

## 1. Collect and extract frames

Annotating full video in CVAT is supported (CVAT can ingest video
directly and interpolate between annotated keyframes), but for a pose
dataset you typically get better label quality annotating a curated set
of **still frames** sampled from your videos, especially around the
moments that matter (walking, sitting down, the fall itself, lying on the
ground).

```bash
# Example: pull 1 frame every 0.5s from a source video into a folder of
# JPEGs ready for CVAT upload.
ffmpeg -i input/my_fall_video.mp4 -vf fps=2 dataset/raw_frames/my_fall_video_%05d.jpg
```

Aim for a mix across all target classes for this project (STANDING,
SITTING, WALKING, FALLING, FALLEN) and across different people, clothing,
lighting, and camera angles -- the two real misses documented in the main
README were both a single hard camera geometry (falling away from the
camera), which more varied training data would help address.

## 2. Create a CVAT project with a skeleton (pose) template

1. In CVAT, create a new **Project**, then a **Task** for your image set
   (upload the frames from step 1).
2. Under the project's **Constructor**, choose **Skeleton** as the label
   type (not "Points" -- Skeleton keeps the 17 points grouped as one
   annotatable person with defined edges, which is what maps cleanly to
   YOLO-pose).
3. Build the skeleton template with exactly 17 points, named to match the
   table above (`nose`, `left_eye`, `right_eye`, ..., `right_ankle`), in
   that exact order -- CVAT preserves point order within a skeleton label,
   and that order must match `KEYPOINT_NAMES` in `src/pose_detector.py`.
4. Add edges between points for visual sanity-checking during annotation
   (not required for the YOLO export, but makes mis-clicked points obvious
   at a glance). Use the same pairs as `SKELETON` in
   `src/pose_detector.py`:
   ```
   (15,13) (13,11) (16,14) (14,12) (11,12)
   (5,11)  (6,12)  (5,6)   (5,7)   (6,8)  (7,9)  (8,10)
   (1,2)   (0,1)   (0,2)   (1,3)   (2,4)  (3,5)  (4,6)
   ```
   (0-indexed here; CVAT's constructor UI lets you draw these as lines
   between points directly.)

## 3. Annotate, using visibility flags correctly

For each person in each frame:

1. Draw the skeleton (CVAT auto-creates a bounding box around it).
2. Place all 17 points at their correct image location, **even for points
   that are occluded** -- estimate where the joint actually is (e.g. a
   hip hidden behind a table), don't skip it.
3. Set each point's **visibility/occluded state** correctly. CVAT exposes
   this per-point (right-click a point, or use the point's context menu /
   keyboard shortcut for "occluded"):

   | Situation | CVAT state | Exported COCO/YOLO `v` value |
   |---|---|---|
   | Joint clearly visible in the image | visible (default) | `2` |
   | Joint's real position is known/estimated but hidden behind something (furniture, another limb, out of frame edge) | **occluded** | `1` |
   | You genuinely cannot tell where the joint is at all | delete the point / leave unlabeled | `0` |

   Getting this right matters a lot for training quality: this project's
   `activity_classifier.py` and `fall_detector.py` both treat a
   keypoint's confidence/visibility as real evidence (e.g. "knee_angle is
   None, don't guess STANDING") -- a model trained
   on carelessly-all-visible labels learns to be falsely confident on
   occluded joints, which is exactly the failure mode documented in this
   project's own real-footage validation (README section 7: falls away
   from the camera, where the torso is genuinely foreshortened/occluded).
4. For a genuinely absent/out-of-frame person (only part of them visible),
   annotate only the visible/inferable points and mark the rest occluded
   or unlabeled per the rules above -- don't guess wildly.
5. Label the **activity/state** as an attribute if you plan to also train
   or evaluate an activity classifier from annotations directly (this
   project's own `activity_classifier.py` derives state from geometry
   rather than a learned label, but recording it is useful for a held-out
   evaluation set). Add a task-level label attribute, e.g. `activity` with
   values `standing / sitting / walking / falling / fallen`, and set it
   per skeleton instance.

### Multi-person frames

Annotate every person in frame as a separate skeleton instance. CVAT and
the exporter below both handle multiple people per image natively.

## 4. Review pass

Before exporting: spot-check a sample of frames per annotator (if more
than one person annotated), especially:
- Point order consistency (a shoulder point placed on a hip, etc.)
- Visibility flags used consistently (not everyone defaulting to
  "visible" out of laziness -- this silently degrades exactly the
  low-confidence-keypoint handling this project relies on)
- Bounding boxes tight around the actual person, not the skeleton's
  convex hull with extra padding

## 5. Export from CVAT

In the CVAT task/project, **Actions → Export dataset**, format:

**`COCO Keypoints 1.0`**

This produces a `.zip` containing `images/` and an
`annotations/person_keypoints_default.json` (COCO-format keypoints JSON).
This is the standard interchange format CVAT provides for skeleton/pose
annotations -- Ultralytics YOLO-pose isn't a CVAT-native export option, so
the conversion below is a required last step regardless of your tooling.

## 6. Convert to YOLO-pose format

Use the converter script included in this repo:

```bash
python tools/coco_keypoints_to_yolo_pose.py \
    --coco-json exported/annotations/person_keypoints_default.json \
    --images-dir exported/images \
    --out-dir dataset/labels/train
```

Run it once per split (train/val/test) with the corresponding CVAT
export. This writes one `.txt` label file per image, in the format
Ultralytics expects:

```
0 0.512 0.634 0.183 0.421 0.501 0.288 2 0.522 0.271 2 ... (17 keypoints total)
```
`class x_center y_center width height  x1 y1 v1  x2 y2 v2 ... x17 y17 v17`
(all coordinates normalized 0-1 by image width/height).

## 7. Final dataset layout

```
dataset/
├── images/
│   ├── train/   *.jpg
│   └── val/     *.jpg
├── labels/
│   ├── train/   *.txt   (from the converter, matching images/train filenames)
│   └── val/     *.txt
└── data.yaml
```

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for the `data.yaml`
contents and how to kick off training/fine-tuning from here.
