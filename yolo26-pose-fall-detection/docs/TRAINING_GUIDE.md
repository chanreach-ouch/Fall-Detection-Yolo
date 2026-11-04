# Training Guide (YOLO26-Pose)

How to train a new model or fine-tune `models/Yolo26s-pose.pt` further, once you
have a YOLO-pose dataset (see **[ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)**
for producing one from CVAT).

## 1. What `models/Yolo26s-pose.pt` was actually trained with

Rather than guessing reasonable defaults, these hyperparameters were
**recovered directly from the checkpoint's own embedded `train_args`**
(`torch.load("models/Yolo26s-pose.pt")["train_args"]`) -- so this section
documents what was actually done, not a generic recipe:

| Setting | Value |
|---|---|
| Base/pretrained weights | `yolo26s-pose.pt` (Ultralytics COCO-pretrained) |
| Ultralytics version used | `8.4.21` |
| Epochs | `300` (with `patience=100` early stopping) |
| Image size | `640` |
| Batch size | `8` |
| Optimizer | `AdamW`, `lr0=0.0005`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005` |
| LR schedule | Cosine (`cos_lr=True`), 3-epoch warmup |
| Loss weights | `box=7.5 cls=0.5 dfl=1.5 pose=12.0 kobj=1.0` |
| Mosaic | `mosaic=1.0`, disabled for the last 10 epochs (`close_mosaic=10`) |
| Other augmentation | `hsv_h=0.015 hsv_s=0.7 hsv_v=0.4`, `translate=0.1 scale=0.5`, `mixup=0.1`, `copy_paste=0.1` (flip mode), `auto_augment=randaugment`, `erasing=0.4` |
| Horizontal/vertical flip | **both disabled** (`fliplr=0.0 flipud=0.0`) |
| Rotation/shear/perspective | all `0.0` (disabled) |
| Mixed precision | `amp=True` |
| Deterministic | `seed=0, deterministic=True` |
| Tracker (for the `track()` API, not training) | `botsort.yaml` |

**Note on `fliplr=0.0`:** horizontal-flip augmentation was off for this
run. If you retrain from scratch and want to enable it (usually helps
pose-model generalization), you must also set `flip_idx` in `data.yaml`
(see below) so left/right keypoints get swapped correctly when a training
image is mirrored -- otherwise a flipped image silently trains the model
with left/right keypoints on the wrong side.

## 2. Environment

Same environment as the main app (see the root README's Installation
section) plus nothing extra -- `ultralytics` already includes the training
CLI. Training benefits a lot from a CUDA GPU; the hyperparameters above
(`device='0'`, `batch=8`, `amp=True`) assume one.

```cmd
pip install -r requirements.txt
```

## 3. `data.yaml`

```yaml
# dataset/data.yaml
path: dataset          # root, relative to this file or absolute
train: images/train
val: images/val

kpt_shape: [17, 3]      # 17 keypoints, each (x, y, visibility)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
```

`flip_idx` maps each keypoint index to its mirror -- e.g. index 1
(`left_eye`) maps to index 2 (`right_eye`) and vice versa; index 0
(`nose`) maps to itself. This list is required if you enable `fliplr` (see
above); harmless to include even if you don't.

## 4. Train (or fine-tune)

**From scratch / from the official pretrained pose weights:**

```cmd
yolo pose train \
    model=yolo26s-pose.pt \
    data=dataset\data.yaml \
    epochs=300 \
    imgsz=640 \
    batch=8 \
    optimizer=AdamW \
    lr0=0.0005 \
    cos_lr=True \
    close_mosaic=10 \
    device=0
```


**Fine-tuning `models/best.pt` further** with newly-annotated data
(recommended over training from scratch once you already have a working
model): start from the existing checkpoint, use a smaller learning rate,
and fewer epochs so you don't overwrite what it already learned:

```cmd
yolo pose train \
    model=models\best.pt \
    data=dataset\data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    optimizer=AdamW \
    lr0=0.0001 \
    device=0
```

Both commands write to `runs/pose/train*/weights/best.pt` -- copy that
into this project's `models/` directory (or point `--model` at it
directly) when you're ready to use it for inference.

## 5. Validate

```cmd
yolo pose val model=runs\pose\train\weights\best.pt data=dataset\data.yaml
```

Reports standard pose metrics (box mAP, pose mAP/OKS) on the `val` split.
Also sanity-check visually:

```cmd
yolo pose predict model=runs\pose\train\weights\best.pt source=dataset\images\val save=True
```
