"""
coco_keypoints_to_yolo_pose.py

Converts a CVAT "COCO Keypoints 1.0" export into Ultralytics YOLO-pose
label format.

Why this exists: CVAT does not export directly to YOLO-pose `.txt` files
for skeletons -- its native pose-adjacent export is COCO Keypoints JSON
(`person_keypoints_*.json`). This script does that last conversion step.

CVAT visibility flags -> COCO/YOLO visibility flags (all three tools agree
on this 0/1/2 convention, so no remapping is needed beyond parsing):
    0 = not labeled at all
    1 = labeled, but occluded / not visible
    2 = labeled and visible

YOLO-pose label line format (one line per person, values space-separated,
all coordinates normalized to [0, 1] by image width/height):

    class x_center y_center width height x1 y1 v1 x2 y2 v2 ... x17 y17 v17

Usage:
    python tools/coco_keypoints_to_yolo_pose.py \
        --coco-json path/to/person_keypoints_default.json \
        --images-dir path/to/exported/images \
        --out-dir dataset/labels/train

Then point your data.yaml at the matching images/ and labels/ directories
(see docs/ANNOTATION_GUIDE.md and docs/TRAINING_GUIDE.md for the full
dataset layout and a data.yaml example).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def convert(coco_json_path: str, images_dir: str, out_dir: str) -> None:
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    CLASS_ID = 0

    os.makedirs(out_dir, exist_ok=True)
    written, skipped_no_ann = 0, 0

    for image_id, img in images_by_id.items():
        file_name = img["file_name"]
        img_w, img_h = img["width"], img["height"]
        stem = os.path.splitext(os.path.basename(file_name))[0]
        label_path = os.path.join(out_dir, f"{stem}.txt")

        anns = anns_by_image.get(image_id, [])
        if not anns:
            skipped_no_ann += 1
            open(label_path, "w", encoding="utf-8").close()
            continue

        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO bbox: top-left x, y, width, height
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h

            kpts = ann["keypoints"]  # flat list: x1,y1,v1, x2,y2,v2, ...
            kpt_fields = []
            for i in range(0, len(kpts), 3):
                kx, ky, kv = kpts[i], kpts[i + 1], kpts[i + 2]
                nx = kx / img_w
                ny = ky / img_h
                kpt_fields.extend([f"{nx:.6f}", f"{ny:.6f}", str(int(kv))])

            fields = [str(CLASS_ID), f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}"] + kpt_fields
            lines.append(" ".join(fields))

        with open(label_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(lines) + "\n")
        written += 1

    print(f"Wrote {written} label file(s) to {out_dir}")
    if skipped_no_ann:
        print(f"({skipped_no_ann} image(s) had no annotations -- wrote empty label files for them)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-json", required=True, help="Path to CVAT's exported person_keypoints_*.json")
    parser.add_argument("--images-dir", required=True, help="Path to the matching exported images directory (used only to sanity-check filenames)")
    parser.add_argument("--out-dir", required=True, help="Directory to write YOLO-pose .txt label files into")
    args = parser.parse_args()
    convert(args.coco_json, args.images_dir, args.out_dir)


if __name__ == "__main__":
    main()
