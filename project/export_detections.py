"""
Convert YOLO detection outputs (labels/*.txt) into a JSONL/CSV-friendly format
for ticketing/reporting workflows. Optionally include crop paths if YOLO
was run with --save-crop.

Usage:
    python export_detections.py --run-dir runs/detect/predict --output results/detections.jsonl

What it outputs (one JSON object per line):
{
  "image": "cb123.jpg",
  "image_path": "/abs/path/to/cb123.jpg",
  "class_id": 0,
  "class_name": "illegal parking",
  "confidence": 0.92,
  "bbox_xywh_norm": [x_center, y_center, w, h],
  "bbox_xyxy_abs": [x1, y1, x2, y2],  # if image size is known
  "label_file": ".../labels/cb123.txt",
  "crop_path": ".../crops/illegal parking/cb123.jpg"  # if exists
}
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


def parse_label_file(path: Path):
    """
    YOLO txt format per line:
    class x_center y_center width height [confidence]
    """
    records = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else None
        records.append((cls, x, y, w, h, conf))
    return records


def find_crop(run_dir: Path, image_stem: str) -> Optional[Path]:
    crops_dir = run_dir / "crops"
    if not crops_dir.exists():
        return None
    # Search for a crop that starts with the image stem
    for sub in crops_dir.rglob("*"):
        if sub.is_file() and sub.stem.startswith(image_stem):
            return sub
    return None


def bbox_xyxy_abs(img_path: Path, xywh_norm: Tuple[float, float, float, float]) -> List[float]:
    with Image.open(img_path) as im:
        w_img, h_img = im.size
    x, y, w, h = xywh_norm
    x1 = (x - w / 2) * w_img
    y1 = (y - h / 2) * h_img
    x2 = (x + w / 2) * w_img
    y2 = (y + h / 2) * h_img
    return [x1, y1, x2, y2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="YOLO detect run directory (e.g., runs/detect/predict)")
    ap.add_argument("--output", default="results/detections.jsonl", help="Output JSONL file")
    ap.add_argument("--class-name", default="illegal parking", help="Class name to tag outputs")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    labels_dir = run_dir / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"No labels directory found at {labels_dir}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines_out = []
    label_files = sorted(labels_dir.glob("*.txt"))
    for lbl in label_files:
        image_stem = lbl.stem
        image_path = run_dir / "images" / f"{image_stem}.jpg"
        if not image_path.exists():
            # Sometimes YOLO saves images at run root; fallback
            alt = run_dir / f"{image_stem}.jpg"
            image_path = alt if alt.exists() else None

        for cls, x, y, w, h, conf in parse_label_file(lbl):
            record = {
                "image": f"{image_stem}.jpg",
                "image_path": str(image_path) if image_path else None,
                "class_id": cls,
                "class_name": args.class_name,
                "confidence": conf,
                "bbox_xywh_norm": [x, y, w, h],
                "label_file": str(lbl),
            }
            if image_path:
                record["bbox_xyxy_abs"] = bbox_xyxy_abs(image_path, (x, y, w, h))

            crop = find_crop(run_dir, image_stem)
            if crop:
                record["crop_path"] = str(crop)

            lines_out.append(record)

    with out_path.open("w") as f:
        for rec in lines_out:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(lines_out)} records to {out_path}")


if __name__ == "__main__":
    main()

