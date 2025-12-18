"""
Generate a simple ticket/evidence PNG for one predicted detection.

- Reads YOLO txt predictions from a run directory (labels/*.txt)
- Loads the corresponding image from the dataset split (data.yaml -> val/images by default)
- Draws the predicted box and confidence, and renders a ticket-style overlay

Usage:
    python generate_ticket_example.py \
        --run-dir results_illegal_parking/yolov8_illegal_ep5_balanced_val \
        --data data/illegal_parking_merged/data.yaml \
        --split val \
        --conf 0.25 \
        --output presentation_assets/ticket_example.png

Note: This is a lightweight visualization for the slide deck, not a production PDF.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import yaml
from PIL import Image, ImageDraw, ImageFont


def load_yaml(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def split_images_dir(data_yaml: Path, split: str) -> Path:
    cfg = load_yaml(data_yaml)
    base = data_yaml.parent
    split_path = Path(cfg[split])
    if not split_path.is_absolute():
        split_path = (base / split_path).resolve()
    # split points to .../images; ensure correct
    return split_path


def yolo_to_xyxy(rel_line: str, img_size: Tuple[int, int]):
    parts = rel_line.strip().split()
    if len(parts) < 6:
        return None
    _, xc, yc, w, h, conf = parts[:6]
    xc, yc, w, h, conf = map(float, (xc, yc, w, h, conf))
    W, H = img_size
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return (x1, y1, x2, y2, conf)


def pick_first_pred(run_dir: Path, conf_thres: float):
    labels_dir = run_dir / "labels"
    for lbl in sorted(labels_dir.glob("*.txt")):
        lines = [ln for ln in lbl.read_text().strip().splitlines() if ln.strip()]
        for ln in lines:
            box = yolo_to_xyxy(ln, (1, 1))  # dummy; will recompute with real size
            if box and box[4] >= conf_thres:
                return lbl, ln
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run dir with labels/")
    ap.add_argument("--data", required=True, help="data.yaml to locate images")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--output", default="presentation_assets/ticket_example.png")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    img_dir = split_images_dir(Path(args.data), args.split)

    lbl_file, line = pick_first_pred(run_dir, args.conf)
    if not lbl_file:
        raise SystemExit("No predictions found above conf threshold")

    stem = lbl_file.stem
    img_path = img_dir / f"{stem}.jpg"
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2, conf = yolo_to_xyxy(line, (W, H))

    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    text = f"Illegal parking (conf {conf:.2f})"

    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    text_bg = draw.textbbox((0, 0), text, font=font)
    tx1, ty1, tx2, ty2 = text_bg
    tw, th = tx2 - tx1, ty2 - ty1
    draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill="red")
    draw.text((x1 + 3, y1 - th - 3), text, fill="white", font=font)

    # Add a simple ticket-style footer
    footer_h = 80
    ticket = Image.new("RGB", (W, H + footer_h), "white")
    ticket.paste(img, (0, 0))
    d2 = ImageDraw.Draw(ticket)
    footer_text = f"Image: {img_path.name} | Detection: illegal parking | Conf: {conf:.2f}"
    d2.rectangle([0, H, W, H + footer_h], fill="#f0f0f0")
    d2.text((10, H + 10), footer_text, fill="black", font=font)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ticket.save(out_path)
    print(json.dumps({"output": str(out_path), "image": str(img_path), "label_file": str(lbl_file), "conf": conf}, indent=2))


if __name__ == "__main__":
    main()

