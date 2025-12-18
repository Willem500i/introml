"""
Create car-centric negative crops from BDD100K images to better match the
Roboflow illegal-parking framing (vehicle centered, ~60% of image).

Output:
  data/negative_crops_cars/{train,val,test}/images/*.jpg
  data/negative_crops_cars/{train,val,test}/labels/*.txt  (empty)

Usage:
  python make_negative_car_crops.py

You can tweak limits and filters in CONFIG below.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


CONFIG = {
    "bdd_images_root": Path("data/100k"),          # images/{train,val,test}
    "bdd_ann_root": Path("data/100k-2"),           # json annotations
    "output_root": Path("data/negative_crops_cars"),
    "splits": {"train": 650, "val": 75, "test": 40},   # number of crops to save
    "padding": 0.2,                                # expand bbox by 20%
    "min_area": 8000,                              # filter tiny/distant cars
    "max_area": 200000,                            # filter huge crops
    "min_vis_width": 80,                           # avoid tiny widths
    "aspect_min": 0.5,                             # w/h lower bound
    "aspect_max": 2.5,                             # w/h upper bound
    "seed": 42,
    "resize_to": (640, 640),                       # final size to match positives
    "vehicle_categories": {"car", "truck", "bus"},
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_ann(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def crop_with_padding(im: Image.Image, box: Dict, pad: float) -> Image.Image:
    w, h = im.size
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw, bh = x2 - x1, y2 - y1
    x1p = max(0, x1 - pad * bw)
    y1p = max(0, y1 - pad * bh)
    x2p = min(w, x2 + pad * bw)
    y2p = min(h, y2 + pad * bh)
    return im.crop((int(x1p), int(y1p), int(x2p), int(y2p)))


def valid_box(box: Dict, cfg: Dict) -> bool:
    w = box["x2"] - box["x1"]
    h = box["y2"] - box["y1"]
    area = w * h
    if area < cfg["min_area"] or area > cfg["max_area"]:
        return False
    if w < cfg["min_vis_width"] or h < cfg["min_vis_width"]:
        return False
    aspect = w / max(h, 1e-6)
    if aspect < cfg["aspect_min"] or aspect > cfg["aspect_max"]:
        return False
    return True


def collect_candidates(split: str, cfg: Dict) -> List[Tuple[Path, Dict]]:
    ann_dir = cfg["bdd_ann_root"] / split
    img_dir = cfg["bdd_images_root"] / split
    cands = []
    for ann_path in ann_dir.glob("*.json"):
        ann = load_ann(ann_path)
        fname = ann.get("name")
        if not fname:
            continue
        img_path = img_dir / f"{fname}.jpg"
        if not img_path.exists():
            continue
        for frame in ann.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") not in cfg["vehicle_categories"]:
                    continue
                box = obj.get("box2d")
                if not box:
                    continue
                if not valid_box(box, cfg):
                    continue
                attrs = obj.get("attributes", {})
                if attrs.get("truncated", False):
                    continue
                # accept occluded or not; keep variety
                cands.append((img_path, box))
    return cands


def save_crops(split: str, candidates: List[Tuple[Path, Dict]], cfg: Dict):
    random.shuffle(candidates)
    target_n = cfg["splits"].get(split, 0)
    out_img_dir = cfg["output_root"] / split / "images"
    out_lbl_dir = cfg["output_root"] / split / "labels"
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)
    saved = 0
    for img_path, box in candidates:
        if saved >= target_n:
            break
        try:
            with Image.open(img_path) as im:
                crop = crop_with_padding(im, box, cfg["padding"])
                crop = crop.convert("RGB")
                if cfg["resize_to"]:
                    crop = crop.resize(cfg["resize_to"], Image.BILINEAR)
                out_name = f"{img_path.stem}_neg{saved}.jpg"
                crop.save(out_img_dir / out_name)
                # empty label file marks background
                (out_lbl_dir / f"{Path(out_name).stem}.txt").write_text("")
                saved += 1
        except Exception:
            continue
    return saved


def main():
    cfg = CONFIG.copy()
    random.seed(cfg["seed"])
    ensure_dir(cfg["output_root"])

    for split in cfg["splits"]:
        cands = collect_candidates(split, cfg)
        saved = save_crops(split, cands, cfg)
        print(f"{split}: requested {cfg['splits'][split]}, saved {saved}, candidates {len(cands)}")


if __name__ == "__main__":
    main()

