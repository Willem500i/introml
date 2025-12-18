"""
Compute a 2x2 confusion matrix that includes background (empty labels) for a YOLO run.
Counts are per-image (any detection over conf threshold counts as positive for that image).

Usage:
  python confusion_with_background.py \
    --run-dir results_illegal_parking/yolov8_illegal_ep1_balanced_v2 \
    --data data/illegal_parking_merged/data.yaml \
    --split val \
    --conf 0.25

Outputs JSON counts and saves confusion_with_bg.png in the run dir.
"""

import argparse
import json
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


def load_yaml(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def collect_gt_labels(data_yaml: Path, split: str):
    cfg = load_yaml(data_yaml)
    base = data_yaml.parent
    split_path = Path(cfg[split])
    if not split_path.is_absolute():
        split_path = (base / split_path).resolve()
    labels_dir = split_path.parent / "labels"
    return {p.stem: p for p in labels_dir.glob("*.txt")}


def read_gt_has(path: Path) -> bool:
    return path.exists() and path.read_text().strip() != ""


def read_pred_has(path: Path, conf_thres: float) -> bool:
    """
    Returns True if the prediction file has at least one detection
    with confidence >= conf_thres.
    """
    if not path.exists():
        return False
    lines = [ln.strip() for ln in path.read_text().strip().splitlines() if ln.strip()]
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 6:
            conf = float(parts[5])
            if conf >= conf_thres:
                return True
    return False


def compute_confusion(run_dir: Path, data_yaml: Path, split: str, conf_thres: float):
    pred_dir = run_dir / "labels"
    gt_files = collect_gt_labels(data_yaml, split)

    tp = fp = fn = tn = 0
    pred_pos = 0
    for stem, gtf in gt_files.items():
        gt_has = read_gt_has(gtf)
        predf = pred_dir / f"{stem}.txt"
        pred_has = read_pred_has(predf, conf_thres)

        if gt_has and pred_has:
            tp += 1
        elif gt_has and not pred_has:
            fn += 1
        elif (not gt_has) and pred_has:
            fp += 1
        else:
            tn += 1

        if pred_has:
            pred_pos += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_total": len(gt_files),
        "gt_pos": sum(1 for p in gt_files.values() if read_gt_has(p)),
        "gt_neg": sum(1 for p in gt_files.values() if not read_gt_has(p)),
        "pred_pos": pred_pos,
        "pred_neg": len(gt_files) - pred_pos,
    }


def save_heatmap(conf, out_path: Path):
    import numpy as np

    mat = np.array([[conf["tp"], conf["fn"]],
                    [conf["fp"], conf["tn"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Illegal (pred)", "Background (pred)"],
        yticklabels=["Illegal (true)", "Background (true)"],
        cbar=True,
        ax=ax,
    )
    ax.set_title(
        f"Confusion incl. background\n"
        f"GT pos/neg: {conf['gt_pos']}/{conf['gt_neg']} | Pred pos/neg: {conf['pred_pos']}/{conf['pred_neg']}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="YOLO run directory with labels/")
    ap.add_argument("--data", required=True, help="data.yaml path (to locate GT labels)")
    ap.add_argument("--split", default="val", help="split key in data.yaml (val/test/train)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for counting a prediction as positive")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_yaml = Path(args.data)

    conf = compute_confusion(run_dir, data_yaml, args.split, args.conf)
    print(json.dumps(conf, indent=2))

    out_png = run_dir / "confusion_with_bg.png"
    save_heatmap(conf, out_png)
    print(f"Saved confusion_with_bg.png to {out_png}")


if __name__ == "__main__":
    main()
