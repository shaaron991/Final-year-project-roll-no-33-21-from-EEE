"""
Driver Drowsiness Detection - YOLOv8s training, validation and test evaluation.

Run from the project root, with the Roboflow YOLOv8 export laid out as:

    dataset/
        train/images, train/labels
        valid/images, valid/labels      (Roboflow calls it "valid"; "validation" also accepted)
        test/images,  test/labels

    python model_training.py                 # audit + train + evaluate
    python model_training.py --audit-only    # dataset checks only, no training

Everything reported in the paper/PPT must come from the files this script writes
into  outputs/<RUN_NAME>/  (results.csv, metrics_summary.json, per_class_*.csv,
confusion matrices, figures/).
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"

# Class order MUST match the label files (index 0..8). Verified against the
# class list stored in the supplied best.pt checkpoint.
CLASS_NAMES = [
    "eyes_closed",             # 0
    "eyes_closed_head_left",   # 1
    "eyes_closed_head_right",  # 2
    "focused",                 # 3
    "head_down",               # 4
    "head_up",                 # 5
    "seeing_left",             # 6
    "seeing_right",            # 7
    "yawning",                 # 8  (the old checkpoint spells it "yarning")
]

MODEL_NAME = "yolov8s.pt"
EPOCHS = 100
PATIENCE = 20
IMG_SIZE = 640
BATCH = 16
SEED = 0
PROJECT = ROOT / "outputs"
RUN_NAME = "yolov8s_drowsiness"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------
# 2. DATASET AUDIT
# ----------------------------------------------------------------------------
def find_split_dir(name):
    """Return the folder for a split, tolerating valid/val/validation naming."""
    aliases = {"val": ["valid", "val", "validation"], "train": ["train"], "test": ["test"]}
    for a in aliases[name]:
        if (DATASET_DIR / a / "images").is_dir():
            return DATASET_DIR / a
    raise FileNotFoundError(f"No '{name}' split with an images/ folder under {DATASET_DIR}")


def source_key(stem):
    """Roboflow names augmented copies '<orig>_jpg.rf.<hash>'; strip the hash so
    copies of one photo share a key (used for train/val/test leakage check)."""
    return stem.split(".rf.")[0]


def audit_split(split_dir, n_classes):
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    lbls = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
    img_stems, lbl_stems = {p.stem for p in imgs}, {p.stem for p in lbls}

    problems, counts, boxes = [], Counter(), 0
    for lf in lbls:
        for ln, line in enumerate(lf.read_text().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                problems.append(f"{lf.name}:{ln} expected 5 fields, got {len(parts)}")
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                problems.append(f"{lf.name}:{ln} non-numeric value")
                continue
            if not 0 <= c < n_classes:
                problems.append(f"{lf.name}:{ln} class id {c} outside 0..{n_classes - 1}")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                problems.append(f"{lf.name}:{ln} box not normalised: {parts[1:]}")
            counts[c] += 1
            boxes += 1

    return {
        "dir": split_dir.name,
        "images": len(imgs),
        "labels": len(lbls),
        "images_without_label": sorted(img_stems - lbl_stems)[:5],
        "labels_without_image": sorted(lbl_stems - img_stems)[:5],
        "n_missing_label": len(img_stems - lbl_stems),
        "n_orphan_label": len(lbl_stems - img_stems),
        "boxes": boxes,
        "class_counts": {CLASS_NAMES[k]: counts[k] for k in sorted(counts) if k < n_classes},
        "problems": problems[:20],
        "n_problems": len(problems),
        "source_keys": {source_key(s) for s in img_stems},
    }


def audit_dataset():
    print("=" * 70, "\nDATASET AUDIT\n", "=" * 70, sep="")
    report, keys = {}, {}
    for split in ("train", "val", "test"):
        info = audit_split(find_split_dir(split), len(CLASS_NAMES))
        keys[split] = info.pop("source_keys")
        report[split] = info
        print(f"[{split:5s}] images={info['images']:5d} labels={info['labels']:5d} "
              f"boxes={info['boxes']:5d} missing_label={info['n_missing_label']} "
              f"orphan_label={info['n_orphan_label']} bad_lines={info['n_problems']}")
        if info["n_problems"]:
            for p in info["problems"][:5]:
                print("    !", p)

    # train/val/test must not share the same source photograph
    leaks = {
        "train_val": len(keys["train"] & keys["val"]),
        "train_test": len(keys["train"] & keys["test"]),
        "val_test": len(keys["val"] & keys["test"]),
    }
    report["leakage_shared_source_images"] = leaks
    print("Shared source images between splits:", leaks)
    if any(leaks.values()):
        print("WARNING: splits share source photos -> metrics will be optimistic.")

    total = Counter()
    for s in ("train", "val", "test"):
        total.update(report[s]["class_counts"])
    report["class_counts_total"] = dict(total)
    absent = [c for c in CLASS_NAMES if total[c] == 0]
    if absent:
        print("WARNING: classes with zero instances:", absent)

    hard_fail = any(report[s]["n_missing_label"] or report[s]["n_orphan_label"] or report[s]["n_problems"]
                    for s in ("train", "val", "test"))
    return report, hard_fail


def write_data_yaml():
    """data.yaml with absolute paths, so it works from any working directory."""
    cfg = {
        "path": str(DATASET_DIR.resolve()),
        "train": str((find_split_dir("train") / "images").relative_to(DATASET_DIR)),
        "val": str((find_split_dir("val") / "images").relative_to(DATASET_DIR)),
        "test": str((find_split_dir("test") / "images").relative_to(DATASET_DIR)),
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    if DATA_YAML.exists():
        old = yaml.safe_load(DATA_YAML.read_text())
        old_names = old.get("names")
        old_names = list(old_names.values()) if isinstance(old_names, dict) else list(old_names or [])
        if [n.lower().replace("yarning", "yawning") for n in old_names] != CLASS_NAMES:
            sys.exit(f"data.yaml class list differs from CLASS_NAMES:\n  yaml  : {old_names}\n"
                     f"  script: {CLASS_NAMES}\nFix one of them; class order must match the labels.")
        print(f"Using existing {DATA_YAML} (class list verified).")
        # still rewrite the path fields so relative-path problems cannot occur
        old.update({k: cfg[k] for k in ("path", "train", "val", "test")})
        cfg = old
    DATA_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return DATA_YAML


# ----------------------------------------------------------------------------
# 3. REPRODUCIBILITY + DEVICE
# ----------------------------------------------------------------------------
def setup_device():
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        return 0
    print("WARNING: no GPU found. 100 epochs on 5,595 images on CPU will take days; "
          "use Colab/Kaggle GPU or reduce EPOCHS.")
    return "cpu"


# ----------------------------------------------------------------------------
# 4. EVALUATION HELPERS
# ----------------------------------------------------------------------------
def metrics_dict(m):
    """Flatten an ultralytics DetMetrics object into plain floats."""
    names = m.names
    per_class = []
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.class_result(i)
        per_class.append({"class": names[int(c)], "precision": float(p), "recall": float(r),
                          "mAP50": float(ap50), "mAP50_95": float(ap)})
    mp, mr, map50, map5095 = m.mean_results()
    return {"precision": float(mp), "recall": float(mr), "mAP50": float(map50),
            "mAP50_95": float(map5095), "per_class": per_class,
            "speed_ms": {k: float(v) for k, v in m.speed.items()}}


def evaluate(weights, data_yaml, split, device, out_dir, tag):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split=split, imgsz=IMG_SIZE, batch=BATCH, device=device,
                  conf=0.001, iou=0.6, plots=True, save_json=False,
                  project=str(out_dir), name=f"eval_{tag}", exist_ok=True)
    d = metrics_dict(m)
    import pandas as pd
    pd.DataFrame(d["per_class"]).to_csv(out_dir / f"per_class_{tag}.csv", index=False)
    cm = getattr(m, "confusion_matrix", None)
    if cm is not None:
        np.savetxt(out_dir / f"confusion_matrix_{tag}.csv", cm.matrix, fmt="%d", delimiter=",")
    return d


def make_figures(run_dir, out_dir):
    """Training curves straight from results.csv written by ultralytics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(run_dir / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    ep = df["epoch"]

    def plot(cols, title, ylabel, fname):
        plt.figure(figsize=(5.2, 3.4))
        for col, lab in cols:
            if col in df:
                plt.plot(ep, df[col], label=lab, lw=1.6)
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
        plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=300); plt.close()

    plot([("train/box_loss", "train"), ("val/box_loss", "val")], "Box loss", "Loss", "loss_box.png")
    plot([("train/cls_loss", "train"), ("val/cls_loss", "val")], "Classification loss", "Loss", "loss_cls.png")
    plot([("train/dfl_loss", "train"), ("val/dfl_loss", "val")], "DFL loss", "Loss", "loss_dfl.png")
    plot([("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")],
         "Precision and recall (validation)", "Score", "precision_recall_epoch.png")
    plot([("metrics/mAP50(B)", "mAP@0.5"), ("metrics/mAP50-95(B)", "mAP@0.5:0.95")],
         "Validation mAP", "mAP", "map_epoch.png")


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="train even if the audit found problems")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    if not DATASET_DIR.is_dir():
        sys.exit(f"Dataset folder not found: {DATASET_DIR}")
    PROJECT.mkdir(exist_ok=True)

    report, dirty = audit_dataset()
    (PROJECT / "dataset_audit.json").write_text(json.dumps(report, indent=2))
    if dirty and not args.allow_dirty:
        sys.exit("Audit found missing/orphan labels or malformed label lines - fix them "
                 "(or pass --allow-dirty).")
    data_yaml = write_data_yaml()
    if args.audit_only:
        return

    device = setup_device()
    from ultralytics import YOLO

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(data_yaml), epochs=args.epochs, patience=PATIENCE, imgsz=IMG_SIZE, batch=BATCH,
        device=device, workers=0 if device == "cpu" else 4, seed=SEED, deterministic=True,
        project=str(PROJECT), name=RUN_NAME, exist_ok=False, cos_lr=False, close_mosaic=10,
        plots=True, save=True, save_period=-1, val=True, amp=(device != "cpu"),
    )

    run_dir = PROJECT / RUN_NAME
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        sys.exit("best.pt was not produced - training failed.")

    # Validation split (used for model selection) and held-out TEST split (reported once).
    summary = {
        "config": {"model": MODEL_NAME, "epochs_requested": args.epochs, "patience": PATIENCE,
                   "imgsz": IMG_SIZE, "batch": BATCH, "seed": SEED},
        "validation": evaluate(best, data_yaml, "val", device, run_dir, "val"),
        "test": evaluate(best, data_yaml, "test", device, run_dir, "test"),
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(run_dir, run_dir)

    shutil.copy2(best, ROOT / "best_trained.pt")
    print("\n" + "=" * 70)
    for split in ("validation", "test"):
        s = summary[split]
        print(f"{split:10s} P={s['precision']:.4f} R={s['recall']:.4f} "
              f"mAP50={s['mAP50']:.4f} mAP50-95={s['mAP50_95']:.4f}")
    print("Results folder:", run_dir)
    print("Model copied to best_trained.pt (use this in main.py)")


if __name__ == "__main__":
    main()