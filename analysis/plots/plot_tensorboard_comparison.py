#!/usr/bin/env python3
"""
Extract TensorBoard scalars from checkpoint zips and plot per-model comparison curves.
Usage: python plot_tensorboard_comparison.py [--ckpts-dir DIR] [--out-dir DIR]
"""
import argparse
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mvc.paths import repo_root
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

METRICS = ["val_accuracy", "val_precision", "val_recall", "val_auroc", "val_loss"]
METRIC_LABELS = {
    "val_accuracy":  "Validation Accuracy",
    "val_precision": "Validation Precision",
    "val_recall":    "Validation Recall",
    "val_auroc":     "Validation AUROC",
    "val_loss":      "Validation Loss",
}


def latest_zip_per_model(ckpts_dir):
    """Return {model_name: path_to_latest_zip} — latest by timestamp in filename."""
    pattern = re.compile(r"^(allclass_\w+)_(\d{8}_\d{6})\.zip$")
    by_model = defaultdict(list)
    for fname in os.listdir(ckpts_dir):
        m = pattern.match(fname)
        if m:
            by_model[m.group(1)].append((m.group(2), os.path.join(ckpts_dir, fname)))
    return {model: sorted(runs)[-1][1] for model, runs in by_model.items()}


def extract_event_files(zip_path, tmp_dir):
    """Extract only tfevents files from zip into tmp_dir."""
    with zipfile.ZipFile(zip_path) as zf:
        event_members = [n for n in zf.namelist() if "events.out.tfevents" in n]
        for member in event_members:
            zf.extract(member, tmp_dir)
    # EventAccumulator needs a directory containing the event files (flat or nested)
    # Return the deepest directory that contains event files
    for root, _, files in os.walk(tmp_dir):
        if any("events.out.tfevents" in f for f in files):
            return root
    return tmp_dir


def read_scalars(event_dir, tags):
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    result = {}
    for tag in tags:
        if tag not in available:
            continue
        result[tag] = [e.value for e in ea.Scalars(tag)]
    return result


def plot_metric(metric, model_data, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for model, data in sorted(model_data.items()):
        if metric not in data:
            continue
        values = data[metric]
        epochs = list(range(1, len(values) + 1))
        ax.plot(epochs, values, label=model.replace("allclass_", ""), linewidth=1.2)

    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts-dir", default="/home/ammar/public_html/magician/ckpts2")
    parser.add_argument("--out-dir", default=os.path.join(repo_root(), "tb_plots"))
    parser.add_argument("--exclude", nargs="+", default=[], metavar="MODEL",
                        help="Model name(s) to skip, e.g. --exclude mnasnet1_0 swin_v2_t")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    exclude = {f"allclass_{m}" if not m.startswith("allclass_") else m for m in args.exclude}
    zips = {m: p for m, p in latest_zip_per_model(args.ckpts_dir).items() if m not in exclude}
    print(f"Found {len(zips)} models:")
    for m in sorted(zips):
        print(f"  {m} <- {os.path.basename(zips[m])}")

    model_data = {}
    tmp_root = tempfile.mkdtemp(prefix="tb_extract_")
    try:
        for model, zip_path in sorted(zips.items()):
            tmp_dir = os.path.join(tmp_root, model)
            os.makedirs(tmp_dir)
            print(f"\nExtracting {model}...")
            event_dir = extract_event_files(zip_path, tmp_dir)
            data = read_scalars(event_dir, METRICS)
            if not any(k in data for k in METRICS):
                print(f"  WARNING: no scalar metrics found, skipping")
                continue
            model_data[model] = data
            n_epochs = max((len(data[m]) for m in METRICS if m in data), default=0)
            print(f"  epochs={n_epochs}, "
                  + ", ".join(f"{m}={len(data[m])}" for m in METRICS if m in data))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nGenerating plots -> {args.out_dir}")
    for metric in METRICS:
        out_path = os.path.join(args.out_dir, f"{metric}.png")
        plot_metric(metric, model_data, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
