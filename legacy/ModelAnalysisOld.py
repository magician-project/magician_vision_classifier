#!/usr/bin/python3
"""
ModelAnalysis.py

All-in-one analysis tool that:
  1. Discovers every (.pth, .json) model pair in the current directory
  2. Evaluates each model on the given dataset (per-class metrics + confusion matrices)
  3. Collects per-sample softmax probabilities for ensemble search
  4. Runs greedy forward/backward ensemble optimisation (all three voting strategies)
  5. Produces a self-contained HTML report with all metrics, confusion matrix images,
     and ensemble recommendations

Usage:
    python3 ModelAnalysis.py <dataset_dir> [models_dir] [--batch <N>] [--fp16] [--out <dir>]

    dataset_dir : ImageFolder-style directory or directory containing dataset.h5
    models_dir  : directory containing *.pth + matching *.json  (default: ".")
    --batch N   : inference batch size  (default: 64)
    --fp16      : run inference under autocast FP16 (matches live-pipeline precision)
    --out dir   : output directory for report + assets  (default: "model_analysis_<timestamp>")
"""

import argparse
import base64
import datetime
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from trainMagicianVisionClassifierTorch import (
    Classifier,
    RGBAImageFolder,
    load_hyperparameters,
    metadata_collate_fn,
)
from evaluateClassifierNew import (
    build_eval_to_model_mapping,
    get_clean_class_id,
    get_state_dict_from_checkpoint,
    is_valid_dataset_dir,
    run_throughput_benchmark,
    BENCHMARK_BATCH_SIZES,
    BENCHMARK_STEP_SIZES,
)

try:
    from DatasetConverter import HDF5Dataset
    _HAS_HDF5 = True
except Exception:
    _HAS_HDF5 = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WARMUP_BATCHES     = 3
MIN_GAIN_THRESHOLD  = 0.0005
MAX_DROP_THRESHOLD  = 0.005
STRATEGIES          = ["soft", "majority", "confidence_weighted"]
DEFAULT_METRIC      = "balanced_accuracy"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(dataset_dir: str):
    """
    Load the dataset from disk, supporting two formats:
      1. HDF5 (dataset.h5) — loaded via HDF5Dataset if available
      2. ImageFolder (PNG files) — loaded via RGBAImageFolder with channel-first tensor transform

    The transform converts PIL images to (C, H, W) contiguous float tensors suitable for model input.

    Args:
        dataset_dir: Path to directory containing either dataset.h5 or image subfolders.

    Returns:
        A dataset object implementing __len__ and __getitem__.
    """
    transform = transforms.Lambda(
        lambda img: torch.from_numpy(img).permute(2, 0, 1).contiguous()
    )
    h5 = os.path.join(dataset_dir, "dataset.h5")
    if _HAS_HDF5 and os.path.isfile(h5):
        print(f"  Loading HDF5 dataset: {h5}")
        return HDF5Dataset(h5)
    print(f"  Loading PNG dataset:  {dataset_dir}")
    return RGBAImageFolder(root=dataset_dir, transform=transform, return_metadata=True)


def _discover_pairs(models_dir: str) -> list[tuple[str, str]]:
    """
    Find all (.pth, .json) model checkpoint + config pairs in models_dir.
    A .pth file is only paired if a corresponding .json (same stem) exists.

    Args:
        models_dir: Directory containing model checkpoint files.

    Returns:
        Sorted list of (pth_path, json_path) tuples.
    """
    pairs = []
    for f in sorted(Path(models_dir).glob("*.pth")):
        cfg = f.with_suffix(".json")
        if cfg.is_file():
            pairs.append((str(f), str(cfg)))
        else:
            print(f"  [WARN] No matching .json for {f.name} — skipping.")
    return pairs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _instantiate(config_json: dict, class_names: list[str]) -> Classifier:
    """
    Reconstruct a Classifier model from its saved config JSON and a class name list.
    Reads all hyperparameters, architecture flags, and training options from the JSON.

    Args:
        config_json: Parsed JSON containing model architecture, hparams, optimizer, and flags.
        class_names: List of class names used to determine num_classes and identify clean_class.

    Returns:
        An unweighted Classifier instance ready for checkpoint loading.
    """
    return Classifier(
        model               = config_json["model"],
        loss                = config_json.get("loss", "focal"),
        tile_size           = config_json["hparams"]["tile_size"],
        num_classes         = len(class_names),
        dropout_rate        = config_json["hparams"]["dropout_rate"],
        lr                  = config_json["optimizer"]["learning_rate"],
        AoLP                = config_json.get("AoLP", False),
        DoLP                = config_json.get("DoLP", False),
        Unpolarized         = config_json.get("Unpolarized", False),
        MaxPolarization     = config_json.get("MaxPolarization", False),
        MinPolarization     = config_json.get("MinPolarization", False),
        RangePolarization   = config_json.get("RangePolarization", False),
        penalize_false_clean= float(config_json.get("penalize_false_clean", 0.0)),
        base_channels       = config_json["hparams"].get("base_channels", 32),
        final_dense_layer   = config_json["hparams"].get("final_dense_layer", 512),
        clean_class         = get_clean_class_id(class_names),
        noise_std           = config_json["hparams"].get("noise_std", 0.0),
        noise_clip          = config_json["hparams"].get("noise_clip", None),
    )


def _load_weights(clf: Classifier, pth_path: str, device: str) -> Classifier:
    """
    Load checkpoint weights into the classifier with strict=False to allow mismatched keys.
    Then move to device and set eval mode.

    Args:
        clf: Instantiated but unweighted Classifier.
        pth_path: Path to the PyTorch checkpoint (.pth) file.
        device: 'cuda' or 'cpu' target device.

    Returns:
        The Classifier with weights loaded, moved to device, and in eval() mode.
    """
    sd = get_state_dict_from_checkpoint(pth_path, device)
    clf.load_state_dict(sd, strict=False)
    return clf.to(device).eval()


def _count_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Number of batches to pre-load for the clean GPU timing pass in _run_model.
# Small enough to keep RAM usage bounded (bs=64: ~13 MB; bs=256: ~50 MB),
# but large enough to amortise CUDA launch overhead across many batches.
_N_TIME_BATCHES = 20


@torch.no_grad()
def _run_model(clf: Classifier, loader: DataLoader, device: str,
               eval_to_model: dict[int, int], catchall_model_indices: list[int],
               catchall_ds_indices: list[int], use_fp16: bool,
               desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Two-pass inference over the dataset:

      Pass 1 — Warmup + Prediction
        Runs _WARMUP_BATCHES batches (discarded) to prime the CUDA kernel cache
        and cuDNN autotuner, then iterates the full dataset to collect y_true,
        y_pred, and softmax probabilities.  Timing is intentionally excluded
        from this pass: DataLoader I/O (HDF5 reads, PNG decodes, numpy→tensor),
        softmax, and .cpu().numpy() transfers all inflate wall-clock time and
        do not represent the cost the live pipeline pays per sample.

      Pass 2 — Clean GPU throughput measurement
        Pre-loads _N_TIME_BATCHES batches from a fresh DataLoader iteration into
        CPU RAM (I/O happens here, before the timer starts).  Then times only
        PCIe host→device transfer + GPU forward pass with a single CUDA
        synchronize at the end (GPU ops are pipelined; the sync flushes them
        all before reading the clock).  The CUDA kernel cache is already warm
        from Pass 1, so no additional warmup is needed.

        ms_per_sample = total_gpu_wall_time_ms / total_samples_timed

    Args:
        clf: Loaded classifier in eval mode on target device.
        loader: DataLoader yielding (images, labels, ...) batches.  DataLoader
                supports re-iteration — each `for` loop creates a fresh iterator,
                so Pass 1 and Pass 2 both see the full dataset from the start.
        device: 'cuda' or 'cpu'.
        eval_to_model: Maps dataset class index → model class index.  Samples
                       whose label is not a key in this dict are skipped.
        catchall_model_indices: Model output indices that act as a catch-all for
                                defect classes the model was not trained on.
        catchall_ds_indices: Dataset class indices covered by the catch-all.
        use_fp16: Apply torch.amp.autocast(float16) on CUDA, matching the live
                  pipeline precision and Tensor Core utilisation.
        desc: Progress bar label string.

    Returns:
        y_true_ds    : (N,) int64 — ground truth in dataset class space.
        y_pred       : (N,) int64 — argmax prediction in model class space.
        probs        : (N, K_model) float32 — softmax probability vectors.
        ms_per_sample: float — PCIe+GPU inference time per sample (Hz = 1000/ms).
    """
    clf.eval()
    clf.to(device)

    # Autocast context: FP16 on CUDA when requested, disabled otherwise.
    # Using FP16 matches the live inference precision and exercises Tensor Cores,
    # giving throughput numbers that reflect the real deployment scenario.
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16 and device == "cuda"
        else torch.amp.autocast(device_type="cuda", enabled=False)
        if device == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    # ── Pass 1a: Warmup ──────────────────────────────────────────────────────
    # Run a few batches (discarded) to:
    #   - Trigger cuDNN autotuner (torch.backends.cudnn.benchmark=True picks
    #     the fastest convolution algorithm for this input shape).
    #   - Pre-allocate CUDA memory pools so the allocator doesn't page during
    #     the prediction or timing loops.
    #   - Compile torch.compile graphs (if the model was compiled).
    for i, (x, *_) in enumerate(loader):
        with autocast_ctx:
            clf(x.to(device))
        if i + 1 >= _WARMUP_BATCHES:
            break
    if device == "cuda":
        torch.cuda.synchronize()

    # ── Pass 1b: Prediction ──────────────────────────────────────────────────
    # Collect ground-truth labels, argmax predictions, and full softmax vectors
    # for every dataset sample whose class appears in eval_to_model.
    # No timing here — wall-clock would include DataLoader I/O and Python loop
    # overhead, which are not part of GPU inference cost.
    y_true_ds_list: list[int] = []
    y_pred_list:    list[int] = []
    prob_list:      list[np.ndarray] = []

    for x, y, *_ in tqdm(loader, desc=f"    {desc}", unit="batch", leave=False, dynamic_ncols=True):
        x = x.to(device)
        with autocast_ctx:
            logits = clf(x)
        # Cast logits back to float32 before softmax in case autocast produced
        # float16 logits — softmax in float16 can overflow for large logit values.
        p = torch.softmax(logits.float(), dim=1).cpu().numpy()
        for i, ev in enumerate(y.numpy()):
            ev = int(ev)
            # Skip samples whose dataset class has no mapping to a model output.
            # This happens for classes the model was never trained on when
            # catch-all binary-model handling is not active.
            if ev not in eval_to_model:
                continue
            y_true_ds_list.append(ev)
            y_pred_list.append(int(p[i].argmax()))
            prob_list.append(p[i])

    if not y_true_ds_list:
        return np.array([], np.int64), np.array([], np.int64), np.zeros((0, 0), np.float32), 0.0

    # ── Pass 2: Clean GPU throughput measurement ─────────────────────────────
    # Pre-load _N_TIME_BATCHES from a fresh DataLoader iteration into CPU RAM.
    # DataLoader supports re-iteration: each `for` loop calls __iter__() which
    # returns a fresh _DataLoaderIter starting from the beginning of the dataset.
    # The collection loop below includes disk I/O — intentionally, since we want
    # the I/O to complete BEFORE the GPU timer starts.
    time_cpu: list[torch.Tensor] = []
    for x, *_ in loader:           # Fresh iteration — I/O happens here
        time_cpu.append(x)
        if len(time_cpu) >= _N_TIME_BATCHES:
            break

    if time_cpu:
        # The CUDA kernel cache is already warm from Pass 1, so we go directly
        # into the timed loop without a separate warmup here.
        if device == "cuda":
            torch.cuda.synchronize()   # Flush any lingering GPU ops from Pass 1
        t_gpu = time.perf_counter()
        for x in time_cpu:
            with autocast_ctx:
                clf(x.to(device))      # PCIe transfer + GPU forward (pipelined)
        if device == "cuda":
            # Single synchronize after all batches: GPU work is queued async, so
            # reading the clock before this would give zero elapsed time.
            # One sync here measures TRUE throughput (not per-batch latency).
            torch.cuda.synchronize()
        timed_n = sum(b.shape[0] for b in time_cpu)
        ms_per_sample = (time.perf_counter() - t_gpu) * 1000.0 / max(timed_n, 1)
    else:
        ms_per_sample = 0.0

    return (
        np.array(y_true_ds_list, dtype=np.int64),
        np.array(y_pred_list,    dtype=np.int64),
        np.array(prob_list,      dtype=np.float32),
        ms_per_sample,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(y_true: np.ndarray, y_pred: np.ndarray,
           class_names: list[str]) -> dict:
    """
    Compute per-class and aggregate classification metrics.

    Args:
        y_true: Ground truth labels (N,) in dataset class space.
        y_pred: Predicted labels (N,) in dataset class space.
        class_names: Ordered list of all class names (defines K and label order).

    Returns:
        Dict with keys:
            accuracy: Overall accuracy.
            balanced_accuracy: Mean of per-class recalls.
            num_samples: Total number of samples.
            per_class: List of dicts (one per class) with tp/tn/fp/fn/precision/recall/f1/support.
            cm: Raw confusion matrix (K, K).
            cm_norm: Row-normalised confusion matrix (each row sums to 1.0).
    """
    K      = len(class_names)
    labels = list(range(K))
    # Return zeros for empty input to avoid sklearn errors.
    if len(y_true) == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0,
                "num_samples": 0, "per_class": [], "cm": [], "cm_norm": []}

    acc     = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm      = confusion_matrix(y_true, y_pred, labels=labels)
    # Row-normalise: cm_norm[i, j] = fraction of class i samples predicted as class j.
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sum,
                        out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        per_class.append({
            "class_name": name, "support": int(sup[i]),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
        })
    return {
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "num_samples": int(len(y_true)),
        "per_class": per_class,
        "cm": cm.tolist(), "cm_norm": cm_norm.tolist(),
    }


# ---------------------------------------------------------------------------
# Confusion matrix image (saved to disk, embedded as base64 in HTML)
# ---------------------------------------------------------------------------

def _save_confusion_png(cm_norm: list[list[float]], class_names: list[str],
                        title: str, out_path: str):
    """
    Render a row-normalised confusion matrix as a seaborn heatmap and save to disk.
    The heatmap is embedded as base64 in the HTML report.

    Args:
        cm_norm: 2D list of normalised confusion matrix values in [0, 1].
        class_names: Class labels for x/y axis ticks.
        title: Plot title string.
        out_path: File path to save the PNG.
    """
    arr = np.array(cm_norm, dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.1),
                                    max(4, len(class_names) * 0.9)))
    sns.heatmap(arr, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_title(title,        fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _png_to_b64(path: str) -> str:
    """Read a PNG file and return its base64-encoded string for inline embedding in HTML."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _save_benchmark_3d_plot(bench_results: list[dict], model_name: str, out_path: str):
    """
    Render a 3-D scatter plot of throughput benchmark results and save to disk.

    Axes:
        X: batch_size     — number of tiles sent to the GPU per forward call.
        Y: step_size      — tile stride in pixels; smaller = denser coverage = more tiles.
        Z: effective_fps  — FULL-IMAGE frames per second (Hz), NOT per-sample throughput.
                            Formula: 1 / (ceil(num_tiles / batch_size) × tpb_seconds)
                            where tpb is measured from pure PCIe+GPU time (no disk I/O).
                            Large batch + large step → high FPS.
                            Small batch + small step → very low FPS (many batches needed).

    Colour: balanced accuracy mapped via RdYlGn (red=low, green=high).
    Red X markers indicate (batch_size, step_size) combinations that caused an OOM error.

    Note on "very low" FPS for small step sizes:
        step=1 on a 1224×1024 image with tile=48 creates ~1.1M tiles; even a fast GPU
        at 64 tiles/batch needs ~18 000 batches → FPS will be a fraction of 1 Hz.
        This is physically correct — it reflects the true cost of dense scanning.
        Use step ≥ 4 for practical real-time operation.

    Args:
        bench_results: List of dicts from run_throughput_benchmark, each with
                       batch_size, step_size, effective_fps (Hz), balanced_accuracy, oom.
        model_name: Displayed in the plot title.
        out_path: File path to save the PNG.
    """
    if not bench_results:
        return

    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from matplotlib.lines import Line2D

    ok   = [r for r in bench_results if not r.get("oom")]
    fail = [r for r in bench_results if r.get("oom")]

    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")

    sc = None
    if ok:
        xs = np.array([r["batch_size"]        for r in ok], dtype=float)
        ys = np.array([r["step_size"]         for r in ok], dtype=float)
        zs = np.array([r["effective_fps"]     for r in ok], dtype=float)
        cs = np.array([r["balanced_accuracy"] for r in ok], dtype=float)

        vmin = max(0.0, cs.min() - 0.05)
        vmax = min(1.0, cs.max() + 0.02)

        sc = ax.scatter(xs, ys, zs,
                        c=cs, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                        s=90, depthshade=True, edgecolors="k", linewidths=0.3)

    if fail:
        fx = np.array([r["batch_size"] for r in fail], dtype=float)
        fy = np.array([r["step_size"]  for r in fail], dtype=float)
        fz = np.zeros(len(fail))
        ax.scatter(fx, fy, fz, c="red", marker="x", s=120, linewidths=2,
                   zorder=5, label="OOM")
        legend_handles = [Line2D([0], [0], marker="x", color="red", linestyle="None",
                                 markersize=8, markeredgewidth=2, label="OOM")]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    ax.set_xlabel("Batch size",    labelpad=8)
    ax.set_ylabel("Step size",     labelpad=8)
    ax.set_zlabel("Effective FPS", labelpad=8)
    ax.set_title(f"{model_name}\nBatch × Step throughput sweep",
                 fontsize=10, fontweight="bold")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.55, aspect=15)
        cbar.set_label("Balanced Accuracy", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def _save_efficiency_plot(model_results: list[dict], dataset_classes: list[str], out_path: str):
    """
    Save a 3-panel efficiency comparison figure:
      - Scatter: num_params (M) vs balanced_accuracy — shows the accuracy/size tradeoff frontier.
      - Bar: balanced_accuracy / M_params per model — overall efficiency ranking.
      - Grouped bar: per-class recall / M_params — one group per class, one bar per model.

    The efficiency metric (accuracy / M_params) rewards models that achieve high accuracy
    with fewer parameters; it is useful for selecting the smallest model that meets a
    given accuracy threshold.
    """
    if not model_results:
        return

    n_models = len(model_results)
    n_cls    = len(dataset_classes)

    m_params = np.array([m["num_params"] / 1_000_000 for m in model_results])
    bal_accs = np.array([m["balanced_accuracy"]       for m in model_results])
    eff      = bal_accs / np.maximum(m_params, 1e-9)   # BalAcc per M params

    # Per-class recall / M_params matrix  (n_cls × n_models)
    per_cls_eff = np.zeros((n_cls, n_models))
    for mi, m in enumerate(model_results):
        cls_map = {pc["class_name"]: pc["recall"] for pc in m["per_class"]}
        for ci, cls in enumerate(dataset_classes):
            per_cls_eff[ci, mi] = cls_map.get(cls, 0.0) / max(m_params[mi], 1e-9)

    cmap   = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(n_models)]

    short_names = [m["name"].replace("allclass_", "").replace("binary_", "⚡ ")
                   for m in model_results]

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.45, wspace=0.35)

    # ── Top-left: scatter params vs accuracy ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(n_models):
        ax1.scatter(m_params[i], bal_accs[i], s=110, color=colors[i], zorder=5,
                    edgecolors="k", linewidths=0.5, label=short_names[i])
        ax1.annotate(short_names[i], (m_params[i], bal_accs[i]),
                     fontsize=6, xytext=(5, 4), textcoords="offset points")
    ax1.set_xlabel("Parameters (M)", fontsize=9)
    ax1.set_ylabel("Balanced Accuracy",  fontsize=9)
    ax1.set_title("Accuracy vs Model Size", fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # ── Top-right: efficiency bar per model (sorted best → worst) ───────────
    ax2   = fig.add_subplot(gs[0, 1])
    order = np.argsort(eff)[::-1]
    ax2.barh(range(n_models), eff[order], color=[colors[i] for i in order], alpha=0.85)
    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels([short_names[i] for i in order], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Balanced Acc / M params", fontsize=9)
    ax2.set_title("Overall Efficiency (BalAcc per M params)", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)
    for yi, mi in enumerate(order):
        ax2.text(eff[mi], yi, f"  {eff[mi]:.4f}", va="center", fontsize=7)

    # ── Bottom: per-class recall / M_params grouped bar ─────────────────────
    ax3   = fig.add_subplot(gs[1, :])
    x     = np.arange(n_cls)
    width = min(0.8 / n_models, 0.14)
    offs  = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)
    for mi in range(n_models):
        ax3.bar(x + offs[mi], per_cls_eff[:, mi], width,
                label=short_names[mi], color=colors[mi], alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(dataset_classes, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Recall / M params", fontsize=9)
    ax3.set_title("Per-class Efficiency (Recall per Million Params)", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax3.grid(True, axis="y", alpha=0.3)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Ensemble helpers (mirrors evaluateOptimalEnsemble.py)
# ---------------------------------------------------------------------------

def _ensemble_predict(probs: np.ndarray, subset: list[int], strategy: str) -> np.ndarray:
    """
    Predict class labels by combining probabilities from a subset of models.

    Args:
        probs: (M, N, K) array — softmax probabilities for all M models, N samples, K classes.
        subset: List of model indices (into probs axis 0) to include in the ensemble.
        strategy: Voting strategy — "soft", "majority", or "confidence_weighted".

    Returns:
        (N,) int64 array of predicted class indices.

    Strategies:
        soft: Average the softmax vectors across models, then argmax.
        majority: Each model votes for its top-1 class; the class with the most votes wins.
        confidence_weighted: Multiply each model's softmax by its top-1 confidence,
                             then average and argmax across models.
    """
    sub = probs[subset]
    if strategy == "soft":
        # Soft voting: average probabilities, pick highest.
        return sub.mean(axis=0).argmax(axis=1).astype(np.int64)
    elif strategy == "majority":
        # Hard voting: each model casts one vote for its top-1 class.
        votes = sub.argmax(axis=2)
        S, N  = votes.shape
        K     = probs.shape[2]
        cnt   = np.zeros((N, K), dtype=np.int32)
        for s in range(S):
            np.add.at(cnt, (np.arange(N), votes[s]), 1)
        return cnt.argmax(axis=1).astype(np.int64)
    elif strategy == "confidence_weighted":
        # Weight each model's probabilities by its confidence (max prob).
        max_c   = sub.max(axis=2, keepdims=True)
        weighted = (sub * max_c).sum(axis=0)
        return weighted.argmax(axis=1).astype(np.int64)
    raise ValueError(strategy)


def _bal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy: the mean of per-class recalls (true positive rates).
    This is the same as sklearn's balanced_accuracy_score but written inline to avoid
    the overhead of repeated calls during greedy search.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Float in [0, 1]. Returns 0.0 if no classes are present in y_true.
    """
    recalls = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.sum() > 0:
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _score(probs, y_true, subset, strategy, ms_arr, params_arr):
    """
    Evaluate an ensemble defined by `subset` of models using the given strategy.
    Computes accuracy, balanced accuracy, sequential/parallel latency, and total parameters.

    Args:
        probs: (M, N, K) softmax probability matrix.
        y_true: (N,) ground truth labels.
        subset: List of model indices forming the ensemble.
        strategy: Voting strategy passed to _ensemble_predict.
        ms_arr: (M,) array of per-model ms/sample latencies.
        params_arr: (M,) array of per-model parameter counts.

    Returns:
        Dict with accuracy, balanced_accuracy, ms_sequential, ms_parallel,
        total_params, subset, and strategy.
    """
    y_pred = _ensemble_predict(probs, subset, strategy)
    acc    = float((y_pred == y_true).mean())
    bal    = _bal_acc(y_true, y_pred)
    return {
        "accuracy": acc, "balanced_accuracy": bal,
        "ms_sequential": float(ms_arr[subset].sum()),
        "ms_parallel":   float(ms_arr[subset].max()),
        "total_params":  int(params_arr[subset].sum()),
        "subset": list(subset), "strategy": strategy,
    }


def _greedy_forward(probs, y_true, ms_arr, params_arr, strategy, metric):
    """
    Greedy forward selection: start with empty set, iteratively add the single model
    that gives the largest improvement in the chosen metric. Stop when the gain falls
    below MIN_GAIN_THRESHOLD or no models remain.

    Args:
        probs: (M, N, K) softmax probability matrix for all M models.
        y_true: (N,) ground truth labels.
        ms_arr: (M,) per-model ms/sample latencies.
        params_arr: (M,) per-model parameter counts.
        strategy: Voting strategy for _ensemble_predict.
        metric: Key name to optimise ("balanced_accuracy" or "accuracy").

    Returns:
        List of step dicts, each describing the ensemble after one model was added.
    """
    M = probs.shape[0]
    remaining, selected, steps = list(range(M)), [], []
    best_score = -1.0
    while remaining:
        bc, br = None, None
        for c in remaining:
            r = _score(probs, y_true, selected + [c], strategy, ms_arr, params_arr)
            if bc is None or r[metric] > br[metric]:
                bc, br = c, r
        if bc is None:
            break
        gain = br[metric] - best_score
        # Stop adding models if the metric gain is negligible.
        if selected and gain < MIN_GAIN_THRESHOLD:
            break
        best_score = br[metric]
        selected.append(bc)
        remaining.remove(bc)
        steps.append({**br, "added_model": bc})
    return steps


def _greedy_backward(probs, y_true, ms_arr, params_arr, strategy, metric):
    """
    Greedy backward elimination: start with all models, iteratively remove the single
    model whose removal causes the smallest drop in the chosen metric. Stop when the
    drop exceeds MAX_DROP_THRESHOLD or only one model remains.

    Args:
        probs: (M, N, K) softmax probability matrix for all M models.
        y_true: (N,) ground truth labels.
        ms_arr: (M,) per-model ms/sample latencies.
        params_arr: (M,) per-model parameter counts.
        strategy: Voting strategy (always "soft" in current usage).
        metric: Key name to optimise ("balanced_accuracy" or "accuracy").

    Returns:
        List of step dicts, each describing the ensemble after one model was removed.
    """
    selected = list(range(probs.shape[0]))
    steps    = []
    while len(selected) > 1:
        base = _score(probs, y_true, selected, strategy, ms_arr, params_arr)
        br, bt, bd = None, None, float("inf")
        for c in selected:
            trial = [m for m in selected if m != c]
            r     = _score(probs, y_true, trial, strategy, ms_arr, params_arr)
            drop  = base[metric] - r[metric]
            if drop < bd:
                bd, br, bt = drop, c, r
        # Stop removing if the next model would cause too large a metric drop.
        if bd > MAX_DROP_THRESHOLD:
            break
        selected.remove(br)
        steps.append({**bt, "removed_model": br})
    return steps


def _pareto(results, metric):
    """
    Compute the Pareto front from a list of ensemble results.
    An ensemble A dominates B if A has better or equal metric AND lower or equal
    sequential latency, with at least one strict improvement.

    Args:
        results: List of score dicts (from _score or greedy steps).
        metric: Key name used for the optimisation axis (e.g. "balanced_accuracy").

    Returns:
        Pareto-optimal ensembles sorted by ascending ms_sequential (fastest first).
    """
    front = []
    for r in results:
        dominated = any(
            o is not r and o[metric] >= r[metric] and o["ms_sequential"] <= r["ms_sequential"]
            and (o[metric] > r[metric] or o["ms_sequential"] < r["ms_sequential"])
            for o in results
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda x: x["ms_sequential"])


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f0f2f5; color: #1a1a2e; }
header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
         color: #fff; padding: 2rem 2.5rem; }
header h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: 0.02em; }
header .meta { margin-top: 0.4rem; font-size: 0.85rem; opacity: 0.75; }
.container { max-width: 1400px; margin: 0 auto; padding: 1.5rem 1rem; }
h2 { font-size: 1.3rem; font-weight: 700; color: #0f3460; margin: 2rem 0 0.8rem;
     border-left: 4px solid #e94560; padding-left: 0.7rem; }
h3 { font-size: 1.05rem; font-weight: 600; color: #16213e; margin: 1.2rem 0 0.5rem; }

/* ── Cards ────────────────────────────────── */
.card { background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08);
        padding: 1.2rem 1.4rem; margin-bottom: 1.2rem; }
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
             gap: 1rem; margin-bottom: 1.2rem; }
.stat-card { background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08);
             padding: 1rem 1.2rem; text-align: center; }
.stat-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em;
                    color: #666; margin-bottom: 0.3rem; }
.stat-card .value { font-size: 1.6rem; font-weight: 700; color: #0f3460; }
.stat-card .sub   { font-size: 0.78rem; color: #888; margin-top: 0.2rem; }

/* ── Tables ───────────────────────────────── */
.tbl-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th { background: #0f3460; color: #fff; padding: 0.55rem 0.8rem; text-align: left;
     white-space: nowrap; }
td { padding: 0.45rem 0.8rem; border-bottom: 1px solid #eee; vertical-align: middle; }
tr:nth-child(even) td { background: #f8f9fb; }
tr:hover td { background: #eef2ff; }
.num  { text-align: right; font-variant-numeric: tabular-nums; }
.good { color: #16a34a; font-weight: 600; }
.warn { color: #ca8a04; font-weight: 600; }
.bad  { color: #dc2626; font-weight: 600; }

/* ── Progress bars ────────────────────────── */
.bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
.bar-bg   { flex: 1; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width .3s; }

/* ── Confusion matrix grid ────────────────── */
.cm-grid { display: flex; flex-wrap: wrap; gap: 1.2rem; }
.cm-item { text-align: center; }
.cm-item img { border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,.12);
               max-width: 100%; height: auto; }
.cm-item .cm-label { font-size: 0.78rem; color: #555; margin-top: 0.35rem; }

/* ── Tabs ─────────────────────────────────── */
.tab-bar { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0; border-bottom: 2px solid #dee2e6; }
.tab-btn { padding: 0.55rem 1.1rem; border: none; background: none; cursor: pointer;
           font-size: 0.83rem; font-weight: 600; color: #555; border-radius: 6px 6px 0 0;
           border-bottom: 2px solid transparent; margin-bottom: -2px; }
.tab-btn.active { color: #0f3460; border-bottom-color: #e94560; background: #fff; }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* ── Ensemble boxes ───────────────────────── */
.ensemble-strategy { border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem;
                     margin-bottom: 1rem; background: #fafbfc; }
.ensemble-strategy h4 { font-size: 0.9rem; font-weight: 700; color: #0f3460; margin-bottom: 0.6rem; }
.pareto-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.4rem; align-items: center; }
.pareto-tag { background: #e0e7ff; color: #3730a3; border-radius: 4px; padding: 0.2rem 0.5rem;
              font-size: 0.75rem; font-weight: 600; }
.rec-box { background: linear-gradient(135deg, #0f3460, #e94560); color: #fff;
           border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem; }
.rec-box h3 { color: #fff; margin-bottom: 0.6rem; }
.rec-box .rec-models { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.rec-box .rec-model  { background: rgba(255,255,255,.2); border-radius: 4px;
                        padding: 0.2rem 0.6rem; font-size: 0.8rem; }
.badge { display: inline-block; border-radius: 4px; padding: 0.15rem 0.5rem;
         font-size: 0.72rem; font-weight: 700; }
.badge-blue   { background: #dbeafe; color: #1d4ed8; }
.badge-green  { background: #dcfce7; color: #15803d; }
.badge-orange { background: #ffedd5; color: #c2410c; }
.toc { display: block; background: #fff; border-radius: 10px; padding: 1rem 1.4rem;
       margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
.toc a { color: #0f3460; text-decoration: none; font-size: 0.85rem; }
.toc a:hover { text-decoration: underline; }
"""

_JS = """
function showTab(groupId, tabId) {
    document.querySelectorAll('#' + groupId + ' .tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('#' + groupId + ' .tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}
"""


def _color_class(value: float) -> str:
    """
    Return a CSS class name for color-coding metric values in the HTML report.
    >= 0.95 → "good" (green), >= 0.80 → "warn" (yellow), else → "bad" (red).
    """
    if value >= 0.95: return "good"
    if value >= 0.80: return "warn"
    return "bad"


def _barOLD(value: float, color: str = "#3b82f6") -> str:
    """
    Render a horizontal progress bar for values in [0, 1]. (Legacy — superseded by _bar.)
    Clamps value to [0, 1] before computing percentage width.
    """
    pct = max(0.0, min(1.0, value)) * 100
    return (f'<div class="bar-wrap">'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<span style="min-width:3.5em;font-size:.8rem;text-align:right">{value:.4f}</span>'
            f'</div>')


def _bar(value: float, color: str = "#3b82f6", max_val: float = 1.0) -> str:
    """
    Render a horizontal progress bar scaled relative to max_val.
    Used for metrics in [0, 1] (accuracy) and for FPS (scaled to max FPS across models).

    Args:
        value: The value to display.
        color: CSS color for the bar fill.
        max_val: The upper bound for scaling (1.0 for accuracy, max FPS for throughput).
    """
    pct = (max(0.0, value) / max_val) * 100 if max_val > 0 else 0
    pct = min(100.0, pct) # Clamp to 100%
    return (f'<div class="bar-wrap">'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<span style="min-width:3.5em;font-size:.8rem;text-align:right">{value:.2f}</span>'
            f'</div>')


def _fmt_params(n: int) -> str:
    """Format a parameter count as a human-readable string (e.g. 12.3M, 456K, 789)."""
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Main HTML assembly
# ---------------------------------------------------------------------------

def _build_html(meta: dict, model_results: list[dict], ensemble_results: dict,
                dataset_classes: list[str], generated: str,
                efficiency_b64: str = "") -> str:
    """
    Assemble the complete self-contained HTML report from analysis results.

    The report includes:
      - Header with dataset/model summary
      - Overview cards (best accuracy, best balanced acc, fastest, lightest)
      - Per-model comparison table with progress bars
      - Per-class detail tabs (precision/recall/F1/TP/FP/FN)
      - Confusion matrix images (base64-embedded)
      - Ensemble optimisation: forward selection, backward elimination, Pareto front
      - Best and lightest ensemble recommendations
      - Throughput benchmark 3-D plots (if available)

    Args:
        meta: Dict with dataset path, num_samples, num_classes, num_models, fp16 flag.
        model_results: List of per-model result dicts (from the main evaluation loop).
        ensemble_results: Dict with forward/pareto/backward steps and recommendations.
        dataset_classes: Ordered list of class names.
        generated: Timestamp string displayed in the header and footer.

    Returns:
        A complete HTML document as a string.
    """

    model_names = [m["name"] for m in model_results]

    # ── header ──────────────────────────────────────────────────────────────
    head = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Analysis — {Path(meta['dataset']).name}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>&#128202; Model Analysis Report</h1>
  <div class="meta">
    Dataset: <strong>{meta['dataset']}</strong> &nbsp;|&nbsp;
    Samples: <strong>{meta['num_samples']:,}</strong> &nbsp;|&nbsp;
    Classes: <strong>{meta['num_classes']}</strong> &nbsp;|&nbsp;
    Models: <strong>{meta['num_models']}</strong> &nbsp;|&nbsp;
    Generated: <strong>{generated}</strong>
    {' &nbsp;|&nbsp; Precision: <span style="color:#fbbf24">FP16</span>' if meta.get('fp16') else ''}
  </div>
</header>
<div class="container">
"""

    # ── table of contents ────────────────────────────────────────────────────
    _has_benchmark = any(m.get("benchmark_b64") for m in model_results)
    toc_items = ["<li><a href='#overview'>Overview</a></li>",
                 "<li><a href='#models'>Per-model results</a></li>",
                 "<li><a href='#confusion'>Confusion matrices</a></li>",
                 "<li><a href='#efficiency'>Efficiency comparison</a></li>",
                 "<li><a href='#ensemble'>Ensemble optimisation</a></li>"]
    if _has_benchmark:
        toc_items.append("<li><a href='#benchmark'>Throughput benchmark</a></li>")
    head += f"""
<div class="toc"><strong>Contents</strong><ol style="margin-top:.5rem;padding-left:1.4rem">
{''.join(toc_items)}</ol></div>
"""

    # ── overview cards ───────────────────────────────────────────────────────
    best_acc    = max(m["accuracy"]          for m in model_results)
    best_bal    = max(m["balanced_accuracy"] for m in model_results)
    fastest     = min(m["ms_per_sample"]     for m in model_results)
    lightest    = min(m["num_params"]        for m in model_results)
    best_acc_n  = model_names[np.argmax([m["accuracy"]          for m in model_results])]
    best_bal_n  = model_names[np.argmax([m["balanced_accuracy"] for m in model_results])]
    fastest_n   = model_names[np.argmin([m["ms_per_sample"]     for m in model_results])]
    lightest_n  = model_names[np.argmin([m["num_params"]        for m in model_results])]

    section_overview = f"""
<h2 id="overview">&#127775; Overview</h2>
<div class="card-grid">
  <div class="stat-card"><div class="label">Best Accuracy</div>
    <div class="value">{best_acc:.4f}</div><div class="sub">{best_acc_n}</div></div>
  <div class="stat-card"><div class="label">Best Balanced Acc</div>
    <div class="value">{best_bal:.4f}</div><div class="sub">{best_bal_n}</div></div>
  <div class="stat-card"><div class="label">Fastest Model</div>
    <div class="value">{fastest:.3f} ms</div><div class="sub">{fastest_n}</div></div>
  <div class="stat-card"><div class="label">Lightest Model</div>
    <div class="value">{_fmt_params(lightest)}</div><div class="sub">{lightest_n}</div></div>
  <div class="stat-card"><div class="label">Dataset Classes</div>
    <div class="value">{meta['num_classes']}</div>
    <div class="sub">{'  ·  '.join(dataset_classes)}</div></div>
  <div class="stat-card"><div class="label">Random Baseline (BalAcc)</div>
    <div class="value">{1/meta['num_classes']:.4f}</div>
    <div class="sub">1 / {meta['num_classes']} classes</div></div>
</div>
"""

    # ── per-model table ──────────────────────────────────────────────────────
    rows = ""
    # Calculate max_fps here 
    max_fps = max([m.get("fps", 1.0) for m in model_results])
    for m in sorted(model_results, key=lambda x: -x["balanced_accuracy"]):
        cba  = _color_class(m["balanced_accuracy"])
        cacc = _color_class(m["accuracy"])
        rows += f"""<tr>
  <td><strong>{m['name']}</strong></td>
  <td><span class="badge badge-blue">{m['model_type']}</span></td>
  <td class="num">{m['tile_size']}</td>
  <td class="num">{_fmt_params(m['num_params'])}</td>
  <td>{_bar(m['accuracy'])}</td>
  <td>{_bar(m['balanced_accuracy'], '#10b981')}</td>
  <td>{_bar(m['fps'], '#f59e0b', max_val=max_fps)}</td>
  <td class="num">{m['ms_per_sample']:.3f}</td>
  <td class="num">{m['num_aligned_samples']:,}</td>
</tr>"""

    section_models = f"""
<h2 id="models">&#127919; Per-model results</h2>
<div class="card"><div class="tbl-wrap">
<table>
<thead><tr>
  <th>Model</th><th>Type</th><th>Tile</th><th>Params</th>
  <th style="min-width:160px">Accuracy</th>
  <th style="min-width:160px">Balanced Acc</th>
  <th>Effective FPS</th><th>ms/sample</th><th>Samples</th>
</tr></thead>
<tbody>{rows}</tbody>
</table></div></div>
"""

    # per-model per-class detail (collapsible tabs)
    per_model_detail = ""
    group_id = "model_tabs"
    tab_bar  = '<div class="tab-bar">'
    tab_panes = ""
    for idx, m in enumerate(sorted(model_results, key=lambda x: -x["balanced_accuracy"])):
        tid    = f"mt_{idx}"
        active = "active" if idx == 0 else ""
        tab_bar  += f'<button class="tab-btn {active}" onclick="showTab(\'{group_id}\',\'{tid}\')">{m["name"].replace("allclass_","").replace("binary_","⚡ ")}</button>'
        # per-class table
        pc_rows = ""
        for pc in m["per_class"]:
            cp = _color_class(pc["precision"])
            cr = _color_class(pc["recall"])
            cf = _color_class(pc["f1"])
            pc_rows += f"""<tr>
  <td>{pc['class_name']}</td>
  <td class="num">{pc['support']:,}</td>
  <td class="num {cp}">{pc['precision']:.4f}</td>
  <td class="num {cr}">{pc['recall']:.4f}</td>
  <td class="num {cf}">{pc['f1']:.4f}</td>
  <td class="num">{pc['tp']:,}</td>
  <td class="num">{pc['fp']:,}</td>
  <td class="num">{pc['fn']:,}</td>
</tr>"""
        tab_panes += f"""<div id="{tid}" class="tab-content {active}">
<div class="tbl-wrap"><table>
<thead><tr><th>Class</th><th>Support</th><th>Precision</th>
<th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
<tbody>{pc_rows}</tbody></table></div></div>"""
    tab_bar += "</div>"
    per_model_detail = f"""
<h3>Per-class breakdown</h3>
<div class="card" id="{group_id}">
{tab_bar}
{tab_panes}
</div>"""
    section_models += per_model_detail

    # ── confusion matrices ───────────────────────────────────────────────────
    cm_html = ""
    for m in sorted(model_results, key=lambda x: -x["balanced_accuracy"]):
        if m.get("cm_b64"):
            cm_html += f"""<div class="cm-item">
  <img src="data:image/png;base64,{m['cm_b64']}" alt="CM {m['name']}" width="360">
  <div class="cm-label">{m['name']}<br>BalAcc&nbsp;{m['balanced_accuracy']:.4f}</div>
</div>"""

    section_cm = f"""
<h2 id="confusion">&#129300; Confusion matrices (row-normalised)</h2>
<div class="card"><div class="cm-grid">{cm_html}</div></div>
"""

    # ── efficiency section ───────────────────────────────────────────────────
    section_eff = ""
    if efficiency_b64:
        section_eff = f"""
<h2 id="efficiency">&#9878; Efficiency comparison (Accuracy / Params)</h2>
<div class="card">
  <p style="font-size:.82rem;color:#555;margin-bottom:.8rem">
    <strong>Top-left</strong>: balanced accuracy vs model size — points toward the top-left
    corner are Pareto-optimal (more accurate <em>and</em> smaller).
    <strong>Top-right</strong>: overall efficiency = balanced accuracy per million parameters,
    higher is better for a given accuracy level.
    <strong>Bottom</strong>: per-class efficiency = recall per million parameters for each class
    across all models — identifies which model extracts the most signal per parameter for each class.
  </p>
  <img src="data:image/png;base64,{efficiency_b64}" alt="Efficiency comparison"
       style="max-width:100%;height:auto;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,.12)">
</div>
"""

    # ── ensemble section ─────────────────────────────────────────────────────
    rec      = ensemble_results.get("recommendation", {})
    rec_min  = ensemble_results.get("recommendation_min_params", {})
    rec_names = [model_names[i] for i in rec.get("subset", [])]
    rmp_names = [model_names[i] for i in rec_min.get("subset", [])]

    rec_box = f"""
<div class="rec-box">
  <h3>&#127942; Best overall ensemble ({rec.get('strategy','—')} voting)</h3>
  <div style="display:flex;flex-wrap:wrap;gap:2rem;margin-top:.5rem">
    <div><div class="label" style="opacity:.7;font-size:.72rem">ACCURACY</div>
         <div style="font-size:1.4rem;font-weight:700">{rec.get('accuracy',0):.4f}</div></div>
    <div><div class="label" style="opacity:.7;font-size:.72rem">BALANCED ACC</div>
         <div style="font-size:1.4rem;font-weight:700">{rec.get('balanced_accuracy',0):.4f}</div></div>
    <div><div class="label" style="opacity:.7;font-size:.72rem">ms/sample (seq)</div>
         <div style="font-size:1.4rem;font-weight:700">{rec.get('ms_sequential',0):.3f}</div></div>
    <div><div class="label" style="opacity:.7;font-size:.72rem">ms/sample (par)</div>
         <div style="font-size:1.4rem;font-weight:700">{rec.get('ms_parallel',0):.3f}</div></div>
    <div><div class="label" style="opacity:.7;font-size:.72rem">TOTAL PARAMS</div>
         <div style="font-size:1.4rem;font-weight:700">{_fmt_params(rec.get('total_params',0))}</div></div>
  </div>
  <div class="rec-models">{''.join(f"<span class='rec-model'>{n}</span>" for n in rec_names)}</div>
</div>
"""
    if rmp_names and rmp_names != rec_names:
        savings = rec.get("total_params", 0) - rec_min.get("total_params", 0)
        rec_box += f"""
<div class="rec-box" style="background:linear-gradient(135deg,#064e3b,#065f46)">
  <h3>&#9889; Lightest ensemble within 0.5% of best ({rec_min.get('strategy','—')} voting)</h3>
  <div style="display:flex;flex-wrap:wrap;gap:2rem;margin-top:.5rem">
    <div><div class="label" style="opacity:.7;font-size:.72rem">BALANCED ACC</div>
         <div style="font-size:1.4rem;font-weight:700">{rec_min.get('balanced_accuracy',0):.4f}</div></div>
    <div><div class="label" style="opacity:.7;font-size:.72rem">PARAM SAVINGS</div>
         <div style="font-size:1.4rem;font-weight:700">{_fmt_params(savings)}</div></div>
    <div><div class="label" style="opacity:.7;font-size:.72rem">ms/sample (par)</div>
         <div style="font-size:1.4rem;font-weight:700">{rec_min.get('ms_parallel',0):.3f}</div></div>
  </div>
  <div class="rec-models">{''.join(f"<span class='rec-model'>{n}</span>" for n in rmp_names)}</div>
</div>
"""

    # forward selection tables per strategy
    fwd_html = ""
    for strategy in STRATEGIES:
        steps = ensemble_results.get("forward", {}).get(strategy, [])
        if not steps:
            continue
        fwd_rows = ""
        for i, s in enumerate(steps):
            names = ", ".join(model_names[idx] for idx in s["subset"])
            fwd_rows += f"""<tr>
  <td class="num">{i+1}</td>
  <td><strong>{model_names[s['added_model']]}</strong></td>
  <td>{_bar(s['accuracy'])}</td>
  <td>{_bar(s['balanced_accuracy'], '#10b981')}</td>
  <td class="num">{s['ms_sequential']:.3f}</td>
  <td class="num">{s['ms_parallel']:.3f}</td>
  <td class="num">{_fmt_params(s['total_params'])}</td>
</tr>"""
        fwd_html += f"""<div class="ensemble-strategy">
<h4>Forward selection — {strategy}</h4>
<div class="tbl-wrap"><table>
<thead><tr><th>#</th><th>Added model</th><th style="min-width:140px">Accuracy</th>
<th style="min-width:140px">BalAcc</th><th>ms-seq</th><th>ms-par</th><th>Params</th></tr></thead>
<tbody>{fwd_rows}</tbody></table></div>"""

        # Pareto front
        front = ensemble_results.get("pareto", {}).get(strategy, [])
        if front:
            fwd_html += '<div style="margin-top:.7rem"><strong style="font-size:.82rem">Pareto front (BalAcc vs latency):</strong>'
            for p in front:
                pnames = ", ".join(model_names[i] for i in p["subset"])
                fwd_html += f"""<div class="pareto-row" style="margin-top:.4rem">
  <span class="pareto-tag">{len(p['subset'])} model{'s' if len(p['subset'])>1 else ''}</span>
  <span style="font-size:.8rem">BalAcc&nbsp;<strong>{p['balanced_accuracy']:.4f}</strong>
  &nbsp;·&nbsp;{p['ms_sequential']:.3f}&nbsp;ms-seq
  &nbsp;·&nbsp;{p['ms_parallel']:.3f}&nbsp;ms-par</span>
  <span style="font-size:.76rem;color:#555">{pnames}</span>
</div>"""
            fwd_html += "</div>"
        fwd_html += "</div>"

    # backward elimination table
    bwd_steps = ensemble_results.get("backward", [])
    bwd_rows  = ""
    for i, s in enumerate(bwd_steps):
        bwd_rows += f"""<tr>
  <td class="num">{i+1}</td>
  <td>{model_names[s['removed_model']]}</td>
  <td>{_bar(s['accuracy'])}</td>
  <td>{_bar(s['balanced_accuracy'], '#10b981')}</td>
  <td class="num">{s['ms_sequential']:.3f}</td>
  <td class="num">{_fmt_params(s['total_params'])}</td>
</tr>"""
    bwd_html = ""
    if bwd_rows:
        surviving = [model_names[i] for i in bwd_steps[-1]["subset"]] if bwd_steps else model_names
        bwd_html  = f"""<div class="ensemble-strategy">
<h4>Backward elimination — soft voting &nbsp;<span style="font-weight:400;font-size:.8rem">(surviving: {', '.join(surviving)})</span></h4>
<div class="tbl-wrap"><table>
<thead><tr><th>#</th><th>Removed model</th>
<th style="min-width:140px">Accuracy</th><th style="min-width:140px">BalAcc</th>
<th>ms-seq</th><th>Params</th></tr></thead>
<tbody>{bwd_rows}</tbody></table></div></div>"""

    section_ens = f"""
<h2 id="ensemble">&#129520; Ensemble optimisation</h2>
{rec_box}
{fwd_html}
{bwd_html}
"""

    # ── throughput benchmark section ─────────────────────────────────────────
    bench_imgs = ""
    for m in sorted(model_results, key=lambda x: -x["balanced_accuracy"]):
        if m.get("benchmark_b64"):
            bench_imgs += f"""<div class="cm-item">
  <img src="data:image/png;base64,{m['benchmark_b64']}" alt="Benchmark {m['name']}" width="480">
  <div class="cm-label">{m['name']}<br>BalAcc&nbsp;{m['balanced_accuracy']:.4f}</div>
</div>"""

    section_bench = ""
    if bench_imgs:
        section_bench = f"""
<h2 id="benchmark">&#9889; Throughput benchmark (batch &times; step sweep)</h2>
<div class="card">
  <p style="font-size:.82rem;color:#555;margin-bottom:.8rem">
    Each point is one <strong>(batch&nbsp;size,&nbsp;step&nbsp;size)</strong> combination.
    <strong>Step&nbsp;size</strong> = tile stride when scanning the live image —
    smaller step = denser coverage, more tiles, lower FPS.
    <strong>Z-axis (FPS)</strong> = 1&nbsp;/&nbsp;(⌈num_tiles&nbsp;/&nbsp;batch_size⌉&nbsp;×&nbsp;time_per_batch),
    where num_tiles is computed analytically from the image resolution and tile size.
    Batch timing is measured once per batch size and reused across all steps.
    <strong>Colour</strong> = balanced accuracy (constant per model — the per-tile classifier
    accuracy does not change with tiling stride).
  </p>
  <div class="cm-grid">{bench_imgs}</div>
</div>
"""

    footer = f"""
<p style="text-align:center;font-size:.75rem;color:#999;margin:2rem 0 1rem">
  Generated by ModelAnalysis.py &nbsp;·&nbsp; {generated}
</p>
</div>
<script>{_JS}</script>
</body></html>"""

    return head + section_overview + section_models + section_cm + section_eff + section_ens + section_bench + footer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Entry point: parse args, validate inputs, run per-model evaluation, ensemble optimisation,
    and produce a self-contained HTML report + JSON summary.

    Pipeline:
      1. Validate dataset directory and discover (.pth, .json) model pairs.
      2. For each model: instantiate, load weights, run inference, compute metrics,
         save confusion matrix PNG, run throughput benchmark.
      3. Align all model probabilities to a common dataset class space.
      4. Run greedy forward selection + backward elimination ensembles under all 3 voting strategies.
      5. Select best and lightest ensemble recommendations.
      6. Save model_analysis.json and index.html to the output directory.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("models_dir", nargs="?", default=".")
    ap.add_argument("--batch",         type=int, default=64)
    ap.add_argument("--fp16",          action="store_true")
    ap.add_argument("--out",           default=None)
    ap.add_argument("--metric",        default=DEFAULT_METRIC,
                    choices=["balanced_accuracy", "accuracy"])
    ap.add_argument("--no-bench",       action="store_true",
                    help="Skip the batch×step throughput benchmark")
    ap.add_argument("--bench-batches",  default=None,
                    help="Comma-separated batch sizes for benchmark (default: 1,2,4,8,16,32,64)")
    ap.add_argument("--bench-steps",    default=None,
                    help="Comma-separated tile step sizes for benchmark (default: 1,2,4,8,16)")
    ap.add_argument("--image-width",    type=int, default=1224,
                    help="Live image width used to compute tile counts (default: 1224)")
    ap.add_argument("--image-height",   type=int, default=1024,
                    help="Live image height used to compute tile counts (default: 1024)")
    args = ap.parse_args()

    bench_batch_sizes = (
        [int(x) for x in args.bench_batches.split(",")]
        if args.bench_batches else BENCHMARK_BATCH_SIZES
    )
    bench_step_sizes = (
        [int(x) for x in args.bench_steps.split(",")]
        if args.bench_steps else BENCHMARK_STEP_SIZES
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(args.out) if args.out else Path(f"model_analysis_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model Analysis")
    print(f"{'='*60}")
    print(f"  Dataset    : {args.dataset_dir}")
    print(f"  Models dir : {args.models_dir}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {args.batch}")
    print(f"  FP16       : {args.fp16}")
    print(f"  Output     : {out_dir}")
    print(f"  Metric     : {args.metric}")

    # ── validate dataset ─────────────────────────────────────────────────────
    if not is_valid_dataset_dir(args.dataset_dir):
        print(f"\n[ERROR] Not a valid dataset directory: {args.dataset_dir}")
        sys.exit(1)

    # ── discover models ──────────────────────────────────────────────────────
    pairs = _discover_pairs(args.models_dir)
    if not pairs:
        print(f"[ERROR] No (.pth, .json) pairs found in: {args.models_dir}")
        sys.exit(1)
    print(f"\n  Found {len(pairs)} model(s):")
    for pth, _ in pairs:
        print(f"    {Path(pth).name}")

    # ── load dataset ─────────────────────────────────────────────────────────
    print("\n  Loading dataset...")
    dataset            = _load_dataset(args.dataset_dir)
    dataset_classes    = dataset.classes
    num_dataset_classes = len(dataset_classes)
    print(f"  Classes ({num_dataset_classes}): {dataset_classes}")
    print(f"  Samples: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size  = args.batch,
        shuffle     = False,
        num_workers = min(8, os.cpu_count() or 1),
        drop_last   = False,
        collate_fn  = metadata_collate_fn,
    )

    # ── per-model evaluation ─────────────────────────────────────────────────
    model_results: list[dict] = []
    probs_all:     list[np.ndarray] = []
    y_true_global: np.ndarray | None = None

    for midx, (pth_path, cfg_path) in enumerate(pairs):
        model_name = Path(pth_path).stem
        print(f"\n  [{midx+1}/{len(pairs)}] {model_name}")

        config_json   = load_hyperparameters(cfg_path)
        model_classes = config_json.get("classes", dataset_classes)

        eval_to_model, shared, miss_eval, miss_model = build_eval_to_model_mapping(
            dataset_classes, model_classes
        )
        if miss_eval:
            print(f"    [WARN] dataset classes absent from model: {miss_eval}")
        if miss_model:
            print(f"    [WARN] model classes absent from dataset: {miss_model}")
        if not eval_to_model:
            print("    [ERROR] zero class overlap — skipping.")
            continue

        # Catch-all: binary models map dataset-exclusive classes → their "defect" output
        catchall_model_indices: list[int] = []
        catchall_ds_indices:    list[int] = []
        if miss_model and miss_eval:
            catchall_model_indices = [model_classes.index(n) for n in miss_model]
            catchall_ds_indices    = [dataset_classes.index(n) for n in miss_eval]
            primary_catchall       = catchall_model_indices[0]
            for ds_name in miss_eval:
                eval_to_model[dataset_classes.index(ds_name)] = primary_catchall
            print(f"    [INFO] Catch-all: {len(miss_eval)} dataset classes → '{miss_model[0]}'")

        try:
            clf = _instantiate(config_json, model_classes)
            clf = _load_weights(clf, pth_path, device)
        except Exception as e:
            print(f"    [ERROR] Could not load model: {e}")
            continue

        num_params = _count_params(clf)

        y_true_ds, y_pred_model, probs_model, ms = _run_model(
            clf, loader, device, eval_to_model,
            catchall_model_indices, catchall_ds_indices,
            args.fp16, model_name,
        )

        if len(y_true_ds) == 0:
            print("    [WARN] No aligned samples — skipping.")
            del clf; torch.cuda.empty_cache() if device == "cuda" else None
            continue

        # ── Probability alignment to dataset class space ────────────────────
        # Models may have been trained on a different class set than the dataset:
        #   • "allclass" models: trained on the full class set → 1-to-1 mapping.
        #   • "binary" models: trained on {clean, defect} where "defect" is a
        #     catch-all for every defect type the dataset distinguishes.
        #
        # For shared classes, copy the model's softmax column directly.
        # For catch-all model classes (miss_model), distribute their probability
        # equally across the dataset classes they represent (miss_eval).  Equal
        # splitting is a conservative assumption — it preserves total probability
        # mass while treating all defect sub-types as equally likely.
        N = len(y_true_ds)
        aligned = np.zeros((N, num_dataset_classes), dtype=np.float32)
        for mi, mn in enumerate(model_classes):
            if mn in dataset_classes:
                # Direct 1-to-1 class mapping.
                aligned[:, dataset_classes.index(mn)] = probs_model[:, mi]
            elif mi in catchall_model_indices and catchall_ds_indices:
                # Catch-all: distribute this model output's probability equally
                # across all dataset classes that map to this catch-all bucket.
                share = probs_model[:, mi] / len(catchall_ds_indices)
                for di in catchall_ds_indices:
                    aligned[:, di] += share

        # Re-argmax in dataset space so metrics are computed on the right labels.
        y_pred_aligned = aligned.argmax(axis=1).astype(np.int64)

        if y_true_global is None:
            y_true_global = y_true_ds
        elif not np.array_equal(y_true_global, y_true_ds):
            print("    [WARN] y_true ordering differs from previous models.")

        # Compute statistics in dataset class space
        st = _stats(y_true_ds, y_pred_aligned, dataset_classes)
        print(f"    Accuracy    : {st['accuracy']:.4f}")
        print(f"    BalAcc      : {st['balanced_accuracy']:.4f}")
        print(f"    ms/sample   : {ms:.4f}")
        print(f"    Params      : {num_params:,}")

        # Confusion matrix image
        cm_path = out_dir / f"{model_name}_confusion_row_normalized.png"
        _save_confusion_png(st["cm_norm"], dataset_classes,
                            f"{model_name}  (BalAcc {st['balanced_accuracy']:.4f})",
                            str(cm_path))
        cm_b64 = _png_to_b64(str(cm_path))

        # Benchmark: sweep batch_size × step_size
        bench_data: list[dict] = []
        bench_b64:  str        = ""
        if not args.no_bench:
            bench_data = run_throughput_benchmark(
                clf, dataset, eval_to_model,
                batch_sizes=bench_batch_sizes,
                step_sizes=bench_step_sizes,
                tile_size=config_json["hparams"].get("tile_size", 48),
                image_width=args.image_width,
                image_height=args.image_height,
                device=device,
                precomputed_accuracy=st["accuracy"],
                precomputed_bal_accuracy=st["balanced_accuracy"],
                # Match live pipeline precision so tpb reflects actual GPU cost.
                use_fp16=args.fp16,
            )
            bench_path = out_dir / f"{model_name}_benchmark_3d.png"
            _save_benchmark_3d_plot(bench_data, model_name, str(bench_path))
            bench_b64 = _png_to_b64(str(bench_path))
            ok_pts  = [r for r in bench_data if not r.get("oom")]
            oom_pts = [r for r in bench_data if r.get("oom")]
            if oom_pts:
                oom_batches = sorted({r["batch_size"] for r in oom_pts})
                print(f"    OOM batch sizes: {oom_batches}")
            if ok_pts:
                best_pt = max(ok_pts, key=lambda r: r["effective_fps"])
                print(f"    Peak eff. FPS : {best_pt['effective_fps']:.1f} "
                      f"(batch={best_pt['batch_size']}, step={best_pt['step_size']})")

        model_results.append({
            "name"               : model_name,
            "model_type"         : config_json.get("model", "unknown"),
            "tile_size"          : config_json["hparams"].get("tile_size", 0),
            "num_params"         : num_params,
            "ms_per_sample"      : ms,
            "accuracy"           : st["accuracy"],
            "balanced_accuracy"  : st["balanced_accuracy"],
            "num_aligned_samples": N,
            "per_class"          : st["per_class"],
            "cm"                 : st["cm"],
            "cm_norm"            : st["cm_norm"],
            "cm_b64"             : cm_b64,
            "benchmark"          : bench_data,
            "benchmark_b64"      : bench_b64,
        })
        probs_all.append(aligned)

        del clf
        if device == "cuda":
            torch.cuda.empty_cache()

    # Derive Hz (frames/samples per second) from the clean GPU timing measurement.
    # ms_per_sample is PCIe + GPU forward only (no DataLoader I/O) from Pass 2
    # of _run_model, so fps represents true GPU inference throughput.
    # max_fps is used to scale the relative bar widths in the HTML table.
    for m in model_results:
        m["fps"] = 1000.0 / m["ms_per_sample"] if m["ms_per_sample"] > 0 else 0.0
    max_fps = max((m["fps"] for m in model_results), default=1.0)


    if not probs_all:
        print("\n[ERROR] No models produced valid predictions.")
        sys.exit(1)

    # ── Ensemble optimisation ────────────────────────────────────────────────
    # Stack per-model probability arrays into a single (M, N, K) matrix where:
    #   M = number of models, N = number of aligned samples, K = dataset classes.
    # Greedy forward selection adds models one at a time, keeping each that
    # improves the target metric by > MIN_GAIN_THRESHOLD.
    # Greedy backward elimination starts with all models and removes them as long
    # as the metric drop stays below MAX_DROP_THRESHOLD.
    # ms_arr feeds into ensemble latency estimates:
    #   ms_sequential = sum(ms_arr[subset])  — models run one after another
    #   ms_parallel   = max(ms_arr[subset])  — models run on separate CUDA streams
    print(f"\n  Running ensemble optimisation ({args.metric})...")
    M            = len(probs_all)
    probs_matrix = np.stack(probs_all, axis=0).astype(np.float32)   # (M, N, K)
    ms_arr       = np.array([m["ms_per_sample"] for m in model_results], dtype=np.float64)
    params_arr   = np.array([m["num_params"]     for m in model_results], dtype=np.int64)
    metric       = args.metric

    all_forward: dict[str, list] = {}
    all_pareto:  dict[str, list] = {}

    for strategy in STRATEGIES:
        print(f"    Forward selection ({strategy})...")
        steps = _greedy_forward(probs_matrix, y_true_global, ms_arr, params_arr, strategy, metric)
        all_forward[strategy] = steps
        all_pareto[strategy]  = _pareto(steps, metric)

    print("    Backward elimination (soft)...")
    bwd_steps = _greedy_backward(probs_matrix, y_true_global, ms_arr, params_arr, "soft", metric)

    # Best overall: highest metric, breaking ties by lowest sequential latency.
    all_candidates = [s for steps in all_forward.values() for s in steps]
    best       = max(all_candidates, key=lambda x: (x[metric], -x["ms_sequential"]))
    best_names = [model_results[i]["name"] for i in best["subset"]]

    # Lightest ensemble within 0.5% of best accuracy: useful for deployment on
    # edge hardware where parameter count (model size / memory) is the constraint.
    threshold    = best[metric] * 0.995
    lw_cands     = [s for s in all_candidates if s[metric] >= threshold and s["total_params"] > 0]
    lightest_ens = min(lw_cands, key=lambda x: (x["total_params"], -x[metric])) if lw_cands else best
    lw_names     = [model_results[i]["name"] for i in lightest_ens["subset"]]

    ensemble_results = {
        "forward" : all_forward,
        "pareto"  : all_pareto,
        "backward": bwd_steps,
        "recommendation": {
            **best,
            "model_names": best_names,
        },
        "recommendation_min_params": {
            **lightest_ens,
            "model_names": lw_names,
        },
    }

    print(f"\n  Best ensemble ({best['strategy']}):")
    print(f"    BalAcc  : {best[metric]:.4f}")
    print(f"    ms-par  : {best['ms_parallel']:.3f}")
    print(f"    Models  : {', '.join(best_names)}")

    # ── Save JSON summary ────────────────────────────────────────────────────
    json_out = out_dir / "model_analysis.json"
    summary  = {
        "generated"   : datetime.datetime.now().isoformat(),
        "dataset"     : args.dataset_dir,
        "num_samples" : int(len(y_true_global)),
        "num_classes" : num_dataset_classes,
        "classes"     : dataset_classes,
        "fp16"        : args.fp16,
        "models"      : [{k: v for k, v in m.items() if k not in ("cm_b64", "benchmark_b64")}
                         for m in model_results],
        "ensemble"    : {
            k: v for k, v in ensemble_results.items() if k != "forward"
        },
    }
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON saved: {json_out}")

    # ── Build and save HTML ──────────────────────────────────────────────────
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_info = {
        "dataset"    : args.dataset_dir,
        "num_samples": int(len(y_true_global)),
        "num_classes": num_dataset_classes,
        "num_models" : M,
        "fp16"       : args.fp16,
    }

    efficiency_b64 = ""
    if len(model_results) >= 1:
        eff_path = out_dir / "efficiency_comparison.png"
        _save_efficiency_plot(model_results, dataset_classes, str(eff_path))
        efficiency_b64 = _png_to_b64(str(eff_path))

    html = _build_html(meta_info, model_results, ensemble_results,
                       dataset_classes, generated, efficiency_b64)
    html_out = out_dir / "index.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  HTML saved: {html_out}")
    print(f"\n{'='*60}")
    print(f"  Open: {html_out.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
