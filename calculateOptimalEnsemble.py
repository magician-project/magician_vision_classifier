#!/usr/bin/python3
"""
calculateOptimalEnsemble.py

Run every (.pth, .json) model pair found in a directory against a labelled
dataset.  For each model collect:
  - Full per-sample softmax probability vectors
  - Ground-truth labels
  - Inference timing (ms / sample, measured after a warmup pass)
  - Per-model classification statistics (accuracy, balanced accuracy,
    per-class TP / FP / TN / FN, precision, recall, F1)

Outputs (written to the current directory):
  ensemble_<dataset>_<timestamp>.json   — human-readable metadata + stats
  ensemble_<dataset>_<timestamp>.npz    — compact arrays:
        y_true  : (N,)       int64
        probs   : (M, N, K)  float32   M models · N samples · K classes

The .npz is referenced by the .json so evaluateOptimalEnsemble.py can load
both with a single path argument.

Usage:
    python3 calculateOptimalEnsemble.py <dataset_dir> [models_dir] [batch_size]

    dataset_dir : directory with ImageFolder-style class sub-dirs or dataset.h5
    models_dir  : directory containing *.pth + matching *.json  (default: ".")
    batch_size  : inference batch size (default: 64)

Example:
    python3 calculateOptimalEnsemble.py /data/MagicianWeldings/ . 64
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
)

try:
    from DatasetConverter import HDF5Dataset
    _HAS_HDF5 = True
except Exception:
    _HAS_HDF5 = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _discover_model_pairs(models_dir: str) -> list[tuple[str, str]]:
    """
    Find all (.pth, matching .json) pairs in *models_dir*.
    Files without a same-stem .json are skipped with a warning.
    """
    pairs = []
    for f in sorted(Path(models_dir).glob("*.pth")):
        cfg = f.with_suffix(".json")
        if cfg.is_file():
            pairs.append((str(f), str(cfg)))
        else:
            print(f"[WARN] No matching .json for {f.name} — skipping.")
    return pairs


def _load_dataset(dataset_dir: str):
    """
    Load the dataset without any class filtering so all models are compared
    on the same full label space.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    h5 = os.path.join(dataset_dir, "dataset.h5")
    if _HAS_HDF5 and os.path.isfile(h5):
        print(f"Loading HDF5 dataset: {h5}")
        return HDF5Dataset(h5)
    print(f"Loading PNG dataset:  {dataset_dir}")
    return RGBAImageFolder(root=dataset_dir, transform=transform, return_metadata=True)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _instantiate_classifier(config_json: dict, class_names: list[str]) -> Classifier:
    """Build a Classifier from a JSON config dict, ready for inference."""
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
    state_dict = get_state_dict_from_checkpoint(pth_path, device)
    missing, unexpected = clf.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    missing keys  : {len(missing)}  (first: {missing[0]})")
    if unexpected:
        print(f"    unexpected    : {len(unexpected)}  (first: {unexpected[0]})")
    return clf


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_WARMUP_BATCHES = 3   # number of batches used for GPU warm-up (not timed)


@torch.no_grad()
def _run_inference(
    clf: Classifier,
    loader: DataLoader,
    device: str,
    eval_to_model_class: dict[int, int],
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Two-pass inference:
      Pass 1 — warmup (first _WARMUP_BATCHES batches).
      Pass 2 — timed, full dataset; builds y_true / y_pred / probs arrays.

    Returns:
        y_true        : (N,) int64  — ground-truth in *model* class space
        y_pred        : (N,) int64  — argmax predictions
        probs         : (N, K) float32 — full softmax probability vectors
        ms_per_sample : float  — milliseconds per sample (pass 2 only)
    """
    clf.eval()
    clf.to(device)

    # ── Warmup ──────────────────────────────────────────────────────────────
    for i, (x, *_) in enumerate(loader):
        clf(x.to(device))
        if i + 1 >= _WARMUP_BATCHES:
            break
    if device == "cuda":
        torch.cuda.synchronize()

    # ── Timed prediction pass ────────────────────────────────────────────────
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    prob_list:   list[np.ndarray] = []

    t0 = time.perf_counter()
    for x, y, *_ in tqdm(loader, desc=f"  {desc}", unit="batch", leave=False, dynamic_ncols=True):
        x = x.to(device)
        logits = clf(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()  # (B, K_model)

        for i, eval_idx in enumerate(y.numpy()):
            eval_idx = int(eval_idx)
            if eval_idx not in eval_to_model_class:
                continue
            gt = eval_to_model_class[eval_idx]
            y_true_list.append(gt)
            y_pred_list.append(int(p[i].argmax()))
            prob_list.append(p[i])

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if not y_true_list:
        empty = np.zeros((0, 0), dtype=np.float32)
        return np.array([], np.int64), np.array([], np.int64), empty, 0.0

    ms_per_sample = 1000.0 * elapsed / len(y_true_list)
    return (
        np.array(y_true_list, dtype=np.int64),
        np.array(y_pred_list, dtype=np.int64),
        np.array(prob_list,  dtype=np.float32),
        ms_per_sample,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Full classification statistics for one model."""
    if len(y_true) == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "num_samples": 0, "per_class": []}

    labels  = list(range(len(class_names)))
    acc     = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm      = confusion_matrix(y_true, y_pred, labels=labels)
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
            "class_id"   : i,
            "class_name" : name,
            "support"    : int(sup[i]),
            "tp"         : tp,
            "tn"         : tn,
            "fp"         : fp,
            "fn"         : fn,
            "precision"  : float(prec[i]),
            "recall"     : float(rec[i]),
            "f1"         : float(f1[i]),
        })

    return {
        "accuracy"         : acc,
        "balanced_accuracy": bal_acc,
        "num_samples"      : int(len(y_true)),
        "per_class"        : per_class,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_dir = sys.argv[1]
    models_dir  = sys.argv[2] if len(sys.argv) > 2 else "."
    batch_size  = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    device      = _default_device()

    print(f"Dataset  : {dataset_dir}")
    print(f"Models   : {models_dir}")
    print(f"Device   : {device}")
    print(f"Batch    : {batch_size}")

    # ── Locate model pairs ──────────────────────────────────────────────────
    pairs = _discover_model_pairs(models_dir)
    if not pairs:
        print(f"[ERROR] No (.pth, .json) pairs found in: {models_dir}")
        sys.exit(1)

    print(f"\nFound {len(pairs)} model(s):")
    for pth, _ in pairs:
        print(f"  {Path(pth).name}")

    # ── Load dataset ────────────────────────────────────────────────────────
    if not is_valid_dataset_dir(dataset_dir):
        print(f"[ERROR] Not a valid dataset directory: {dataset_dir}")
        sys.exit(1)

    dataset            = _load_dataset(dataset_dir)
    dataset_classes    = dataset.classes
    num_dataset_classes = len(dataset_classes)
    print(f"\nDataset classes ({num_dataset_classes}): {dataset_classes}")
    print(f"Dataset samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = min(4, os.cpu_count() or 1),
        drop_last   = False,
        collate_fn  = metadata_collate_fn,
    )

    # ── Per-model inference ─────────────────────────────────────────────────
    # probs_all[m] : (N, num_dataset_classes) — probabilities aligned to
    #                the dataset class space (zeros for unknown classes).
    all_model_meta : list[dict] = []
    probs_all      : list[np.ndarray] = []
    y_true_global  : np.ndarray | None = None

    for model_idx, (pth_path, cfg_path) in enumerate(pairs):
        model_name = Path(pth_path).stem
        print(f"\n{'='*70}")
        print(f"[{model_idx + 1}/{len(pairs)}]  {model_name}")
        print(f"{'='*70}")

        config_json      = load_hyperparameters(cfg_path)
        model_classes    = config_json.get("classes", dataset_classes)

        # Map dataset label → model label so we can align predictions.
        eval_to_model, shared, miss_eval, miss_model = build_eval_to_model_mapping(
            dataset_classes, model_classes
        )
        if miss_eval:
            print(f"  [WARN] dataset classes absent from model output: {miss_eval}")
        if miss_model:
            print(f"  [WARN] model classes absent from dataset:        {miss_model}")
        if not eval_to_model:
            print("  [ERROR] zero class overlap — skipping.")
            continue

        # Instantiate and load weights
        try:
            clf = _instantiate_classifier(config_json, model_classes)
            clf = _load_weights(clf, pth_path, device)
        except Exception as e:
            print(f"  [ERROR] could not load model: {e}")
            continue

        num_params = _count_parameters(clf)

        # Run timed inference (model outputs are in *model_classes* space)
        y_true_model, y_pred_model, probs_model, ms_per_sample = _run_inference(
            clf, loader, device, eval_to_model, desc=model_name
        )

        if len(y_true_model) == 0:
            print("  [WARN] no aligned samples produced — skipping.")
            del clf
            continue

        # ── Re-align probability vectors to the dataset class space ─────────
        # probs_model columns correspond to model_classes indices;
        # we scatter them into the dataset_classes layout.
        N = len(y_true_model)
        aligned_probs = np.zeros((N, num_dataset_classes), dtype=np.float32)
        for model_cls_idx, model_cls_name in enumerate(model_classes):
            if model_cls_name in dataset_classes:
                ds_idx = dataset_classes.index(model_cls_name)
                aligned_probs[:, ds_idx] = probs_model[:, model_cls_idx]

        y_pred_aligned = aligned_probs.argmax(axis=1).astype(np.int64)

        # Re-map y_true from model space → dataset space
        model_to_ds = {
            model_classes.index(name): dataset_classes.index(name)
            for name in shared
        }
        y_true_ds = np.array(
            [model_to_ds.get(int(t), int(t)) for t in y_true_model],
            dtype=np.int64,
        )

        # Verify consistency across models (same sample order expected)
        if y_true_global is None:
            y_true_global = y_true_ds
        elif not np.array_equal(y_true_global, y_true_ds):
            print("  [WARN] y_true mismatch vs previous models — sample ordering may differ.")

        stats = _compute_stats(y_true_ds, y_pred_aligned, dataset_classes)

        print(f"  Accuracy         : {stats['accuracy']:.4f}")
        print(f"  Balanced Acc     : {stats['balanced_accuracy']:.4f}")
        print(f"  ms / sample      : {ms_per_sample:.4f}")
        print(f"  Parameters       : {num_params:,}")

        all_model_meta.append({
            "index"              : model_idx,
            "name"               : model_name,
            "pth"                : pth_path,
            "json"               : cfg_path,
            "model_type"         : config_json.get("model", "unknown"),
            "tile_size"          : config_json["hparams"].get("tile_size", 0),
            "num_params"         : num_params,
            "ms_per_sample"      : ms_per_sample,
            "accuracy"           : stats["accuracy"],
            "balanced_accuracy"  : stats["balanced_accuracy"],
            "num_aligned_samples": int(N),
            "per_class"          : stats["per_class"],
        })
        probs_all.append(aligned_probs)

        # Free GPU memory between models
        del clf
        if device == "cuda":
            torch.cuda.empty_cache()

    if not probs_all:
        print("\n[ERROR] No models produced valid predictions.  Aborting.")
        sys.exit(1)

    # ── Save outputs ────────────────────────────────────────────────────────
    dataset_slug = Path(dataset_dir.rstrip("/\\")).name
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem         = f"ensemble_{dataset_slug}_{timestamp}"
    json_path    = f"{stem}.json"
    npz_path     = f"{stem}.npz"

    # Stack probability arrays: (M, N, K)
    probs_matrix = np.stack(probs_all, axis=0).astype(np.float32)

    np.savez_compressed(
        npz_path,
        y_true = y_true_global,
        probs  = probs_matrix,
    )

    output = {
        "generated"   : datetime.datetime.now().isoformat(),
        "dataset"     : dataset_dir,
        "num_samples" : int(len(y_true_global)),
        "num_classes" : num_dataset_classes,
        "classes"     : dataset_classes,
        "npz_file"    : npz_path,        # companion array file
        "models"      : all_model_meta,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    mb = probs_matrix.nbytes / 1e6
    print(f"\n{'='*70}")
    print(f"Saved metadata : {json_path}")
    print(f"Saved arrays   : {npz_path}  ({mb:.1f} MB, shape {probs_matrix.shape})")
    print(f"Models stored  : {len(probs_all)}")
    print(f"{'='*70}")
    print(f"\nNext step:")
    print(f"  python3 evaluateOptimalEnsemble.py {json_path}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
