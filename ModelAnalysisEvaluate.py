#!/usr/bin/python3
"""
ModelAnalysisEvaluate.py

Run GPU inference for every (.pth, .json) model pair in a directory and save
raw per-sample probabilities + model metadata to disk.  The saved data can
then be loaded by ModelAnalysisReport.py to generate (and re-generate) plots
and HTML reports without repeating the expensive inference step.

Output layout:
    <out_dir>/raw_data/
        manifest.json               — dataset info, class list, evaluated model list
        <model_name>_meta.json      — per-model metadata and benchmark sweep data
        <model_name>_probs.npz      — y_true (N,) and probs_aligned (N, K) float32

Usage:
    python3 ModelAnalysisEvaluate.py <dataset_dir> [models_dir]
            [--batch N] [--fp16] [--out <dir>] [--no-bench]
            [--bench-batches B1,B2,...] [--bench-steps S1,S2,...]
            [--image-width W] [--image-height H]

    dataset_dir : ImageFolder-style directory or directory containing dataset.h5
    models_dir  : directory containing *.pth + matching *.json  (default: ".")
    --batch N   : inference batch size  (default: 64)
    --fp16      : run inference under autocast FP16
    --out dir   : parent output directory  (default: "model_analysis_<timestamp>")
    --no-bench  : skip the batch×step throughput benchmark sweep
"""

import datetime
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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

_WARMUP_BATCHES = 3
_N_TIME_BATCHES = 20


@dataclass
class EvalConfig:
    dataset_dir: str
    models_dir: str = "."
    batch: int = 64
    fp16: bool = False
    out: str | None = None
    no_bench: bool = False
    bench_batch_sizes: list[int] | None = None
    bench_step_sizes: list[int] | None = None
    image_width: int = 1224
    image_height: int = 1024


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(dataset_dir: str):
    """Load dataset from HDF5 or ImageFolder format."""
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
    """Find all (.pth, .json) model pairs in models_dir."""
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
    """Reconstruct a Classifier from its saved config JSON and class list."""
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
    """Load checkpoint weights and move model to device in eval mode."""
    sd = get_state_dict_from_checkpoint(pth_path, device)
    clf.load_state_dict(sd, strict=False)
    return clf.to(device).eval()


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_model(clf: Classifier, loader: DataLoader, device: str,
               eval_to_model: dict[int, int], catchall_model_indices: list[int],
               catchall_ds_indices: list[int], use_fp16: bool,
               desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Two-pass inference: Pass 1 warms up the GPU and collects predictions;
    Pass 2 pre-loads batches into CPU RAM then times only PCIe+GPU work.

    Returns:
        y_true_ds    : (N,) int64 — ground truth in dataset class space.
        y_pred       : (N,) int64 — argmax prediction in model class space.
        probs        : (N, K_model) float32 — softmax probability vectors.
        ms_per_sample: float — PCIe+GPU inference time per sample.
    """
    clf.eval()
    clf.to(device)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16 and device == "cuda"
        else torch.amp.autocast(device_type="cuda", enabled=False)
        if device == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    # Pass 1a: warmup
    for i, (x, *_) in enumerate(loader):
        with autocast_ctx:
            clf(x.to(device))
        if i + 1 >= _WARMUP_BATCHES:
            break
    if device == "cuda":
        torch.cuda.synchronize()

    # Pass 1b: full prediction
    y_true_ds_list: list[int] = []
    y_pred_list:    list[int] = []
    prob_list:      list[np.ndarray] = []

    for x, y, *_ in tqdm(loader, desc=f"    {desc}", unit="batch", leave=False, dynamic_ncols=True):
        x = x.to(device)
        with autocast_ctx:
            logits = clf(x)
        p = torch.softmax(logits.float(), dim=1).cpu().numpy()
        for i, ev in enumerate(y.numpy()):
            ev = int(ev)
            if ev not in eval_to_model:
                continue
            y_true_ds_list.append(ev)
            y_pred_list.append(int(p[i].argmax()))
            prob_list.append(p[i])

    if not y_true_ds_list:
        return np.array([], np.int64), np.array([], np.int64), np.zeros((0, 0), np.float32), 0.0

    # Pass 2: clean GPU throughput measurement (I/O pre-loaded into CPU RAM)
    time_cpu: list[torch.Tensor] = []
    for x, *_ in loader:
        time_cpu.append(x)
        if len(time_cpu) >= _N_TIME_BATCHES:
            break

    ms_per_sample = 0.0
    if time_cpu:
        if device == "cuda":
            torch.cuda.synchronize()
        t_gpu = time.perf_counter()
        for x in time_cpu:
            with autocast_ctx:
                clf(x.to(device))
        if device == "cuda":
            torch.cuda.synchronize()
        timed_n = sum(b.shape[0] for b in time_cpu)
        ms_per_sample = (time.perf_counter() - t_gpu) * 1000.0 / max(timed_n, 1)

    return (
        np.array(y_true_ds_list, dtype=np.int64),
        np.array(y_pred_list,    dtype=np.int64),
        np.array(prob_list,      dtype=np.float32),
        ms_per_sample,
    )


# ---------------------------------------------------------------------------
# Inline balanced accuracy (avoids sklearn import just for this)
# ---------------------------------------------------------------------------

def _quick_bal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.sum() > 0:
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig) -> str:
    """
    Run the full evaluation pipeline and return the output directory path.

    Args:
        cfg: EvalConfig with dataset_dir, models_dir, and evaluation options.

    Returns:
        Absolute path to the output directory (contains raw_data/ subdirectory).
    """
    bench_batch_sizes = cfg.bench_batch_sizes or BENCHMARK_BATCH_SIZES
    bench_step_sizes  = cfg.bench_step_sizes or BENCHMARK_STEP_SIZES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.out) if cfg.out else Path(f"model_analysis_{timestamp}")
    raw_dir   = out_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model Analysis — Evaluate")
    print(f"{'='*60}")
    print(f"  Dataset    : {cfg.dataset_dir}")
    print(f"  Models dir : {cfg.models_dir}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {cfg.batch}")
    print(f"  FP16       : {cfg.fp16}")
    print(f"  Output     : {out_dir}")

    if not is_valid_dataset_dir(cfg.dataset_dir):
        print(f"\n[ERROR] Not a valid dataset directory: {cfg.dataset_dir}")
        sys.exit(1)

    pairs = _discover_pairs(cfg.models_dir)
    if not pairs:
        print(f"[ERROR] No (.pth, .json) pairs found in: {cfg.models_dir}")
        sys.exit(1)
    print(f"\n  Found {len(pairs)} model(s):")
    for pth, _ in pairs:
        print(f"    {Path(pth).name}")

    print("\n  Loading dataset...")
    dataset             = _load_dataset(cfg.dataset_dir)
    dataset_classes     = dataset.classes
    num_dataset_classes = len(dataset_classes)
    print(f"  Classes ({num_dataset_classes}): {dataset_classes}")
    print(f"  Samples: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size  = cfg.batch,
        shuffle     = False,
        num_workers = min(8, os.cpu_count() or 1),
        drop_last   = False,
        collate_fn  = metadata_collate_fn,
    )

    evaluated_models: list[str] = []

    for midx, (pth_path, cfg_path) in enumerate(pairs):
        model_name  = Path(pth_path).stem
        print(f"\n  [{midx+1}/{len(pairs)}] {model_name}")

        config_json   = load_hyperparameters(cfg_path)
        model_classes = config_json.get("classes", dataset_classes)

        eval_to_model, _, miss_eval, miss_model = build_eval_to_model_mapping(
            dataset_classes, model_classes
        )
        if miss_eval:
            print(f"    [WARN] dataset classes absent from model: {miss_eval}")
        if miss_model:
            print(f"    [WARN] model classes absent from dataset: {miss_model}")
        if not eval_to_model:
            print("    [ERROR] zero class overlap — skipping.")
            continue

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

        y_true_ds, _, probs_model, ms = _run_model(
            clf, loader, device, eval_to_model,
            catchall_model_indices, catchall_ds_indices,
            cfg.fp16, model_name,
        )

        if len(y_true_ds) == 0:
            print("    [WARN] No aligned samples — skipping.")
            del clf
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        # Align probabilities to dataset class space
        N = len(y_true_ds)
        aligned = np.zeros((N, num_dataset_classes), dtype=np.float32)
        for mi, mn in enumerate(model_classes):
            if mn in dataset_classes:
                aligned[:, dataset_classes.index(mn)] = probs_model[:, mi]
            elif mi in catchall_model_indices and catchall_ds_indices:
                share = probs_model[:, mi] / len(catchall_ds_indices)
                for di in catchall_ds_indices:
                    aligned[:, di] += share

        y_pred_aligned = aligned.argmax(axis=1).astype(np.int64)
        acc     = float((y_pred_aligned == y_true_ds).mean())
        bal_acc = _quick_bal_acc(y_true_ds, y_pred_aligned)

        print(f"    Accuracy  : {acc:.4f}")
        print(f"    BalAcc    : {bal_acc:.4f}")
        print(f"    ms/sample : {ms:.4f}")
        print(f"    Params    : {num_params:,}")
        print(f"    Samples   : {N:,}")

        # Save per-sample probabilities
        np.savez_compressed(
            raw_dir / f"{model_name}_probs.npz",
            y_true=y_true_ds,
            probs=aligned,
        )

        # Benchmark: sweep batch_size × step_size combinations
        bench_data: list[dict] = []
        if not cfg.no_bench:
            bench_data = run_throughput_benchmark(
                clf, dataset, eval_to_model,
                batch_sizes              = bench_batch_sizes,
                step_sizes               = bench_step_sizes,
                tile_size                = config_json["hparams"].get("tile_size", 48),
                image_width              = cfg.image_width,
                image_height             = cfg.image_height,
                device                   = device,
                precomputed_accuracy     = acc,
                precomputed_bal_accuracy = bal_acc,
                use_fp16                 = cfg.fp16,
            )
            ok_pts  = [r for r in bench_data if not r.get("oom")]
            oom_pts = [r for r in bench_data if r.get("oom")]
            if oom_pts:
                print(f"    OOM batch sizes: {sorted({r['batch_size'] for r in oom_pts})}")
            if ok_pts:
                best_pt = max(ok_pts, key=lambda r: r["effective_fps"])
                print(f"    Peak eff. FPS : {best_pt['effective_fps']:.1f} "
                      f"(batch={best_pt['batch_size']}, step={best_pt['step_size']})")

        meta = {
            "name"               : model_name,
            "model_type"         : config_json.get("model", "unknown"),
            "tile_size"          : config_json["hparams"].get("tile_size", 0),
            "num_params"         : num_params,
            "ms_per_sample"      : ms,
            "accuracy"           : acc,
            "balanced_accuracy"  : bal_acc,
            "num_aligned_samples": N,
            "benchmark"          : bench_data,
        }
        with open(raw_dir / f"{model_name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        evaluated_models.append(model_name)

        del clf
        if device == "cuda":
            torch.cuda.empty_cache()

    if not evaluated_models:
        print("\n[ERROR] No models produced valid predictions.")
        sys.exit(1)

    manifest = {
        "generated"      : datetime.datetime.now().isoformat(),
        "dataset"        : cfg.dataset_dir,
        "dataset_classes": dataset_classes,
        "num_samples"    : len(dataset),
        "fp16"           : cfg.fp16,
        "batch_size"     : cfg.batch,
        "models"         : evaluated_models,
    }
    with open(raw_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Raw data saved to: {raw_dir}")

    return str(out_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("models_dir", nargs="?", default=".")
    ap.add_argument("--batch",         type=int, default=64)
    ap.add_argument("--fp16",          action="store_true")
    ap.add_argument("--out",           default=None)
    ap.add_argument("--no-bench",      action="store_true",
                    help="Skip the batch×step throughput benchmark sweep")
    ap.add_argument("--bench-batches", default=None,
                    help="Comma-separated batch sizes (default: from evaluateClassifierNew)")
    ap.add_argument("--bench-steps",   default=None,
                    help="Comma-separated tile step sizes")
    ap.add_argument("--image-width",   type=int, default=1224)
    ap.add_argument("--image-height",  type=int, default=1024)
    args = ap.parse_args()

    bench_batch_sizes = (
        [int(x) for x in args.bench_batches.split(",")]
        if args.bench_batches else None
    )
    bench_step_sizes = (
        [int(x) for x in args.bench_steps.split(",")]
        if args.bench_steps else None
    )

    cfg = EvalConfig(
        dataset_dir       = args.dataset_dir,
        models_dir        = args.models_dir,
        batch             = args.batch,
        fp16              = args.fp16,
        out               = args.out,
        no_bench          = args.no_bench,
        bench_batch_sizes = bench_batch_sizes,
        bench_step_sizes  = bench_step_sizes,
        image_width       = args.image_width,
        image_height      = args.image_height,
    )
    run_evaluation(cfg)
    print(f"\n{'='*60}")
    print(f"  Run:  python3 ModelAnalysisReport.py {cfg.out or 'model_analysis_<timestamp>'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
