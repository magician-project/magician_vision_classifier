#!/usr/bin/python3
"""
Benchmark inference throughput for all allclass_*.pth models on raw PNM frames.

Usage:
    python3 benchmarkAllModels.py --frames 100 --step 15 20 --from /path/to/frames
    python3 benchmarkAllModels.py --frames 100 --step 15 20 --from /path/to/frames --models .

Outputs:
    benchmark_results.csv  — per-model, per-step FPS results
    benchmark_results.png  — grouped bar chart
"""

import argparse
import csv
import os
import sys
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from readData import readPolarPNMToRGBA
from liveClassifierTorch import tile_and_cast_data_torch
from trainMagicianVisionClassifierTorch import Classifier, load_hyperparameters


def _get_clean_class_id(class_names):
    for i, name in enumerate(class_names):
        if name in ("class_clean", "Clean"):
            return i
    return 0


def _load_state_dict(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _build_classifier(cfg: dict, class_names: list) -> Classifier:
    # Single source of truth: Classifier.from_config reads every knob from
    # config["hparams"], the nesting the trainer writes. This block used to read
    # AoLP/DoLP/Unpolarized/Max/Min/RangePolarization from the config TOP LEVEL,
    # where no config has ever carried them (167 have them under hparams, 0 at the
    # top), so it silently built a 4-channel model for every 5-channel run. It also
    # never passed monochrome, custom_channels, custom_res_blocks, pretrained or
    # timm_stem_stride -- see test_classifier_from_config.py.
    return Classifier.from_config(cfg, num_classes=len(class_names),
                                  clean_class=_get_clean_class_id(class_names))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TRY_BATCH_SIZES = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 1]
_WARMUP_FRAMES = 3


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(models_dir: str):
    """Return list of (pth_path, cfg_path) for all valid allclass_*.pth models."""
    pairs = []
    for name in sorted(os.listdir(models_dir)):
        if not (name.startswith("allclass_") and name.endswith(".pth")):
            continue
        base = name[:-4]
        pth = os.path.join(models_dir, f"{base}.pth")
        cfg = os.path.join(models_dir, f"{base}.json")
        if not os.path.isfile(cfg):
            print(f"  [skip] {name}: no matching .json")
            continue
        if not zipfile.is_zipfile(pth):
            print(f"  [skip] {name}: not a valid checkpoint")
            continue
        pairs.append((pth, cfg))
    return pairs


def discover_frames(frames_dir: str, max_frames: int):
    """Return sorted list of PNM/PNG frame paths, capped at max_frames."""
    frames = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith(".pnm") or f.lower().endswith(".png")
    )
    return frames[:max_frames]


# ---------------------------------------------------------------------------
# Frame pre-loading
# ---------------------------------------------------------------------------

def load_frames(frame_paths):
    """Load and debayer PNM frames to RGBA numpy arrays (H, W, 4) uint8."""
    loaded = []
    print(f"Pre-loading {len(frame_paths)} frames ...")
    for p in tqdm(frame_paths, unit="frame"):
        rgba = readPolarPNMToRGBA(p)
        if rgba is None:
            print(f"  [warn] failed to load {p}")
            continue
        loaded.append(rgba)
    print(f"  {len(loaded)} frames ready  shape={loaded[0].shape if loaded else 'n/a'}")
    return loaded


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(pth_path: str, cfg_path: str, device: str):
    """Return (classifier, tile_size, model_name) ready for eval."""
    cfg = load_hyperparameters(cfg_path)
    class_names = list(cfg["classes"])
    tile_size = int(cfg["hparams"].get("tile_size", 48))

    clf = _build_classifier(cfg, class_names)
    clf.load_state_dict(_load_state_dict(pth_path, device), strict=False)
    clf.to(device)
    clf.eval()

    model_name = os.path.splitext(os.path.basename(pth_path))[0]
    return clf, tile_size, model_name


# ---------------------------------------------------------------------------
# Batch-size probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def find_max_batch_size(clf, sample_nchw: torch.Tensor, device: str) -> int:
    """Find the largest batch size from _TRY_BATCH_SIZES that runs without OOM."""
    n_tiles = sample_nchw.shape[0]
    autocast = torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda"))
    for bs in _TRY_BATCH_SIZES:
        if bs > n_tiles:
            continue
        try:
            chunk = sample_nchw[:bs]
            with autocast:
                _ = clf(chunk)
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return bs
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise
    return 1


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def _tiles_nchw(rgba: np.ndarray, tile_size: int, step: int, device: str) -> torch.Tensor:
    """Tile one RGBA frame → (N, C, tile_size, tile_size) uint8 on device.

    Kept uint8 to match the live pipeline: the model dequantises internally, so
    benchmarking float32 tiles would measure a transfer size inference never uses.
    """
    tiles = tile_and_cast_data_torch(rgba, tile_size=tile_size, step=step)
    return tiles.permute(0, 3, 1, 2).contiguous().to(device)


@torch.no_grad()
def benchmark_model_step(
    clf,
    frames_rgba: list,
    tile_size: int,
    step: int,
    device: str,
) -> dict:
    """
    Measure end-to-end fps (tile + forward) over all frames for one step size.

    Returns a dict with fps, batch_size, num_tiles, time_per_frame_ms.
    """
    if not frames_rgba:
        return {"fps": 0.0, "batch_size": 0, "num_tiles": 0, "time_per_frame_ms": 0.0}

    # Determine tile count and find the largest batch that fits in memory
    sample_tiles = _tiles_nchw(frames_rgba[0], tile_size, step, device)
    num_tiles = sample_tiles.shape[0]
    if num_tiles == 0:
        return {"fps": 0.0, "batch_size": 0, "num_tiles": 0, "time_per_frame_ms": 0.0}

    max_bs = find_max_batch_size(clf, sample_tiles, device)

    autocast = torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda"))

    def _infer_frame(rgba):
        tiles = _tiles_nchw(rgba, tile_size, step, device)
        for start in range(0, tiles.shape[0], max_bs):
            with autocast:
                _ = clf(tiles[start:start + max_bs])

    # Warmup
    warmup_frames = frames_rgba[:_WARMUP_FRAMES]
    for rgba in warmup_frames:
        _infer_frame(rgba)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed run
    t0 = time.perf_counter()
    for rgba in frames_rgba:
        _infer_frame(rgba)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = len(frames_rgba)
    fps = n / elapsed if elapsed > 0 else 0.0
    tpf_ms = elapsed / n * 1000.0 if n > 0 else 0.0

    return {
        "fps": fps,
        "batch_size": max_bs,
        "num_tiles": num_tiles,
        "time_per_frame_ms": tpf_ms,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_csv(results: list, out_path: str):
    fields = ["model", "step", "tile_size", "batch_size", "fps",
              "num_tiles_per_frame", "time_per_frame_ms"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"Saved CSV  : {out_path}")


def plot_results(results: list, out_path: str):
    if not results:
        return

    steps = sorted(set(r["step"] for r in results))
    models = list(dict.fromkeys(r["model"] for r in results))

    fps_table = {m: {} for m in models}
    for r in results:
        fps_table[r["model"]][r["step"]] = r["fps"]

    n_models = len(models)
    n_steps = len(steps)
    x = np.arange(n_models)
    width = 0.8 / n_steps

    fig, ax = plt.subplots(figsize=(max(14, n_models * 0.9), 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_steps))

    for i, step in enumerate(steps):
        fps_vals = [fps_table[m].get(step, 0.0) for m in models]
        offset = (i - n_steps / 2 + 0.5) * width
        bars = ax.bar(x + offset, fps_vals, width * 0.92, label=f"step={step}", color=colors[i])
        for bar, val in zip(bars, fps_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    short_names = [m.replace("allclass_", "") for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Frames per second (Hz)")
    ax.set_title("Inference throughput — all models")
    ax.legend(title="Tile step")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--frames",  type=int,     default=100,       help="Number of frames to benchmark (default: 100)")
    ap.add_argument("--step",    type=int,     nargs="+",         help="Tile step size(s) (e.g. --step 15 20)")
    ap.add_argument("--from",    dest="frames_dir", required=True, help="Directory containing colorFrame_*.pnm files")
    ap.add_argument("--models",  default=_script_dir,             help="Directory with allclass_*.pth models (default: script dir)")
    ap.add_argument("--out",     default="benchmark_results",     help="Output file stem (default: benchmark_results)")
    args = ap.parse_args()

    step_sizes = args.step if args.step else [15, 20]

    print(f"Device     : {DEVICE}")
    print(f"Frames     : {args.frames}")
    print(f"Step sizes : {step_sizes}")
    print(f"Frames dir : {args.frames_dir}")
    print(f"Models dir : {args.models}")

    frame_paths = discover_frames(args.frames_dir, args.frames)
    if not frame_paths:
        print(f"No PNM/PNG frames found in: {args.frames_dir}")
        sys.exit(1)
    print(f"Found {len(frame_paths)} frames")

    frames_rgba = load_frames(frame_paths)
    if not frames_rgba:
        print("No frames loaded.")
        sys.exit(1)

    model_pairs = discover_models(args.models)
    if not model_pairs:
        print(f"No allclass_*.pth models found in: {args.models}")
        sys.exit(1)
    print(f"Found {len(model_pairs)} models\n")

    results = []
    total_combos = len(model_pairs) * len(step_sizes)
    pbar = tqdm(total=total_combos, unit="combo", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")

    for pth, cfg in model_pairs:
        model_name = os.path.splitext(os.path.basename(pth))[0]
        short_name = model_name.replace("allclass_", "")
        pbar.set_postfix_str(f"loading {short_name}")
        try:
            clf, tile_size, _ = load_model(pth, cfg, DEVICE)
        except Exception as exc:
            tqdm.write(f"[skip] {model_name}: load failed: {exc}")
            for step in step_sizes:
                results.append({
                    "model": model_name, "step": step, "tile_size": 0,
                    "batch_size": 0, "fps": 0.0,
                    "num_tiles_per_frame": 0, "time_per_frame_ms": 0.0,
                })
                pbar.update(1)
            continue

        for step in step_sizes:
            pbar.set_description(f"{short_name} / step={step}")
            pbar.set_postfix_str("running...")
            try:
                r = benchmark_model_step(clf, frames_rgba, tile_size, step, DEVICE)
                tqdm.write(
                    f"  {short_name:<30}  step={step:3d}  tiles={r['num_tiles']:5d}  "
                    f"batch={r['batch_size']:5d}  fps={r['fps']:7.2f}  "
                    f"t/frame={r['time_per_frame_ms']:.1f}ms"
                )
                pbar.set_postfix_str(f"fps={r['fps']:.2f}")
                results.append({
                    "model":               model_name,
                    "step":                step,
                    "tile_size":           tile_size,
                    "batch_size":          r["batch_size"],
                    "fps":                 round(r["fps"], 4),
                    "num_tiles_per_frame": r["num_tiles"],
                    "time_per_frame_ms":   round(r["time_per_frame_ms"], 2),
                })
            except Exception as exc:
                tqdm.write(f"  {short_name}  step={step}  [error] {exc}")
                pbar.set_postfix_str("error")
                results.append({
                    "model": model_name, "step": step, "tile_size": tile_size,
                    "batch_size": 0, "fps": 0.0,
                    "num_tiles_per_frame": 0, "time_per_frame_ms": 0.0,
                })
            pbar.update(1)

        del clf
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    if not results:
        print("No results collected.")
        sys.exit(1)

    save_csv(results, f"{args.out}.csv")
    plot_results(results, f"{args.out}.png")


if __name__ == "__main__":
    main()
