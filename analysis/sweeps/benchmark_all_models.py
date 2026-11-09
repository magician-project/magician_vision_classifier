#!/usr/bin/python3
"""
Benchmark inference throughput for all allclass_*.pth models on raw PNM frames.

Times the same code the ROS node runs per frame: ClassifierPnm.forward() on a raw frame
(debayer, whole-grid fp16 forward, gate, majority vote, voting, heatmap), with the node's
compile mode, global torch settings and recommended_configuration.json preset. Not
timed: shared-memory reads, ROS publishing and the visualization window. Reparameterizable
backbones run unfused, because ClassifierPnm does not fuse them.

Requires timm; without it every timm-based model fails to load and is reported as 0 fps:
    pip install timm==1.0.28

Usage:
    python3 benchmarkAllModels.py --frames 100 --step 15 20 --from /path/to/frames
    python3 benchmarkAllModels.py --frames 100 --step 15 20 --from /path/to/frames --models .

    # Every archive listed in a text file (one URL per line), downloaded as needed:
    python3 -m analysis.sweeps.benchmark_all_models --from /path/to/frames --sources sources.txt

Outputs:
    benchmark_results.csv  — per-model, per-step FPS results
    benchmark_results.png  — grouped bar chart
"""

import argparse
import contextlib
import csv
import io
import os
import re
import sys
import time
import zipfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mvc.inference.classifier_pnm import ClassifierPnm, load_recommended_configuration
from mvc.inference.model_download import BASE_URL, download_model
from mvc.paths import repo_root


# The node's ClassifierPnm settings; see live_torch_ros.main() and DefectPublisher.__init__.
ROS_COMPILE_MODE = "max-autotune-no-cudagraphs"
ROS_MAJORITY_WINDOW = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


def fetch_sources(sources_path: str, dest: str):
    """Download whatever is missing from `dest`; return (pth_path, cfg_path) in file order.

    Each non-blank, non-# line is an archive URL, or a bare archive name on the model
    server. Exits if any model is unavailable: a partial run must not be mistaken for
    the full table.
    """
    with open(sources_path) as f:
        sources = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]

    pairs = []
    for i, source in enumerate(sources, 1):
        base, _, zip_name = source.rpartition("/")
        run = re.sub(r"_\d{8}_\d{6}\.zip$", "", zip_name)
        pth, cfg = os.path.join(dest, f"{run}.pth"), os.path.join(dest, f"{run}.json")
        if not (os.path.isfile(cfg) and zipfile.is_zipfile(pth)):
            print(f"[{i}/{len(sources)}] downloading {zip_name}")
            download_model(zip_name, dest, base_url=f"{base}/" if base else BASE_URL)
        if not (os.path.isfile(cfg) and zipfile.is_zipfile(pth)):
            sys.exit(f"{zip_name} did not yield {run}.pth + {run}.json")
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
# Progress
# ---------------------------------------------------------------------------

def _hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Progress:
    """Single-line \\r progress bar with elapsed time and ETA over all model/step combos."""

    WIDTH = 30
    MIN_INTERVAL_S = 0.5    # throttled: this also runs inside the timed loop

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.label = ""
        self.start = time.perf_counter()
        self.last_draw = 0.0

    def draw(self, partial: float = 0.0, force: bool = True):
        """Redraw; `partial` is the fraction completed of the current combo."""
        now = time.perf_counter()
        if not force and now - self.last_draw < self.MIN_INTERVAL_S:
            return
        self.last_draw = now
        frac = (self.done + partial) / self.total
        elapsed = now - self.start
        eta = _hms(elapsed * (1 - frac) / frac) if frac > 0 else "--:--:--"
        filled = int(frac * self.WIDTH)
        bar = "#" * filled + "-" * (self.WIDTH - filled)
        sys.stdout.write(f"\r[{bar}] {self.done}/{self.total} {frac * 100:5.1f}% "
                         f"elapsed {_hms(elapsed)} eta {eta}  {self.label}\033[K")
        sys.stdout.flush()

    def log(self, message: str):
        """Print a permanent line above the bar."""
        sys.stdout.write(f"\r\033[K{message}\n")
        self.draw()

    def advance(self):
        self.done += 1
        self.draw()


# ---------------------------------------------------------------------------
# Frame pre-loading
# ---------------------------------------------------------------------------

def load_frames(frame_paths):
    """Load raw frames as the node receives them; debayering is part of the timed path."""
    loaded = []
    print(f"Pre-loading {len(frame_paths)} frames ...")
    for p in tqdm(frame_paths, unit="frame"):
        frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"  [warn] failed to load {p}")
            continue
        loaded.append(frame)
    print(f"  {len(loaded)} frames ready  shape={loaded[0].shape if loaded else 'n/a'}")
    return loaded


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_runtime_classifier(pth_path: str, cfg_path: str, step: int, preset: dict):
    """Build a ClassifierPnm configured like the node's; return (classifier, warnings).

    One instance per model and step: the node compiles for a single constant batch shape,
    and sharing a compiled model between steps would specialise it for two.
    """
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            clf = ClassifierPnm(model_path=pth_path, cfg_path=cfg_path, step=step,
                                compile_mode=ROS_COMPILE_MODE)
    except (Exception, SystemExit) as exc:      # ClassifierPnm calls sys.exit on bad files
        raise RuntimeError(f"{exc!r}; last output: {captured.getvalue()[-300:]}") from exc

    gate = preset["gate"]
    clf.gateMode = gate.get("mode", clf.gateMode)
    clf.assignBestDefectClass = bool(gate.get("assign_best_defect_class", clf.assignBestDefectClass))
    if gate.get("threshold") is not None:
        clf.maxProbabilityThreshold = float(gate["threshold"])

    # load_state_dict is non-strict, so a checkpoint that does not fit the built network
    # loads anyway and would be timed as the wrong model.
    warnings = [line[:300] for line in captured.getvalue().splitlines()
                if line.startswith(("Missing keys:", "Unexpected keys:")) and not line.endswith("[]")]
    return clf, warnings


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark_model_step(clf, frames: list, preset: dict, device: str, on_frame=None) -> dict:
    """Measure ClassifierPnm.forward() per frame, called exactly as the node calls it.

    Returns a dict with fps, batch_size, num_tiles, time_per_frame_ms.
    `on_frame(fraction_done)` is called after each timed frame, if given.
    """
    if not frames:
        return {"fps": 0.0, "batch_size": 0, "num_tiles": 0, "time_per_frame_ms": 0.0}

    runtime = preset["runtime"]
    stats = {}          # a stats sink also stops runSingle printing a line per frame

    def _infer_frame(frame):
        with torch.inference_mode():
            clf.forward(frame,
                        majorityVote=bool(runtime["majority_voting"]),
                        erosion_kernel=int(runtime["erosion_kernel"]),
                        erosion_threshold=int(runtime["min_votes"]),
                        log=False,
                        stats=stats,
                        majority_window=ROS_MAJORITY_WINDOW)

    # Warmup: includes the autotuned compile, which the node pays once at startup.
    # Autotune writes pages of kernel timings to stderr, which would break the progress bar.
    with contextlib.redirect_stderr(io.StringIO()):
        for frame in frames[:_WARMUP_FRAMES]:
            _infer_frame(frame)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed run
    t0 = time.perf_counter()
    for i, frame in enumerate(frames, 1):
        _infer_frame(frame)
        if on_frame:
            on_frame(i / len(frames))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = len(frames)
    return {
        "fps": n / elapsed if elapsed > 0 else 0.0,
        "batch_size": stats["tiles"],   # the node forwards the whole grid as one batch
        "num_tiles": stats["tiles"],
        "time_per_frame_ms": elapsed / n * 1000.0,
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
    ap.add_argument("--models",  default=None,                    help="Directory with models (default: repo root, or zoo_models with --sources)")
    ap.add_argument("--sources", default=None,                    help="Text file with one model archive URL per line; missing ones are downloaded into --models")
    ap.add_argument("--out",     default="benchmark_results",     help="Output file stem (default: benchmark_results)")
    args = ap.parse_args()

    if args.models is None:
        args.models = "zoo_models" if args.sources else repo_root()
    step_sizes = args.step if args.step else ([16] if args.sources else [15, 20])

    # Same global settings as live_torch_ros.main(), before any model is loaded.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    preset = load_recommended_configuration()

    print(f"Device     : {DEVICE}")
    print(f"Preset     : {preset.get('name')}")
    print(f"Frames     : {args.frames}")
    print(f"Step sizes : {step_sizes}")
    print(f"Frames dir : {args.frames_dir}")
    print(f"Models dir : {args.models}")

    frame_paths = discover_frames(args.frames_dir, args.frames)
    if not frame_paths:
        print(f"No PNM/PNG frames found in: {args.frames_dir}")
        sys.exit(1)
    print(f"Found {len(frame_paths)} frames")

    frames = load_frames(frame_paths)
    if not frames:
        print("No frames loaded.")
        sys.exit(1)

    model_pairs = fetch_sources(args.sources, args.models) if args.sources else discover_models(args.models)
    if not model_pairs:
        print(f"No allclass_*.pth models found in: {args.models}")
        sys.exit(1)
    print(f"Found {len(model_pairs)} models\n")

    results = []
    progress = Progress(len(model_pairs) * len(step_sizes))

    def _failed(model_name, step, tile_size=0):
        return {"model": model_name, "step": step, "tile_size": tile_size,
                "batch_size": 0, "fps": 0.0,
                "num_tiles_per_frame": 0, "time_per_frame_ms": 0.0}

    for pth, cfg in model_pairs:
        model_name = os.path.splitext(os.path.basename(pth))[0]
        short_name = model_name.replace("allclass_", "")

        for step in step_sizes:
            progress.label = f"{short_name} step={step} compiling"
            progress.draw()
            try:
                clf, warnings = build_runtime_classifier(pth, cfg, step, preset)
            except Exception as exc:
                progress.log(f"[skip] {model_name}: load failed: {exc}")
                results.append(_failed(model_name, step))
                progress.advance()
                continue
            for warning in warnings:
                progress.log(f"[warn] {model_name}: {warning}")

            try:
                r = benchmark_model_step(clf, frames, preset, DEVICE,
                                         on_frame=lambda f: progress.draw(f, force=False))
                progress.log(
                    f"  {short_name:<30}  step={step:3d}  tiles={r['num_tiles']:5d}  "
                    f"fps={r['fps']:7.2f}  t/frame={r['time_per_frame_ms']:.1f}ms"
                )
                results.append({
                    "model":               model_name,
                    "step":                step,
                    "tile_size":           clf.tile_size,
                    "batch_size":          r["batch_size"],
                    "fps":                 round(r["fps"], 4),
                    "num_tiles_per_frame": r["num_tiles"],
                    "time_per_frame_ms":   round(r["time_per_frame_ms"], 2),
                })
            except Exception as exc:
                progress.log(f"  {short_name}  step={step}  [error] {exc}")
                results.append(_failed(model_name, step, clf.tile_size))
            progress.advance()

            del clf
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print()

    if not results:
        print("No results collected.")
        sys.exit(1)

    save_csv(results, f"{args.out}.csv")
    plot_results(results, f"{args.out}.png")


if __name__ == "__main__":
    main()
