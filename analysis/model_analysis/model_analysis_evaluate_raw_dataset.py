#!/usr/bin/python3
"""
ModelAnalysisEvaluateRawDataset.py

Run GPU inference directly on a raw dataset directory containing polarization
frames (colorFrame_0_x.png / colorFrame_0_x.pnm) with paired annotation JSON
files (colorFrame_0_x.json / colorFrame_0_x.pnm.json).

Unlike ModelAnalysisEvaluate.py which works on pre-tiled ImageFolder datasets,
this module evaluates the full live tiling pipeline — loading each full frame,
debayering it to RGBA, sliding a tile window at each requested step size, and
comparing per-tile predictions to ground-truth point annotations.

This naturally measures both accuracy AND effective FPS across step sizes,
because the tile count (and therefore inference time per frame) varies with step.

Output layout (same as ModelAnalysisEvaluate.py so ModelAnalysisReport.py can
consume either):
    <out_dir>/raw_data/
        manifest.json               — dataset info, class list, evaluated model list
        <model_name>_meta.json      — per-model metadata + step-size sweep data
        <model_name>_probs.npz      — y_true (N,) and probs_aligned (N, K) float32

Usage:
    python3 ModelAnalysisEvaluateRawDataset.py <raw_dataset_dir> [models_dir]
            [--batch N] [--fp16] [--out <dir>] [--no-bench]
            [--bench-steps S1,S2,...] [--image-width W] [--image-height H]

    raw_dataset_dir : directory containing colorFrame_*.png/pnm + matching .json files
    models_dir      : directory containing *.pth + matching *.json  (default: ".")
    --batch N       : inference batch size for tile classification  (default: 256)
    --fp16          : run inference under autocast FP16
    --out dir       : parent output directory  (default: "model_analysis_<timestamp>")
    --no-bench      : skip the per-step throughput + accuracy sweep
    --bench-steps   : comma-separated step sizes to evaluate  (default: 1,2,4,8,16)
    --image-width W : expected image width after debayering  (default: 1224)
    --image-height H: expected image height after debayering (default: 1024)
"""

import datetime
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mvc.core.read_data import readPolarPNMToRGBALive
from mvc.core.lit_classifier import Classifier
from mvc.core.config import load_hyperparameters
from mvc.evaluate import (
    get_clean_class_id,
    get_state_dict_from_checkpoint,
    BENCHMARK_STEP_SIZES,
)

_DEFAULT_BATCH    = 256
_WARMUP_FRAMES    = 2
_N_TIMING_FRAMES  = 10


@dataclass
class RawEvalConfig:
    raw_dataset_dir: str
    models_dir: str = "."
    batch: int = _DEFAULT_BATCH
    fp16: bool = False
    out: str | None = None
    no_bench: bool = False
    bench_step_sizes: list[int] | None = None
    image_width: int = 1224
    image_height: int = 1024
    max_frames: int | None = None   # cap number of frames evaluated (None = all)


# ---------------------------------------------------------------------------
# Class name normalisation
# ---------------------------------------------------------------------------

def _normalize_class(name: str) -> str:
    """
    Normalise an annotation class name to the model's convention.

    Empty string or any form of "clean" → "class_clean".
    Other names get a "class_" prefix and are lowercased to match model JSON.
    """
    low = name.strip().lower()
    if not low or low == "clean":
        return "class_clean"
    if low.startswith("class_"):
        return low
    return "class_" + low


def _build_raw_class_mapping(
    annotation_classes: list[str],
    model_classes: list[str],
    model_name: str,
) -> dict[int, list[int]]:
    """
    Map each annotation class index → list of model class indices.

    Annotation classes are already normalised (e.g. "class_clean", "class_welding").
    Model classes are lowercased before comparison.

    For allclass_* models: prefix match — "class_welding" maps to every model
    class whose lowercase name starts with "class_welding" (e.g. class_WeldingClassA,
    class_WeldingClassB). This lets a type-only annotation cover all severities.

    For binary_* models: "class_clean" → clean class; any other annotation →
    the single non-clean model class (class_defect or equivalent).
    """
    is_binary  = model_name.startswith("binary_")
    mc_lower   = [c.lower() for c in model_classes]
    clean_idx  = next((i for i, c in enumerate(mc_lower) if c == "class_clean"), None)
    defect_idx = [i for i, c in enumerate(mc_lower) if c != "class_clean"]

    mapping: dict[int, list[int]] = {}
    for ann_idx, ann_cls in enumerate(annotation_classes):
        ann_low = ann_cls.lower()
        if ann_low == "class_clean":
            if clean_idx is not None:
                mapping[ann_idx] = [clean_idx]
        elif is_binary:
            if defect_idx:
                mapping[ann_idx] = [defect_idx[0]]
        else:
            hits = [i for i, mc in enumerate(mc_lower) if mc.startswith(ann_low)]
            if hits:
                mapping[ann_idx] = hits
            elif defect_idx:
                # annotation type not in model — treat as generic defect
                mapping[ann_idx] = defect_idx[:1]
    return mapping


def _aggregate_probs(
    p_model: np.ndarray,
    ann_to_model: dict[int, list[int]],
    num_ann_classes: int,
) -> np.ndarray:
    """
    Collapse model-space probabilities into annotation-class space.

    Each annotation class gets the SUM of its mapped model class probabilities,
    so probabilities for WeldingClassA + WeldingClassB merge into "class_welding".
    """
    p_ann = np.zeros(num_ann_classes, dtype=np.float32)
    for ann_idx, model_indices in ann_to_model.items():
        for mi in model_indices:
            if mi < len(p_model):
                p_ann[ann_idx] += p_model[mi]
    return p_ann


# Raw dataset discovery
# ---------------------------------------------------------------------------

def _find_annotation(image_path: str) -> str | None:
    """Return the annotation JSON path for an image, trying both sidecar conventions."""
    # <image>.json  (e.g. colorFrame_0_00001.png.json or colorFrame_0_00001.json)
    p1 = image_path + ".json"
    if os.path.isfile(p1):
        return p1
    # <basename>.pnm.json  (legacy: image was .pnm, annotation kept .pnm.json)
    base, _ext = os.path.splitext(image_path)
    p2 = base + ".pnm.json"
    if os.path.isfile(p2):
        return p2
    # Also try <basename>.json (no extension appended)
    p3 = base + ".json"
    if os.path.isfile(p3):
        return p3
    return None


def _discover_frames(raw_dir: str) -> list[tuple[str, str | None]]:
    """
    Scan *raw_dir* for image files, optionally paired with annotation JSON.

    Images without a matching JSON are included with ann_path=None; all their
    tiles will be treated as ground-truth "clean".

    Returns list of (image_path, json_path_or_None) sorted by filename.
    """
    pairs = []
    for entry in sorted(os.listdir(raw_dir)):
        if not entry.lower().endswith((".png", ".pnm", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(raw_dir, entry)
        ann_path = _find_annotation(img_path)
        pairs.append((img_path, ann_path))
    return pairs


def _is_raw_dataset_dir(path: str) -> bool:
    """Return True if *path* looks like a raw frame dataset (has image files)."""
    if not os.path.isdir(path):
        return False
    try:
        frames = _discover_frames(path)
        return len(frames) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_rgba_frame(image_path: str) -> np.ndarray | None:
    """
    Load an image from disk and convert to 4-channel RGBA via debayering.

    Handles both .pnm polarization images and standard RGB/RGBA images.
    Returns an (H, W, 4) uint8 array, or None on failure.
    """
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    if raw.ndim == 2:
        # Grayscale/raw polarization — debayer to 4-channel RGBA
        rgba = readPolarPNMToRGBALive(raw)
        rgba = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        return rgba

    if raw.ndim == 3:
        if raw.shape[2] == 3:
            # BGR → BGRA
            return cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
        if raw.shape[2] == 4:
            return raw

    return None


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _load_annotations(json_path: str | None) -> tuple[list, list]:
    """
    Return (point_clicks, combined_classes) from an annotation JSON file.

    combined_classes merges pointClasses and pointSeverities: "Welding" + "Class A"
    → "WeldingClassA" (space stripped from severity so it matches model class names).
    Returns ([], []) if json_path is None.
    """
    if json_path is None:
        return [], []
    with open(json_path) as f:
        data = json.load(f)
    clicks     = data.get("pointClicks", [])
    classes    = data.get("pointClasses", [])
    severities = data.get("pointSeverities", [])
    combined = [
        c + severities[i].replace(" ", "")
        if i < len(severities) and severities[i]
        else c
        for i, c in enumerate(classes)
    ]
    return clicks, combined


def _assign_tile_class(
    start_x: int, end_x: int,
    start_y: int, end_y: int,
    point_clicks: list,
    point_classes: list,
    full_to_debayer_scale: int = 2,
) -> str:
    """
    Return the class label for a tile given annotation clicks.

    Annotation coordinates are in the full (pre-debayer) image space; divide by
    *full_to_debayer_scale* (default 2) to map to debayered tile coordinates.
    Returns "" (empty) for unannotated tiles, treated as "clean".
    """
    label = ""
    for (xFull, yFull), cls in zip(point_clicks, point_classes):
        x = xFull // full_to_debayer_scale
        y = yFull // full_to_debayer_scale
        if start_x <= x <= end_x and start_y <= y <= end_y:
            label = label + cls
    return label


# ---------------------------------------------------------------------------
# Streaming tile extraction + inference
# ---------------------------------------------------------------------------

def _iter_tiles(
    rgba: np.ndarray,
    point_clicks: list,
    point_classes: list,
    tile_size: int,
    step: int,
):
    """
    Yield (tile, label) pairs by sliding a window over *rgba* (H, W, 4).

    Yields views into *rgba* one tile at a time — no large intermediate array.
    """
    H, W = rgba.shape[:2]
    for y in range(0, H - tile_size, step):
        for x in range(0, W - tile_size, step):
            tile = rgba[y:y + tile_size, x:x + tile_size]
            cls  = _assign_tile_class(x, x + tile_size, y, y + tile_size,
                                      point_clicks, point_classes)
            yield tile, cls


@torch.no_grad()
def _infer_frame(
    clf: Classifier,
    rgba: np.ndarray,
    point_clicks: list,
    point_classes: list,
    tile_size: int,
    step: int,
    batch_size: int,
    device: str,
    autocast_ctx,
):
    """
    Classify every tile in a frame, yielding (label, prob_vector) pairs.

    Processes tiles in batches of *batch_size* so memory use is bounded to
    batch_size × tile_size² × 4 × 4 bytes regardless of step size or image size.
    """
    buf_tiles:  list[np.ndarray] = []
    buf_labels: list[str]        = []

    def _flush():
        arr = np.stack(buf_tiles, axis=0)          # (B, H, W, C) uint8
        # Keep uint8: Classifier.build_input_features() applies the /255 on the
        # GPU. Casting to float32 here skips it and feeds the model 0-255.
        t   = (torch.from_numpy(arr)
                    .permute(0, 3, 1, 2)
                    .to(device)
                    .contiguous())
        with autocast_ctx:
            logits = clf(t)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return zip(buf_labels, probs)

    for tile, label in _iter_tiles(rgba, point_clicks, point_classes, tile_size, step):
        buf_tiles.append(tile.copy())
        buf_labels.append(label)
        if len(buf_tiles) == batch_size:
            yield from _flush()
            buf_tiles  = []
            buf_labels = []

    if buf_tiles:
        yield from _flush()


# ---------------------------------------------------------------------------
# Model helpers (shared with ModelAnalysisEvaluate)
# ---------------------------------------------------------------------------

def _instantiate(config_json: dict, class_names: list[str]) -> Classifier:
    # Single source of truth: Classifier.from_config reads every knob from
    # config["hparams"], the nesting the trainer writes. This block used to read
    # AoLP/DoLP/Unpolarized/Max/Min/RangePolarization from the config TOP LEVEL,
    # where no config has ever carried them (167 have them under hparams, 0 at the
    # top), so it silently built a 4-channel model for every 5-channel run. It also
    # never passed monochrome, custom_channels, custom_res_blocks, pretrained or
    # timm_stem_stride -- see test_classifier_from_config.py.
    return Classifier.from_config(config_json, num_classes=len(class_names),
                                  clean_class=get_clean_class_id(class_names))


def _load_weights(clf: Classifier, pth_path: str, device: str) -> Classifier:
    from mvc.evaluate import get_state_dict_from_checkpoint
    sd = get_state_dict_from_checkpoint(pth_path, device)
    clf.load_state_dict(sd, strict=False)
    return clf.to(device).eval()


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _discover_pairs(models_dir: str) -> list[tuple[str, str]]:
    pairs = []
    for f in sorted(Path(models_dir).glob("*.pth")):
        cfg = f.with_suffix(".json")
        if cfg.is_file():
            pairs.append((str(f), str(cfg)))
        else:
            print(f"  [WARN] No matching .json for {f.name} — skipping.")
    return pairs


# ---------------------------------------------------------------------------
# Per-step accuracy + FPS evaluation
# ---------------------------------------------------------------------------

def _quick_bal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.sum() > 0:
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _eval_step(
    clf: Classifier,
    frames: list[tuple[np.ndarray, list, list]],
    step: int,
    tile_size: int,
    batch_size: int,
    device: str,
    autocast_ctx,
    dataset_classes: list[str],
    ann_to_model: dict[int, list[int]],
    n_timing_frames: int = _N_TIMING_FRAMES,
    collect_probs: bool = False,
) -> dict:
    """
    Evaluate accuracy + effective FPS for one step size.

    *frames* is a list of (rgba_np, point_clicks, point_classes) tuples.
    *ann_to_model* maps annotation class index → list of model class indices
    (built by _build_raw_class_mapping).
    If *collect_probs* is True, the returned dict also contains "y_true_all"
    and "probs_all" so the caller can save the .npz without a second pass.
    Returns a dict with accuracy, balanced_accuracy, effective_fps, etc.
    """
    ds_name_to_idx = {n: i for i, n in enumerate(dataset_classes)}
    num_ann        = len(dataset_classes)

    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    y_true_all:  list[int] = []
    probs_all:   list[np.ndarray] = []

    # Accuracy pass — streaming: only batch_size tiles live in memory at once
    for rgba, point_clicks, point_classes in tqdm(
            frames, desc=f"      step={step}", unit="frame", leave=False, dynamic_ncols=True):
        for label, p in _infer_frame(clf, rgba, point_clicks, point_classes,
                                     tile_size, step, batch_size, device, autocast_ctx):
            gt_idx = ds_name_to_idx.get(_normalize_class(label))
            if gt_idx is None or gt_idx not in ann_to_model:
                continue
            p_ann = _aggregate_probs(p, ann_to_model, num_ann)
            y_true_list.append(gt_idx)
            y_pred_list.append(int(p_ann.argmax()))
            if collect_probs:
                y_true_all.append(gt_idx)
                probs_all.append(p_ann)

    acc     = 0.0
    bal_acc = 0.0
    if y_true_list:
        yt = np.array(y_true_list, np.int64)
        yp = np.array(y_pred_list, np.int64)
        acc     = float((yt == yp).mean())
        bal_acc = _quick_bal_acc(yt, yp)

    # FPS timing pass — pre-collect a capped number of batches into CPU RAM
    # so disk/decode latency doesn't inflate the timer.
    timing_batches: list[torch.Tensor] = []
    for rgba, point_clicks, point_classes in frames[:n_timing_frames]:
        buf: list[np.ndarray] = []
        for tile, _ in _iter_tiles(rgba, point_clicks, point_classes, tile_size, step):
            buf.append(tile.copy())
            if len(buf) == batch_size:
                arr = np.stack(buf, axis=0)
                timing_batches.append(
                    torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
                )
                buf = []
                if len(timing_batches) >= n_timing_frames * 4:
                    break  # cap total pre-loaded batches
            if len(timing_batches) >= n_timing_frames * 4:
                break
        if buf:
            arr = np.stack(buf, axis=0)
            timing_batches.append(
                torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            )

    fps = 0.0
    if timing_batches:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch_t in timing_batches:
            with autocast_ctx:
                clf(batch_t.to(device))
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_tiles = sum(b.shape[0] for b in timing_batches)
        tiles_per_frame_count = ((frames[0][0].shape[0] - tile_size) // step) * \
                                ((frames[0][0].shape[1] - tile_size) // step)
        fps = total_tiles / max(elapsed * tiles_per_frame_count, 1e-9)

    # Analytical tile count for this step (matches liveClassifierTorch formula)
    first_rgba = frames[0][0]
    H, W = first_rgba.shape[:2]
    num_tiles = ((H - tile_size) // step) * ((W - tile_size) // step)

    result = {
        "step_size"          : step,
        "num_tiles_per_frame": num_tiles,
        "effective_fps"      : fps,
        "accuracy"           : acc,
        "balanced_accuracy"  : bal_acc,
        "num_annotated_tiles": len(y_true_list),
        "oom"                : False,
    }
    if collect_probs:
        result["y_true_all"] = y_true_all
        result["probs_all"]  = probs_all
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(cfg: RawEvalConfig) -> str:
    """
    Run full raw-dataset evaluation pipeline and return the output directory path.
    """
    bench_step_sizes = cfg.bench_step_sizes or BENCHMARK_STEP_SIZES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.out) if cfg.out else Path(f"model_analysis_{timestamp}")
    raw_dir   = out_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model Analysis — Evaluate Raw Dataset")
    print(f"{'='*60}")
    print(f"  Dataset    : {cfg.raw_dataset_dir}")
    print(f"  Models dir : {cfg.models_dir}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {cfg.batch}")
    print(f"  FP16       : {cfg.fp16}")
    print(f"  Steps      : {bench_step_sizes}")
    print(f"  Output     : {out_dir}")

    frames_list = _discover_frames(cfg.raw_dataset_dir)
    if not frames_list:
        print(f"\n[ERROR] No image files found in: {cfg.raw_dataset_dir}")
        sys.exit(1)
    n_annotated = sum(1 for _, a in frames_list if a is not None)
    n_clean_only = len(frames_list) - n_annotated
    print(f"\n  Found {len(frames_list)} frame(s): "
          f"{n_annotated} annotated, {n_clean_only} clean-only (no JSON).")

    pairs = _discover_pairs(cfg.models_dir)
    if not pairs:
        print(f"[ERROR] No (.pth, .json) pairs found in: {cfg.models_dir}")
        sys.exit(1)
    print(f"  Found {len(pairs)} model(s):")
    for pth, _ in pairs:
        print(f"    {Path(pth).name}")

    # Pre-load all frames into RAM (debayered RGBA)
    print("\n  Loading frames...")
    loaded_frames: list[tuple[np.ndarray, list, list]] = []
    for img_path, ann_path in tqdm(frames_list, desc="    Loading", unit="frame",
                                   dynamic_ncols=True):
        rgba = _load_rgba_frame(img_path)
        if rgba is None:
            print(f"  [WARN] Could not load image: {img_path} — skipping.")
            continue
        point_clicks, point_classes = _load_annotations(ann_path)
        loaded_frames.append((rgba, point_clicks, point_classes))

    if not loaded_frames:
        print("[ERROR] No frames could be loaded.")
        sys.exit(1)
    if cfg.max_frames and len(loaded_frames) > cfg.max_frames:
        loaded_frames = loaded_frames[:cfg.max_frames]
        print(f"  Capped to {cfg.max_frames} frame(s) (--max-frames).")
    print(f"  Loaded {len(loaded_frames)} frame(s). "
          f"Frame size: {loaded_frames[0][0].shape[1]}×{loaded_frames[0][0].shape[0]} px")

    # Collect all class names from annotations, normalised to model convention
    annotation_classes: set[str] = set()
    for _, clicks, classes in loaded_frames:
        annotation_classes.update(_normalize_class(c) for c in classes)
    annotation_classes.add("class_clean")
    dataset_classes = sorted(annotation_classes)
    # Put "class_clean" first to match typical convention
    if "class_clean" in dataset_classes:
        dataset_classes.remove("class_clean")
        dataset_classes = ["class_clean"] + dataset_classes
    print(f"  Annotation classes ({len(dataset_classes)}): {dataset_classes}")

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.fp16 and device == "cuda"
        else torch.amp.autocast(device_type="cuda", enabled=False)
        if device == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    evaluated_models: list[str] = []

    for midx, (pth_path, cfg_path) in enumerate(pairs):
        model_name  = Path(pth_path).stem
        print(f"\n  [{midx+1}/{len(pairs)}] {model_name}")

        config_json   = load_hyperparameters(cfg_path)
        model_classes = [c.lower() for c in config_json.get("classes", dataset_classes)]
        tile_size     = config_json["hparams"].get("tile_size", 48)

        ann_to_model = _build_raw_class_mapping(dataset_classes, model_classes, model_name)
        if not ann_to_model:
            print("    [ERROR] no annotation classes could be mapped to model — skipping.")
            continue
        unmapped = [dataset_classes[i] for i in range(len(dataset_classes)) if i not in ann_to_model]
        if unmapped:
            print(f"    [WARN] annotation classes not mapped to any model class: {unmapped}")

        try:
            clf = _instantiate(config_json, model_classes)
            clf = _load_weights(clf, pth_path, device)
        except Exception as e:
            print(f"    [ERROR] Could not load model: {e}")
            continue

        num_params = _count_params(clf)
        print(f"    Tile size : {tile_size}")
        print(f"    Params    : {num_params:,}")

        # GPU warmup — run one batch per warmup frame (avoids materialising all tiles)
        print("    Warming up GPU...")
        warmup_step = bench_step_sizes[0]
        for rgba, clicks, classes in loaded_frames[:_WARMUP_FRAMES]:
            # consume just enough tiles to fill one batch
            for _ in _infer_frame(clf, rgba, clicks, classes,
                                  tile_size, warmup_step, cfg.batch, device, autocast_ctx):
                break  # one batch is enough per warmup frame
        if device == "cuda":
            torch.cuda.synchronize()

        # Per-step evaluation.
        # The first step also collects per-tile probs for the .npz (no second pass).
        bench_data: list[dict] = []
        if not cfg.no_bench:
            print(f"    Evaluating {len(bench_step_sizes)} step size(s)...")
            for sidx, step in enumerate(bench_step_sizes):
                result = _eval_step(
                    clf, loaded_frames, step, tile_size,
                    cfg.batch, device, autocast_ctx,
                    dataset_classes, ann_to_model,
                    collect_probs=(sidx == 0),
                )
                bench_data.append(result)
                print(f"      step={step:3d} | "
                      f"tiles/frame={result['num_tiles_per_frame']:6d} | "
                      f"acc={result['accuracy']:.4f} | "
                      f"bal_acc={result['balanced_accuracy']:.4f} | "
                      f"fps={result['effective_fps']:.1f}")
        else:
            result = _eval_step(
                clf, loaded_frames, bench_step_sizes[0], tile_size,
                cfg.batch, device, autocast_ctx,
                dataset_classes, ann_to_model,
                collect_probs=True,
            )
            bench_data.append(result)

        # Use the smallest-step result as the canonical accuracy (most tiles → best coverage)
        best_acc    = bench_data[0]["accuracy"]
        best_bal    = bench_data[0]["balanced_accuracy"]
        best_n_ann  = bench_data[0]["num_annotated_tiles"]

        # Best FPS across steps
        ok_pts = [r for r in bench_data if not r.get("oom")]
        best_fps_pt = max(ok_pts, key=lambda r: r["effective_fps"]) if ok_pts else {}

        print(f"    Accuracy  : {best_acc:.4f}")
        print(f"    BalAcc    : {best_bal:.4f}")
        print(f"    Ann tiles : {best_n_ann:,}")
        if best_fps_pt:
            print(f"    Peak FPS  : {best_fps_pt['effective_fps']:.1f} "
                  f"(step={best_fps_pt['step_size']})")

        # Save probs collected during the first step (no extra inference pass needed)
        y_true_all = bench_data[0].get("y_true_all", [])
        probs_all  = bench_data[0].get("probs_all",  [])
        if y_true_all:
            np.savez_compressed(
                raw_dir / f"{model_name}_probs.npz",
                y_true = np.array(y_true_all, dtype=np.int64),
                probs  = np.array(probs_all,  dtype=np.float32),
            )
        # Strip probs from bench_data before serialising to JSON
        for r in bench_data:
            r.pop("y_true_all", None)
            r.pop("probs_all",  None)

        meta = {
            "name"               : model_name,
            "model_type"         : config_json.get("model", "unknown"),
            "tile_size"          : tile_size,
            "num_params"         : num_params,
            "ms_per_sample"      : 0.0,  # not applicable; see benchmark for FPS
            "accuracy"           : best_acc,
            "balanced_accuracy"  : best_bal,
            "num_aligned_samples": best_n_ann,
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
        "dataset"        : cfg.raw_dataset_dir,
        "dataset_classes": dataset_classes,
        "num_samples"    : len(loaded_frames),
        "fp16"           : cfg.fp16,
        "batch_size"     : cfg.batch,
        "models"         : evaluated_models,
        "raw_dataset"    : True,
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
    ap.add_argument("raw_dataset_dir")
    ap.add_argument("models_dir", nargs="?", default=".")
    ap.add_argument("--batch",       type=int, default=_DEFAULT_BATCH)
    ap.add_argument("--fp16",        action="store_true")
    ap.add_argument("--out",         default=None)
    ap.add_argument("--no-bench",    action="store_true",
                    help="Skip the per-step throughput + accuracy sweep")
    ap.add_argument("--bench-steps",  default=None,
                    help="Comma-separated tile step sizes (default: 1,2,4,8,16)")
    ap.add_argument("--image-width",  type=int, default=1224)
    ap.add_argument("--image-height", type=int, default=1024)
    ap.add_argument("--max-frames",   type=int, default=None,
                    help="Limit number of frames evaluated (default: all)")
    args = ap.parse_args()

    bench_step_sizes = (
        [int(x) for x in args.bench_steps.split(",")]
        if args.bench_steps else None
    )

    cfg = RawEvalConfig(
        raw_dataset_dir  = args.raw_dataset_dir,
        models_dir       = args.models_dir,
        batch            = args.batch,
        fp16             = args.fp16,
        out              = args.out,
        no_bench         = args.no_bench,
        bench_step_sizes = bench_step_sizes,
        image_width      = args.image_width,
        image_height     = args.image_height,
        max_frames       = args.max_frames,
    )
    out_dir = run_evaluation(cfg)
    print(f"\n{'='*60}")
    print(f"  Run:  python3 ModelAnalysisReport.py {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
