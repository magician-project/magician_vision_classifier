#!/usr/bin/python3
"""
ModelAnalysis.py

All-in-one analysis tool that orchestrates ModelAnalysisEvaluate.py (or
ModelAnalysisEvaluateRawDataset.py for raw frame datasets) and
ModelAnalysisReport.py to produce a self-contained HTML report with all
metrics, confusion matrices, and ensemble recommendations.

Usage:
    python3 ModelAnalysis.py <dataset_dir> [models_dir] [--batch <N>] [--fp16] [--out <dir>]
                            [--metric <metric>] [--no-bench]
                            [--bench-batches B1,B2,...] [--bench-steps S1,S2,...]
                            [--image-width W] [--image-height H]

    dataset_dir : ImageFolder-style dir, dataset.h5, OR raw frame dir
                  (colorFrame_*.png/pnm + matching .json annotation files)
    models_dir  : directory containing *.pth + matching *.json  (default: ".")
    --batch N   : inference batch size  (default: 64 for tiled, 256 for raw)
    --fp16      : run inference under autocast FP16
    --out dir   : output directory for report + assets  (default: "model_analysis_<timestamp>")
    --metric    : ensemble optimisation target (balanced_accuracy | accuracy)
    --no-bench  : skip the batch×step throughput benchmark
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the script's directory is on sys.path so sibling imports work
# regardless of the working directory the user invokes from.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from ModelAnalysisEvaluate import EvalConfig, run_evaluation
from ModelAnalysisEvaluateRawDataset import RawEvalConfig, run_evaluation as run_evaluation_raw, _is_raw_dataset_dir
from ModelAnalysisReport import ReportConfig, run_report

DEFAULT_METRIC = "balanced_accuracy"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("models_dir", nargs="?", default=".")
    ap.add_argument("--batch",         type=int, default=None,
                    help="Inference batch size (default: 64 for tiled datasets, 256 for raw)")
    ap.add_argument("--fp16",          action="store_true")
    ap.add_argument("--out",           default=None)
    ap.add_argument("--metric",        default=DEFAULT_METRIC,
                    choices=["balanced_accuracy", "accuracy"])
    ap.add_argument("--no-bench",      action="store_true",
                    help="Skip the batch×step throughput benchmark")
    ap.add_argument("--bench-batches", default=None,
                    help="Comma-separated batch sizes for benchmark (tiled datasets only)")
    ap.add_argument("--bench-steps",   default=None,
                    help="Comma-separated tile step sizes for benchmark (default: 12,16,32)")
    ap.add_argument("--image-width",   type=int, default=1224,
                    help="Live image width for benchmark tile count (default: 1224)")
    ap.add_argument("--image-height",  type=int, default=1024,
                    help="Live image height for benchmark tile count (default: 1024)")
    ap.add_argument("--max-frames",    type=int, default=None,
                    help="Raw dataset only: cap number of frames evaluated (default: all)")
    args = ap.parse_args()

    bench_batch_sizes = (
        [int(x) for x in args.bench_batches.split(",")]
        if args.bench_batches else None
    )
    bench_step_sizes = (
        [int(x) for x in args.bench_steps.split(",")]
        if args.bench_steps else [12, 16, 32]
    )

    # ── Phase 1: GPU evaluation ──────────────────────────────────────────────
    if _is_raw_dataset_dir(args.dataset_dir):
        print(f"  Detected raw frame dataset: {args.dataset_dir}")
        raw_cfg = RawEvalConfig(
            raw_dataset_dir  = args.dataset_dir,
            models_dir       = args.models_dir,
            batch            = args.batch if args.batch is not None else 256,
            fp16             = args.fp16,
            out              = args.out,
            no_bench         = args.no_bench,
            bench_step_sizes = bench_step_sizes,
            image_width      = args.image_width,
            image_height     = args.image_height,
            max_frames       = args.max_frames,
        )
        out_dir = run_evaluation_raw(raw_cfg)
    else:
        eval_cfg = EvalConfig(
            dataset_dir       = args.dataset_dir,
            models_dir        = args.models_dir,
            batch             = args.batch if args.batch is not None else 64,
            fp16              = args.fp16,
            out               = args.out,
            no_bench          = args.no_bench,
            bench_batch_sizes = bench_batch_sizes,
            bench_step_sizes  = bench_step_sizes,
            image_width       = args.image_width,
            image_height      = args.image_height,
        )
        out_dir = run_evaluation(eval_cfg)

    # ── Phase 2: Report generation ───────────────────────────────────────────
    report_cfg = ReportConfig(
        analysis_dir = out_dir,
        metric       = args.metric,
    )
    run_report(report_cfg)


if __name__ == "__main__":
    main()
