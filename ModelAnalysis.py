#!/usr/bin/python3
"""
ModelAnalysis.py

All-in-one analysis tool that orchestrates ModelAnalysisEvaluate.py and
ModelAnalysisReport.py to produce a self-contained HTML report with all
metrics, confusion matrices, and ensemble recommendations.

Usage:
    python3 ModelAnalysis.py <dataset_dir> [models_dir] [--batch <N>] [--fp16] [--out <dir>]
                            [--metric <metric>] [--no-bench]
                            [--bench-batches B1,B2,...] [--bench-steps S1,S2,...]
                            [--image-width W] [--image-height H]

    dataset_dir : ImageFolder-style directory or directory containing dataset.h5
    models_dir  : directory containing *.pth + matching *.json  (default: ".")
    --batch N   : inference batch size  (default: 64)
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
from ModelAnalysisReport import ReportConfig, run_report

DEFAULT_METRIC = "balanced_accuracy"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("models_dir", nargs="?", default=".")
    ap.add_argument("--batch",         type=int, default=64)
    ap.add_argument("--fp16",          action="store_true")
    ap.add_argument("--out",           default=None)
    ap.add_argument("--metric",        default=DEFAULT_METRIC,
                    choices=["balanced_accuracy", "accuracy"])
    ap.add_argument("--no-bench",      action="store_true",
                    help="Skip the batch×step throughput benchmark")
    ap.add_argument("--bench-batches", default=None,
                    help="Comma-separated batch sizes for benchmark (default: 1,2,4,8,16,32,64)")
    ap.add_argument("--bench-steps",   default=None,
                    help="Comma-separated tile step sizes for benchmark (default: 1,2,4,8,16)")
    ap.add_argument("--image-width",   type=int, default=1224,
                    help="Live image width for benchmark tile count (default: 1224)")
    ap.add_argument("--image-height",  type=int, default=1024,
                    help="Live image height for benchmark tile count (default: 1024)")
    args = ap.parse_args()

    bench_batch_sizes = (
        [int(x) for x in args.bench_batches.split(",")]
        if args.bench_batches else None
    )
    bench_step_sizes = (
        [int(x) for x in args.bench_steps.split(",")]
        if args.bench_steps else None
    )

    # ── Phase 1: GPU evaluation ──────────────────────────────────────────────
    eval_cfg = EvalConfig(
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
    out_dir = run_evaluation(eval_cfg)

    # ── Phase 2: Report generation ───────────────────────────────────────────
    report_cfg = ReportConfig(
        analysis_dir = out_dir,
        metric       = args.metric,
    )
    run_report(report_cfg)


if __name__ == "__main__":
    main()
