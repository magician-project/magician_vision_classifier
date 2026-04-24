#!/usr/bin/python3
"""
ModelAnalysisReport.py

Load raw evaluation data produced by ModelAnalysisEvaluate.py and generate
a self-contained HTML report with all metrics, confusion matrices, efficiency
plots, and ensemble recommendations.  Re-running this script is fast (no GPU
required) so you can add plots or tweak styling without re-evaluating models.

Usage:
    python3 ModelAnalysisReport.py <analysis_dir> [--metric <metric>]

    analysis_dir : directory created by ModelAnalysisEvaluate.py that contains
                   a raw_data/ subdirectory with manifest.json and model files.
    --metric     : optimisation target for ensemble search
                   (balanced_accuracy | accuracy, default: balanced_accuracy)

Output (written into analysis_dir):
    index.html
    model_analysis.json
    <model_name>_confusion_row_normalized.png
    <model_name>_benchmark_3d.png   (if benchmark data is present)
    efficiency_comparison.png
"""

import base64
import datetime
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

MIN_GAIN_THRESHOLD = 0.0005
MAX_DROP_THRESHOLD = 0.005
STRATEGIES         = ["soft", "majority", "confidence_weighted"]
DEFAULT_METRIC     = "balanced_accuracy"


@dataclass
class ReportConfig:
    analysis_dir: str
    metric: str = DEFAULT_METRIC


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """
    Compute per-class and aggregate classification metrics.

    Returns dict with accuracy, balanced_accuracy, num_samples, per_class list,
    cm (raw confusion matrix), and cm_norm (row-normalised).
    """
    K      = len(class_names)
    labels = list(range(K))
    if len(y_true) == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0,
                "num_samples": 0, "per_class": [], "cm": [], "cm_norm": []}

    acc     = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm      = confusion_matrix(y_true, y_pred, labels=labels)
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
# Plots
# ---------------------------------------------------------------------------

def _save_confusion_png(cm_norm: list[list[float]], class_names: list[str],
                        title: str, out_path: str):
    """Render a row-normalised confusion matrix heatmap and save to disk."""
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
    """Read a PNG file and return its base64-encoded string for HTML embedding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _save_benchmark_3d_plot(bench_results: list[dict], model_name: str, out_path: str):
    """
    Render a 3-D scatter of throughput benchmark results (batch × step → FPS).
    Colour encodes balanced accuracy.  OOM points are marked with red X.
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
        ax.scatter(fx, fy, fz, c="red", marker="x", s=120, linewidths=2, zorder=5)
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
    Save a 3-panel efficiency figure:
      - Scatter: num_params (M) vs balanced_accuracy
      - Bar: balanced_accuracy / M_params per model
      - Grouped bar: per-class recall / M_params across models
    """
    if not model_results:
        return

    n_models = len(model_results)
    n_cls    = len(dataset_classes)

    m_params = np.array([m["num_params"] / 1_000_000 for m in model_results])
    bal_accs = np.array([m["balanced_accuracy"]       for m in model_results])
    eff      = bal_accs / np.maximum(m_params, 1e-9)

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

    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(n_models):
        ax1.scatter(m_params[i], bal_accs[i], s=110, color=colors[i], zorder=5,
                    edgecolors="k", linewidths=0.5, label=short_names[i])
        ax1.annotate(short_names[i], (m_params[i], bal_accs[i]),
                     fontsize=6, xytext=(5, 4), textcoords="offset points")
    ax1.set_xlabel("Parameters (M)", fontsize=9)
    ax1.set_ylabel("Balanced Accuracy", fontsize=9)
    ax1.set_title("Accuracy vs Model Size", fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.3)

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
# Ensemble helpers
# ---------------------------------------------------------------------------

def _ensemble_predict(probs: np.ndarray, subset: list[int], strategy: str) -> np.ndarray:
    """
    Predict class labels by combining probabilities from a subset of models.

    strategies:
        soft               — average softmax vectors, then argmax
        majority           — each model votes top-1; majority wins
        confidence_weighted — weight each model's probs by its top-1 confidence
    """
    sub = probs[subset]
    if strategy == "soft":
        return sub.mean(axis=0).argmax(axis=1).astype(np.int64)
    elif strategy == "majority":
        votes = sub.argmax(axis=2)
        S, N  = votes.shape
        K     = probs.shape[2]
        cnt   = np.zeros((N, K), dtype=np.int32)
        for s in range(S):
            np.add.at(cnt, (np.arange(N), votes[s]), 1)
        return cnt.argmax(axis=1).astype(np.int64)
    elif strategy == "confidence_weighted":
        max_c    = sub.max(axis=2, keepdims=True)
        weighted = (sub * max_c).sum(axis=0)
        return weighted.argmax(axis=1).astype(np.int64)
    raise ValueError(strategy)


def _bal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.sum() > 0:
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _score(probs, y_true, subset, strategy, ms_arr, params_arr):
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
    """Greedy forward selection: add models one at a time while metric improves."""
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
        if selected and gain < MIN_GAIN_THRESHOLD:
            break
        best_score = br[metric]
        selected.append(bc)
        remaining.remove(bc)
        steps.append({**br, "added_model": bc})
    return steps


def _greedy_backward(probs, y_true, ms_arr, params_arr, strategy, metric):
    """Greedy backward elimination: remove models while metric drop stays small."""
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
        if bd > MAX_DROP_THRESHOLD:
            break
        selected.remove(br)
        steps.append({**bt, "removed_model": br})
    return steps


def _pareto(results, metric):
    """Return the Pareto front optimising metric vs ms_sequential."""
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

.bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
.bar-bg   { flex: 1; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width .3s; }

.cm-grid { display: flex; flex-wrap: wrap; gap: 1.2rem; }
.cm-item { text-align: center; }
.cm-item img { border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,.12);
               max-width: 100%; height: auto; }
.cm-item .cm-label { font-size: 0.78rem; color: #555; margin-top: 0.35rem; }

.tab-bar { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0; border-bottom: 2px solid #dee2e6; }
.tab-btn { padding: 0.55rem 1.1rem; border: none; background: none; cursor: pointer;
           font-size: 0.83rem; font-weight: 600; color: #555; border-radius: 6px 6px 0 0;
           border-bottom: 2px solid transparent; margin-bottom: -2px; }
.tab-btn.active { color: #0f3460; border-bottom-color: #e94560; background: #fff; }
.tab-content { display: none; }
.tab-content.active { display: block; }

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
    if value >= 0.95: return "good"
    if value >= 0.80: return "warn"
    return "bad"


def _bar(value: float, color: str = "#3b82f6", max_val: float = 1.0) -> str:
    pct = (max(0.0, value) / max_val) * 100 if max_val > 0 else 0
    pct = min(100.0, pct)
    return (f'<div class="bar-wrap">'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<span style="min-width:3.5em;font-size:.8rem;text-align:right">{value:.2f}</span>'
            f'</div>')


def _fmt_params(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)


def _build_html(meta: dict, model_results: list[dict], ensemble_results: dict,
                dataset_classes: list[str], generated: str,
                efficiency_b64: str = "") -> str:
    """Assemble the complete self-contained HTML report."""
    model_names = [m["name"] for m in model_results]

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

    best_acc   = max(m["accuracy"]          for m in model_results)
    best_bal   = max(m["balanced_accuracy"] for m in model_results)
    fastest    = min(m["ms_per_sample"]     for m in model_results)
    lightest   = min(m["num_params"]        for m in model_results)
    best_acc_n = model_names[np.argmax([m["accuracy"]          for m in model_results])]
    best_bal_n = model_names[np.argmax([m["balanced_accuracy"] for m in model_results])]
    fastest_n  = model_names[np.argmin([m["ms_per_sample"]     for m in model_results])]
    lightest_n = model_names[np.argmin([m["num_params"]        for m in model_results])]

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

    rows    = ""
    max_fps = max([m.get("fps", 1.0) for m in model_results])
    for m in sorted(model_results, key=lambda x: -x["balanced_accuracy"]):
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

    group_id  = "model_tabs"
    tab_bar   = '<div class="tab-bar">'
    tab_panes = ""
    for idx, m in enumerate(sorted(model_results, key=lambda x: -x["balanced_accuracy"])):
        tid    = f"mt_{idx}"
        active = "active" if idx == 0 else ""
        tab_bar  += (f'<button class="tab-btn {active}" '
                     f'onclick="showTab(\'{group_id}\',\'{tid}\')">'
                     f'{m["name"].replace("allclass_","").replace("binary_","⚡ ")}</button>')
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
    section_models += f"""
<h3>Per-class breakdown</h3>
<div class="card" id="{group_id}">
{tab_bar}
{tab_panes}
</div>"""

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

    section_eff = ""
    if efficiency_b64:
        section_eff = f"""
<h2 id="efficiency">&#9878; Efficiency comparison (Accuracy / Params)</h2>
<div class="card">
  <p style="font-size:.82rem;color:#555;margin-bottom:.8rem">
    <strong>Top-left</strong>: balanced accuracy vs model size.
    <strong>Top-right</strong>: overall efficiency = balanced accuracy per million parameters.
    <strong>Bottom</strong>: per-class recall per million parameters.
  </p>
  <img src="data:image/png;base64,{efficiency_b64}" alt="Efficiency comparison"
       style="max-width:100%;height:auto;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,.12)">
</div>
"""

    rec     = ensemble_results.get("recommendation", {})
    rec_min = ensemble_results.get("recommendation_min_params", {})
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

    fwd_html = ""
    for strategy in STRATEGIES:
        steps = ensemble_results.get("forward", {}).get(strategy, [])
        if not steps:
            continue
        fwd_rows = ""
        for i, s in enumerate(steps):
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
    <strong>Z-axis (FPS)</strong> = 1&nbsp;/&nbsp;(&#8968;num_tiles&nbsp;/&nbsp;batch_size&#8969;&nbsp;&times;&nbsp;time_per_batch).
    <strong>Colour</strong> = balanced accuracy (constant per model).
  </p>
  <div class="cm-grid">{bench_imgs}</div>
</div>
"""

    footer = f"""
<p style="text-align:center;font-size:.75rem;color:#999;margin:2rem 0 1rem">
  Generated by ModelAnalysisReport.py &nbsp;·&nbsp; {generated}
</p>
</div>
<script>{_JS}</script>
</body></html>"""

    return head + section_overview + section_models + section_cm + section_eff + section_ens + section_bench + footer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_report(cfg: ReportConfig):
    """
    Load raw evaluation data and generate the HTML report + JSON summary.

    Args:
        cfg: ReportConfig with analysis_dir and metric.

    Returns:
        Absolute path to the output directory containing index.html and assets.
    """
    out_dir = Path(cfg.analysis_dir)
    raw_dir = out_dir / "raw_data"

    if not (raw_dir / "manifest.json").is_file():
        print(f"[ERROR] No raw_data/manifest.json found in: {out_dir}")
        print("  Run ModelAnalysisEvaluate.py first to generate the raw data.")
        sys.exit(1)

    with open(raw_dir / "manifest.json") as f:
        manifest = json.load(f)

    dataset_classes     = manifest["dataset_classes"]
    num_dataset_classes = len(dataset_classes)
    metric              = cfg.metric

    print(f"\n{'='*60}")
    print(f"  Model Analysis — Report")
    print(f"{'='*60}")
    print(f"  Input      : {raw_dir}")
    print(f"  Dataset    : {manifest['dataset']}")
    print(f"  Classes    : {dataset_classes}")
    print(f"  Models     : {manifest['models']}")
    print(f"  Metric     : {metric}")
    print(f"  Output     : {out_dir}")

    model_results: list[dict]     = []
    probs_all:     list[np.ndarray] = []
    y_true_global: np.ndarray | None = None

    for model_name in manifest["models"]:
        meta_path  = raw_dir / f"{model_name}_meta.json"
        probs_path = raw_dir / f"{model_name}_probs.npz"

        if not meta_path.is_file() or not probs_path.is_file():
            print(f"  [WARN] Missing files for {model_name} — skipping.")
            continue

        print(f"\n  {model_name}")

        with open(meta_path) as f:
            meta = json.load(f)

        npz    = np.load(probs_path)
        y_true = npz["y_true"].astype(np.int64)
        probs  = npz["probs"].astype(np.float32)   # (N, K)
        N      = len(y_true)

        if y_true_global is None:
            y_true_global = y_true
        elif not np.array_equal(y_true_global, y_true):
            print(f"    [WARN] y_true differs from first model — excluded from ensemble.")
            probs = None

        y_pred = probs.argmax(axis=1).astype(np.int64) if probs is not None else np.zeros(N, np.int64)
        st     = _stats(y_true, y_pred, dataset_classes)
        print(f"    Accuracy  : {st['accuracy']:.4f}")
        print(f"    BalAcc    : {st['balanced_accuracy']:.4f}")
        print(f"    ms/sample : {meta['ms_per_sample']:.4f}")

        cm_path = out_dir / f"{model_name}_confusion_row_normalized.png"
        _save_confusion_png(st["cm_norm"], dataset_classes,
                            f"{model_name}  (BalAcc {st['balanced_accuracy']:.4f})",
                            str(cm_path))
        cm_b64 = _png_to_b64(str(cm_path))

        bench_data = meta.get("benchmark", [])
        bench_b64  = ""
        if bench_data:
            bench_path = out_dir / f"{model_name}_benchmark_3d.png"
            _save_benchmark_3d_plot(bench_data, model_name, str(bench_path))
            bench_b64 = _png_to_b64(str(bench_path))

        ms = meta["ms_per_sample"]
        model_results.append({
            "name"               : model_name,
            "model_type"         : meta["model_type"],
            "tile_size"          : meta["tile_size"],
            "num_params"         : meta["num_params"],
            "ms_per_sample"      : ms,
            "fps"                : 1000.0 / ms if ms > 0 else 0.0,
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

        if probs is not None:
            probs_all.append(probs)

    if not model_results:
        print("\n[ERROR] No models could be loaded.")
        sys.exit(1)

    # ── Ensemble optimisation ────────────────────────────────────────────────
    ensemble_results: dict = {}
    if len(probs_all) >= 2 and y_true_global is not None:
        print(f"\n  Running ensemble optimisation ({metric})...")

        ensemble_model_results = model_results[:len(probs_all)]

        probs_matrix = np.stack(probs_all, axis=0).astype(np.float32)   # (M, N, K)
        ms_arr       = np.array([m["ms_per_sample"] for m in ensemble_model_results], dtype=np.float64)
        params_arr   = np.array([m["num_params"]     for m in ensemble_model_results], dtype=np.int64)

        all_forward: dict[str, list] = {}
        all_pareto:  dict[str, list] = {}
        for strategy in STRATEGIES:
            print(f"    Forward selection ({strategy})...")
            steps = _greedy_forward(probs_matrix, y_true_global, ms_arr, params_arr, strategy, metric)
            all_forward[strategy] = steps
            all_pareto[strategy]  = _pareto(steps, metric)

        print("    Backward elimination (soft)...")
        bwd_steps = _greedy_backward(probs_matrix, y_true_global, ms_arr, params_arr, "soft", metric)

        all_candidates = [s for steps in all_forward.values() for s in steps]
        best       = max(all_candidates, key=lambda x: (x[metric], -x["ms_sequential"]))
        best_names = [ensemble_model_results[i]["name"] for i in best["subset"]]

        threshold    = best[metric] * 0.995
        lw_cands     = [s for s in all_candidates if s[metric] >= threshold and s["total_params"] > 0]
        lightest_ens = min(lw_cands, key=lambda x: (x["total_params"], -x[metric])) if lw_cands else best
        lw_names     = [ensemble_model_results[i]["name"] for i in lightest_ens["subset"]]

        ensemble_results = {
            "forward" : all_forward,
            "pareto"  : all_pareto,
            "backward": bwd_steps,
            "recommendation": {**best, "model_names": best_names},
            "recommendation_min_params": {**lightest_ens, "model_names": lw_names},
        }

        print(f"\n  Best ensemble ({best['strategy']}):")
        print(f"    BalAcc : {best[metric]:.4f}")
        print(f"    ms-par : {best['ms_parallel']:.3f}")
        print(f"    Models : {', '.join(best_names)}")
    else:
        print("\n  Skipping ensemble (fewer than 2 models with aligned predictions).")
        ensemble_results = {"forward": {}, "pareto": {}, "backward": [],
                            "recommendation": {}, "recommendation_min_params": {}}

    # ── JSON summary ─────────────────────────────────────────────────────────
    json_out = out_dir / "model_analysis.json"
    summary  = {
        "generated"   : datetime.datetime.now().isoformat(),
        "dataset"     : manifest["dataset"],
        "num_samples" : int(len(y_true_global)) if y_true_global is not None else 0,
        "num_classes" : num_dataset_classes,
        "classes"     : dataset_classes,
        "fp16"        : manifest.get("fp16", False),
        "models"      : [{k: v for k, v in m.items() if k not in ("cm_b64", "benchmark_b64")}
                         for m in model_results],
        "ensemble"    : {k: v for k, v in ensemble_results.items() if k != "forward"},
    }
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON saved: {json_out}")

    # ── HTML report ───────────────────────────────────────────────────────────
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_info = {
        "dataset"    : manifest["dataset"],
        "num_samples": int(len(y_true_global)) if y_true_global is not None else 0,
        "num_classes": num_dataset_classes,
        "num_models" : len(model_results),
        "fp16"       : manifest.get("fp16", False),
    }

    efficiency_b64 = ""
    if len(model_results) >= 1:
        eff_path = out_dir / "efficiency_comparison.png"
        _save_efficiency_plot(model_results, dataset_classes, str(eff_path))
        efficiency_b64 = _png_to_b64(str(eff_path))

    html     = _build_html(meta_info, model_results, ensemble_results,
                           dataset_classes, generated, efficiency_b64)
    html_out = out_dir / "index.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  HTML saved: {html_out}")
    print(f"\n{'='*60}")
    print(f"  Open: {html_out.resolve()}")
    print(f"{'='*60}\n")

    return str(out_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis_dir",
                    help="Directory created by ModelAnalysisEvaluate.py (contains raw_data/)")
    ap.add_argument("--metric", default=DEFAULT_METRIC,
                    choices=["balanced_accuracy", "accuracy"])
    args = ap.parse_args()

    cfg = ReportConfig(
        analysis_dir = args.analysis_dir,
        metric       = args.metric,
    )
    run_report(cfg)


if __name__ == "__main__":
    main()
