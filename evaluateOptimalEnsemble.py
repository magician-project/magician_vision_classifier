#!/usr/bin/python3
"""
evaluateOptimalEnsemble.py

Load the output of calculateOptimalEnsemble.py and search for the best
combination of classifiers to maximise accuracy while minimising inference time.

Ensemble prediction strategies evaluated:
  soft                 — average softmax probability vectors, then argmax
                         (best for well-calibrated models)
  majority             — each model votes for its top class; take the mode
                         (robust to one bad model)
  confidence_weighted  — weight each model's probability vector by its own
                         max confidence for that sample before averaging

Search methods:
  1. Single-model baseline ranking
  2. Greedy forward selection  — add the model that most improves the metric
  3. Greedy backward elimination — remove the model whose removal costs least
  4. Pareto front (accuracy vs ms/sample) extracted from forward selection

Timing assumptions:
  ms_sequential — sum of all included model latencies (worst case, CPU pipeline)
  ms_parallel   — max of all included model latencies (CUDA-stream parallel, as
                  used by EnsembleClassifier.py with run_models_async())

Usage:
    python3 evaluateOptimalEnsemble.py <ensemble_data.json> [--metric <name>]

    metric: balanced_accuracy (default) | accuracy

Output:
    <stem>_results.json — full search results and recommendations
"""

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGIES        = ["soft", "majority", "confidence_weighted"]
DEFAULT_METRIC    = "balanced_accuracy"
# Stop greedy forward selection when the gain per added model falls below this.
MIN_GAIN_THRESHOLD = 0.0005
# Stop greedy backward elimination when removing a model costs more than this.
MAX_DROP_THRESHOLD = 0.005


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(json_path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Load the metadata JSON and companion .npz produced by calculateOptimalEnsemble.py.

    Returns:
        meta   : full metadata dict (models list, classes, dataset info)
        y_true : (N,) int64 — ground-truth labels
        probs  : (M, N, K) float32 — M models · N samples · K classes
    """
    with open(json_path) as f:
        meta = json.load(f)

    npz_path = meta["npz_file"]
    if not Path(npz_path).is_file():
        # Try same directory as the JSON (data may have been moved)
        npz_path = str(Path(json_path).parent / Path(npz_path).name)

    arrays = np.load(npz_path)
    return meta, arrays["y_true"], arrays["probs"]


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------

def ensemble_predict(
    probs: np.ndarray,
    subset: list[int],
    strategy: str,
) -> np.ndarray:
    """
    Compute predictions for a *subset* of models using the given strategy.

    probs   : (M, N, K) — full probability array (all models)
    subset  : list of model indices to include
    strategy: 'soft' | 'majority' | 'confidence_weighted'

    Returns (N,) int64 predictions.
    """
    sub = probs[subset]   # (|S|, N, K)

    if strategy == "soft":
        # Average probability vectors, then argmax — best overall accuracy
        return sub.mean(axis=0).argmax(axis=1).astype(np.int64)

    elif strategy == "majority":
        # Each model votes for its top class; take the per-position mode.
        votes = sub.argmax(axis=2)   # (|S|, N)
        S, N = votes.shape
        K    = probs.shape[2]
        # Accumulate vote counts without a Python loop
        vote_counts = np.zeros((N, K), dtype=np.int32)
        for s_idx in range(S):
            np.add.at(vote_counts, (np.arange(N), votes[s_idx]), 1)
        return vote_counts.argmax(axis=1).astype(np.int64)

    elif strategy == "confidence_weighted":
        # Scale each model's prob vector by that model's own max confidence
        # for each sample, then sum and argmax — rewards decisive models.
        max_conf = sub.max(axis=2, keepdims=True)   # (|S|, N, 1)
        weighted = (sub * max_conf).sum(axis=0)      # (N, K)
        return weighted.argmax(axis=1).astype(np.int64)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall (balanced accuracy), computed without sklearn."""
    classes  = np.unique(y_true)
    recalls  = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def score_subset(
    probs        : np.ndarray,
    y_true       : np.ndarray,
    subset       : list[int],
    strategy     : str,
    ms_per_sample: np.ndarray,
) -> dict:
    """
    Evaluate one ensemble configuration.

    Returns a dict with accuracy, balanced_accuracy, ms_sequential,
    ms_parallel, subset, and strategy.
    """
    y_pred = ensemble_predict(probs, subset, strategy)
    acc    = float((y_pred == y_true).mean())
    bal    = _balanced_accuracy(y_true, y_pred)

    return {
        "accuracy"         : acc,
        "balanced_accuracy": bal,
        # Sequential: each model runs after the other (safe lower bound)
        "ms_sequential"    : float(ms_per_sample[subset].sum()),
        # Parallel: models run on separate CUDA streams simultaneously
        "ms_parallel"      : float(ms_per_sample[subset].max()),
        "subset"           : list(subset),
        "strategy"         : strategy,
    }


# ---------------------------------------------------------------------------
# Greedy search
# ---------------------------------------------------------------------------

def greedy_forward_selection(
    probs        : np.ndarray,
    y_true       : np.ndarray,
    ms_per_sample: np.ndarray,
    strategy     : str,
    metric       : str,
    max_models   : int | None = None,
) -> list[dict]:
    """
    Greedy forward selection: repeatedly add the model that most improves
    *metric* until no model improves it by more than MIN_GAIN_THRESHOLD or
    *max_models* is reached.

    Returns a list of step dicts (one per added model), each containing the
    cumulative subset and scores after that model was added.
    """
    M         = probs.shape[0]
    remaining = list(range(M))
    selected  : list[int] = []
    steps     : list[dict] = []
    best_score = -1.0

    limit = max_models if max_models is not None else M

    for _ in range(limit):
        if not remaining:
            break

        best_candidate = None
        best_result    = None

        for candidate in remaining:
            trial  = selected + [candidate]
            result = score_subset(probs, y_true, trial, strategy, ms_per_sample)
            if best_candidate is None or result[metric] > (best_result[metric] if best_result else -1):
                best_candidate = candidate
                best_result    = result

        if best_candidate is None:
            break

        gain = best_result[metric] - best_score
        if selected and gain < MIN_GAIN_THRESHOLD:
            # Marginal gain too small — stop early
            break

        best_score = best_result[metric]
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        steps.append({**best_result, "added_model": best_candidate})

    return steps


def greedy_backward_elimination(
    probs        : np.ndarray,
    y_true       : np.ndarray,
    ms_per_sample: np.ndarray,
    strategy     : str,
    metric       : str,
) -> list[dict]:
    """
    Greedy backward elimination: start with ALL models and iteratively remove
    the one whose removal costs the least accuracy (or even improves it), until
    any further removal would cost more than MAX_DROP_THRESHOLD.

    Returns a list of step dicts (one per removed model).
    """
    selected = list(range(probs.shape[0]))
    steps    : list[dict] = []

    while len(selected) > 1:
        baseline = score_subset(probs, y_true, selected, strategy, ms_per_sample)

        best_to_remove = None
        best_after     = None
        best_drop      = float("inf")

        for candidate in selected:
            trial = [m for m in selected if m != candidate]
            after = score_subset(probs, y_true, trial, strategy, ms_per_sample)
            drop  = baseline[metric] - after[metric]
            if drop < best_drop:
                best_drop      = drop
                best_to_remove = candidate
                best_after     = after

        if best_drop > MAX_DROP_THRESHOLD:
            # Removing any remaining model hurts accuracy too much — stop.
            break

        selected.remove(best_to_remove)
        steps.append({**best_after, "removed_model": best_to_remove})

    return steps


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def pareto_front(
    results      : list[dict],
    metric       : str,
    time_key     : str = "ms_sequential",
) -> list[dict]:
    """
    Extract the Pareto-optimal configurations from *results*.

    A result is Pareto-optimal if no other result achieves both strictly
    higher *metric* AND strictly lower *time_key*.
    Results are sorted by ascending time.
    """
    front = []
    for r in results:
        dominated = any(
            o is not r
            and o[metric]   >= r[metric]
            and o[time_key] <= r[time_key]
            and (o[metric] > r[metric] or o[time_key] < r[time_key])
            for o in results
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda x: x[time_key])


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _model_names_for(subset: list[int], model_names: list[str]) -> list[str]:
    return [model_names[i] for i in subset]


def _print_header(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def _print_step_table(
    steps      : list[dict],
    model_names: list[str],
    key        : str,
    label      : str,
):
    print(f"  {'Step':<5} {label:<40} {'Acc':>7} {'BalAcc':>8} {'ms-seq':>8} {'ms-par':>8}")
    print(f"  {'─'*5} {'─'*40} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
    for i, s in enumerate(steps):
        name = model_names[s[key]]
        print(
            f"  {i + 1:<5} {name:<40} {s['accuracy']:>7.4f} "
            f"{s['balanced_accuracy']:>8.4f} "
            f"{s['ms_sequential']:>8.3f} {s['ms_parallel']:>8.3f}"
        )


def _print_pareto(front: list[dict], model_names: list[str]):
    print(f"  {'#M':<4} {'BalAcc':>8} {'Acc':>7} {'ms-seq':>8} {'ms-par':>8}  Models")
    print(f"  {'─'*4} {'─'*8} {'─'*7} {'─'*8} {'─'*8}  {'─'*40}")
    for p in front:
        names = ", ".join(_model_names_for(p["subset"], model_names))
        print(
            f"  {len(p['subset']):<4} {p['balanced_accuracy']:>8.4f} "
            f"{p['accuracy']:>7.4f} {p['ms_sequential']:>8.3f} "
            f"{p['ms_parallel']:>8.3f}  {names}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_path = sys.argv[1]
    metric    = DEFAULT_METRIC
    if "--metric" in sys.argv:
        idx    = sys.argv.index("--metric")
        metric = sys.argv[idx + 1]

    meta, y_true, probs = load_data(json_path)

    M, N, K       = probs.shape
    model_names   = [m["name"]            for m in meta["models"]]
    ms_per_sample = np.array([m["ms_per_sample"] for m in meta["models"]], dtype=np.float64)
    classes       = meta["classes"]

    print(f"\n{'='*70}")
    print(f"  Ensemble Optimizer")
    print(f"{'='*70}")
    print(f"  Dataset   : {meta['dataset']}")
    print(f"  Samples   : {N}")
    print(f"  Classes   : {classes}")
    print(f"  Models    : {M}")
    print(f"  Metric    : {metric}")

    # ── Single-model baseline ───────────────────────────────────────────────
    _print_header("Single-model baseline")
    print(f"  {'Name':<40} {'Acc':>7} {'BalAcc':>8} {'ms-seq':>8} {'Params':>12}")
    print(f"  {'─'*40} {'─'*7} {'─'*8} {'─'*8} {'─'*12}")

    single_results = []
    for i, m in enumerate(meta["models"]):
        r = score_subset(probs, y_true, [i], "soft", ms_per_sample)
        r["name"]       = m["name"]
        r["num_params"] = m.get("num_params", 0)
        single_results.append(r)
        print(
            f"  {m['name']:<40} {r['accuracy']:>7.4f} {r['balanced_accuracy']:>8.4f} "
            f"{r['ms_sequential']:>8.3f} {m.get('num_params', 0):>12,}"
        )

    single_results.sort(key=lambda x: -x[metric])

    # ── Greedy forward selection (all strategies) ───────────────────────────
    all_forward : dict[str, list[dict]] = {}
    all_pareto  : dict[str, list[dict]] = {}

    for strategy in STRATEGIES:
        _print_header(f"Greedy Forward Selection — {strategy}")
        steps = greedy_forward_selection(
            probs, y_true, ms_per_sample, strategy, metric
        )
        all_forward[strategy] = steps

        _print_step_table(steps, model_names, key="added_model", label="Added model")

        # Pareto front from the intermediate-step scores
        front = pareto_front(steps, metric)
        all_pareto[strategy] = front

        print(f"\n  Pareto front ({strategy}):")
        _print_pareto(front, model_names)

    # ── Greedy backward elimination (soft voting only — fastest search) ─────
    _print_header("Greedy Backward Elimination — soft voting")
    backward_steps = greedy_backward_elimination(
        probs, y_true, ms_per_sample, "soft", metric
    )
    if backward_steps:
        _print_step_table(backward_steps, model_names, key="removed_model", label="Removed model")
        final_subset = backward_steps[-1]["subset"]
        print(f"\n  Surviving models ({len(final_subset)}):")
        for i in final_subset:
            print(f"    - {model_names[i]}")
    else:
        print("  No models could be removed without exceeding the accuracy drop threshold.")

    # ── Recommendation ──────────────────────────────────────────────────────
    # Pick the configuration with highest metric across all strategies;
    # break ties by choosing the faster one (lower ms_sequential).
    candidates = []
    for strategy, steps in all_forward.items():
        for s in steps:
            candidates.append(s)

    best = max(candidates, key=lambda x: (x[metric], -x["ms_sequential"]))
    best_strategy = best["strategy"]
    best_names    = _model_names_for(best["subset"], model_names)

    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")
    print(f"  Strategy        : {best_strategy}")
    print(f"  Accuracy        : {best['accuracy']:.4f}")
    print(f"  Balanced Acc    : {best['balanced_accuracy']:.4f}")
    print(f"  ms/sample (seq) : {best['ms_sequential']:.3f}")
    print(f"  ms/sample (par) : {best['ms_parallel']:.3f}")
    print(f"  Models ({len(best_names)}):")
    for name in best_names:
        print(f"    - {name}")

    # ── Save JSON results ───────────────────────────────────────────────────
    out_path = str(Path(json_path).stem) + "_results.json"

    def _annotate(step_list: list[dict], key: str) -> list[dict]:
        """Add human-readable model names to a list of step dicts."""
        out = []
        for s in step_list:
            entry = dict(s)
            entry["model_names"] = _model_names_for(s["subset"], model_names)
            if key in s:
                entry[f"{key}_name"] = model_names[s[key]]
            out.append(entry)
        return out

    output = {
        "source"      : json_path,
        "metric"      : metric,
        "classes"     : classes,
        "model_names" : model_names,
        "ms_per_sample": ms_per_sample.tolist(),
        "single_model_baseline": single_results,
        "forward_selection": {
            strategy: _annotate(steps, "added_model")
            for strategy, steps in all_forward.items()
        },
        "pareto_fronts": {
            strategy: _annotate(front, "added_model")
            for strategy, front in all_pareto.items()
        },
        "backward_elimination": _annotate(backward_steps, "removed_model"),
        "recommendation": {
            "strategy"         : best_strategy,
            "subset"           : best["subset"],
            "model_names"      : best_names,
            "accuracy"         : best["accuracy"],
            "balanced_accuracy": best["balanced_accuracy"],
            "ms_sequential"    : best["ms_sequential"],
            "ms_parallel"      : best["ms_parallel"],
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
