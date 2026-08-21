"""Post-training evaluation: confusion matrix + detection threshold sweep.

This used to live inline in the trainer's __main__, which meant (a) the CPU-inference
bug had to be fixed in two copies, and (b) re-scoring an already-trained checkpoint was
impossible without repeating the whole training run. Keeping it here makes the sweep
callable on any (classifier, val_loader) pair.

Writes, all keyed on `model_name`:
    {name}_confusion.json         + plots
    {name}_threshold_curve.json   + plot     <- the miss@FA KPI comes from here
and folds the tuned thresholds / confusion matrix back into `config_json`.

The detection score is defect_mass = 1 - P(clean); miss@FA is read off the curve by
interpolating `detected` at a target `false_alarm`.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch
from mvc.core.artifact_paths import out_path   # writers emit into experiments/<campaign>/<run>/
from mvc.paths import repo_root


class _SkipSweep(Exception):
    """Raised when there is no clean class, so defect_mass is undefined."""
    pass


def _write_plots(json_path):
    """Render the plots for an artifact JSON, and SAY SO if it cannot.

    This used to be `subprocess.run([sys.executable, "plotTool.py", ...], check=False)`.
    The layout move turned `plotTool.py` into `analysis/plots/plot_tool.py`, so the call
    found nothing -- and `check=False` meant every run since then wrote its confusion and
    threshold JSONs and silently produced no PNGs at all. Nothing failed, nothing logged,
    the plots simply stopped existing.

    Invoked as `-m` from the repo root so it does not depend on the cwd, and a non-zero
    exit is reported with the `!!!` prefix the sweep drivers already grep for. It is a
    warning rather than a raise on purpose: the JSON is the artifact of record and the
    plots are derived from it, so losing a plot must be loud but must not throw away a
    finished evaluation.
    """
    r = subprocess.run([sys.executable, "-m", "analysis.plots.plot_tool", json_path],
                       cwd=repo_root(), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"!!! plotting failed for {json_path} (rc={r.returncode}); the JSON is "
              f"written, the PNGs are not", file=sys.stderr)
        if r.stderr:
            print(r.stderr.strip()[-500:], file=sys.stderr)


def run_confusion_and_threshold_sweep(classifier, trainer, val_loader, dataset,
                                      config_json, model_name, cleanClassID,
                                      tile_size=None, epochs=None):
    """Validate, build the confusion matrix, sweep detection thresholds and write
    every artifact. Mutates `config_json` in place with the results.

    Runs the sweep on the accelerator: trainer.validate() tears the module down onto
    the CPU, and inferring every val tile on one CPU core was ~100x slower (3.7 h vs
    94 s on a 669k-tile split).
    """
    # Only used in plot titles; fall back to the config so callers re-scoring a saved
    # checkpoint do not have to thread them through.
    if tile_size is None:
        tile_size = config_json['hparams']['tile_size']
    if epochs is None:
        epochs = config_json['hparams'].get('training_epochs')

    #Predictions
    #------------------------------------------------------------------
    print("Final model validation")
    classifier.eval()
    trainer.validate(classifier, val_loader)
    #------------------------------------------------------------------

    try:
        print("Removing previous confusion matrix data..")
        for stale in glob.glob(out_path(model_name, "_confusion.json")) + glob.glob(out_path(model_name) + "*.png"):
            try:
                os.remove(stale)
            except OSError:
                pass

        print("Generating new confusion matrix data..")
        # trainer.validate() tears the module down onto the CPU, so classifier.device
        # is 'cpu' by the time we get here and the loop below would run every val tile
        # through the net on ONE CPU core -- ~100x slower than the GPU and the real
        # reason this stage used to take hours (it was never disk-bound). Put the model
        # back on the accelerator for the sweep.
        eval_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier.to(eval_device)
        print(f"Confusion/threshold sweep running on: {eval_device}")
        y_true = []
        y_pred = []
        y_maxp = []   # max softmax probability      -> legacy max_prob gate sweep
        y_pcln = []   # P(clean) per tile            -> defect_mass gate sweep

        # torch.no_grad() prevents graph construction for every batch (saves memory).
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(classifier.device)
                probs = torch.softmax(classifier(x), dim=1)
                mp, pr = probs.max(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pr.cpu().numpy())
                y_maxp.extend(mp.cpu().numpy())
                if cleanClassID is not None:
                    y_pcln.extend(probs[:, cleanClassID].cpu().numpy())

        num_classes = len(dataset.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            confusion_matrix[true, pred] += 1
        print(confusion_matrix)

        # Embed confusion matrix in the main config JSON
        config_json["confusion_matrix"] = confusion_matrix.tolist()
        # sorted(): a set has no defined iteration order, so this key used to come
        # out in an arbitrary order and was meaningless to any reader.
        config_json["classes_int"] = sorted(int(c) for c in set(y_true))
        config_json["classes"] = dataset.classes

        # Write a separate confusion JSON and generate the plot image
        print("Generating confusion matrix plot")
        confusion_json = {
            "title":  f"{model_name} / Tile Size = {tile_size} / Epochs = {epochs}",
            "labels": dataset.classes,
            "matrix": confusion_matrix.tolist(),
        }
        with open(out_path(model_name, "_confusion.json"), "w") as f:
            json.dump(confusion_json, f, indent=2)
        _write_plots(out_path(model_name, "_confusion.json"))

        # Threshold sweep -> operating curve + the gate the live path will use.
        # Swept for BOTH gates (liveClassifierTorch.gate_tiles implements them):
        #   "defect_mass" : score = 1 - P(clean), the total mass on ANY defect
        #                   class. Recommended. Flags a tile the model is sure is
        #                   a defect even when it cannot say which one.
        #   "max_prob"    : score = max_c P(c), and the tile must not argmax to
        #                   clean. Legacy, kept so older runs stay comparable. It
        #                   discards a 0.40 Welding / 0.40 Seal / 0.20 clean tile
        #                   as clean despite it being 80% likely a defect.
        # Thresholds are NOT comparable between the two modes or across models,
        # so we write the chosen one into config_json["gate"] and the live path
        # reads it from there instead of hardcoding a number.
        # The sweep is a defect-vs-clean operating curve -- meaningless without a
        # clean class (e.g. an 'alldefect' typer), so skip it there.
        if cleanClassID is None:
            print("No clean class -> skipping threshold sweep / gate calibration")
            raise _SkipSweep()
        print("Generating threshold sweep / operating curve")
        yt = np.array(y_true); yp = np.array(y_pred)
        mp = np.array(y_maxp); pcl = np.array(y_pcln)
        isdef = yt != cleanClassID
        n_def = max(1, int(isdef.sum())); n_cln = max(1, int((~isdef).sum()))

        def run_sweep(score, also_require=None):
            out = []
            for t in np.arange(0.0, 0.996, 0.005):
                flagged = score >= t
                if also_require is not None:
                    flagged = flagged & also_require
                out.append({
                    "threshold":   round(float(t), 3),
                    "detected":    float((flagged &  isdef).sum() / n_def),
                    "false_alarm": float((flagged & ~isdef).sum() / n_cln),
                })
            return out

        def pick(sweep):
            balance  = [s["detected"] - s["false_alarm"] for s in sweep]        # Youden J
            kpi_cost = [2*(1-s["detected"]) + s["false_alarm"] for s in sweep]  # misses weigh 2x
            # Deployment pick: tile rates ignore prevalence (a frame is ~99% clean,
            # so 1% FA = dozens of false crosses). Highest detection subject to a
            # false-alarm budget of 0.5% of clean tiles (~30 crosses/frame @ step 14).
            in_budget = [s for s in sweep if s["false_alarm"] <= 0.005]
            return (sweep[int(np.argmax(balance))],
                    sweep[int(np.argmin(kpi_cost))],
                    max(in_budget, key=lambda s: s["detected"]) if in_budget else sweep[-1])

        sweeps = {
            "max_prob":    run_sweep(mp, also_require=(yp != cleanClassID)),
            "defect_mass": run_sweep(1.0 - pcl),
        }
        picks = {k: pick(v) for k, v in sweeps.items()}
        for mode, (b, k, d) in picks.items():
            print(f"[{mode}] balanced t={b['threshold']:.3f} "
                  f"(detected {b['detected']:.1%}, FA {b['false_alarm']:.1%}) | "
                  f"KPI t={k['threshold']:.3f} (detected {k['detected']:.1%}, FA {k['false_alarm']:.1%}) | "
                  f"deploy t={d['threshold']:.3f} (detected {d['detected']:.1%}, FA {d['false_alarm']:.2%})")

        # Which gate ships. Override per run with "gate_mode" in the config.
        gate_mode = config_json.get("gate_mode", "defect_mass")
        best_bal, best_kpi, best_dep = picks[gate_mode]
        # Legacy keys keep their original max_prob meaning so old configs/readers
        # do not silently change semantics.
        lb, lk, ld = picks["max_prob"]
        config_json["best_threshold_balanced"]   = lb
        config_json["best_threshold_kpi"]        = lk
        config_json["best_threshold_deployment"] = ld
        config_json["gate"] = {
            "mode": gate_mode,
            "threshold": best_kpi["threshold"],      # KPI-optimal: misses weigh 2x
            "assign_best_defect_class": True,
            "calibrated_on": config_json.get("validation_dataset", ""),
            "detected": best_kpi["detected"],
            "false_alarm": best_kpi["false_alarm"],
            "alternatives": {"balanced": best_bal, "deployment": best_dep},
        }
        print(f"config['gate'] -> mode={gate_mode} threshold={best_kpi['threshold']:.3f} "
              f"(detected {best_kpi['detected']:.1%}, FA {best_kpi['false_alarm']:.1%})")
        curve_json = {
            "title": f"{model_name} / Tile Size = {tile_size} / Epochs = {epochs}",
            "sweep": sweeps[gate_mode],
            "sweeps": sweeps,
            "best_balanced": best_bal,
            "best_kpi": best_kpi,
        }
        with open(out_path(model_name, "_threshold_curve.json"), "w") as f:
            json.dump(curve_json, f, indent=2)
        _write_plots(out_path(model_name, "_threshold_curve.json"))
    except _SkipSweep:
        pass   # no clean class; confusion matrix already written above
    except Exception as e:
        # Do NOT fail silently. This block produces the confusion matrix AND the
        # gate calibration; if it dies, config_json["gate"] is never set and the
        # model ships with no operating point at all. ClassifierPnm then falls back
        # to threshold 0.0 -- the gate is simply OFF -- which is indistinguishable
        # from a deliberate configuration choice. Previously the only trace was the
        # literal string "Failed" written into the confusion JSON.
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] confusion matrix / threshold sweep FAILED: {e!r}")
        print(f"[ERROR] {model_name}.json will ship WITHOUT a calibrated gate -- "
              f"the runtime will run with the gate OFF unless you set one by hand.")
        # Record the failure IN the shipped config so it is visible downstream
        # instead of looking like an intentional absence.
        config_json["gate_error"] = repr(e)
        config_json["gate_traceback"] = traceback.format_exc()
        with open(out_path(model_name, "_confusion.json"), "w") as f:
            json.dump({"error": repr(e), "traceback": traceback.format_exc()}, f, indent=2)


    #Compute MD5 of saved model for corruption detection
    #------------------------------------------------------------------
