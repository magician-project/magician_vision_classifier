#!/usr/bin/python3
"""
evaluateDetection.py

Evaluate a classifier as a DEFECT DETECTOR -- the quantity the MAGICIAN KPI
(skipped defects) actually cares about -- instead of as a k-way classifier.

What it reports, and why each choice matters (all established 2026-07-17/18):

  * Detector score = 1 - P(clean), NOT max-prob. Max-prob discards a tile whose
    probability is split across defect classes (0.40 Welding / 0.40 Seal / 0.20
    clean scores 0.40 and is called clean, though it is 80% likely a defect).
    Summing the defect mass fixes this and lowers miss at every false-alarm rate.

  * Defect-vs-clean AUROC -- threshold-free, so it ranks models WITHOUT the
    operating-point bias you get from comparing each model at its own argmax.
    (val_loss is uncorrelated with this; don't rank models by val_loss.)

  * miss vs FALSE-POSITIVE budget, reported at TILE level and FRAME level. A
    physical defect spans many tiles/frames, so tile miss over-states the
    defect-level miss; the frame column (a frame = proxy for a defect instance,
    caught if any of its true-defect tiles fires) is closer to the KPI. NOTE:
    this still under-counts the deployment's advantage -- live sliding-window
    inference at step S gives many overlapping looks per defect, and the same
    defect recurs across frames of a continuous scan; the val H5 is ~one tile per
    defect, so neither multiplier is captured here.

  * Per-class detection at a chosen FP -- computed against the ORIGINAL class
    labels even for a binary-trained model, so you still see whether PositiveDent
    etc. survive. (Pass --no-merge-eval, the default, to keep original labels.)

  * --split-frames: pick the operating threshold on half the FRAMES, report on
    the other half. Selecting the threshold on the same tiles you report on
    overfits; frame-disjoint is the honest protocol (tiles of one frame would
    otherwise leak across the split).

Usage:
    python3 evaluateDetection.py <model.pth> <config.json> [dataset_dir] [--split-frames] [--fp P]

    model.pth     a trained model (or an SWA model from swaCheckpoints.py)
    config.json   its sidecar (hparams/model + a 'classes' list)
    dataset_dir   default: the config's validation_dataset
    --fp P        false-positive budget for the per-class table (default 27,
                  the historically-deployed tile FP; use e.g. 5 for a realistic
                  operating point)
    --split-frames  select threshold on frame-half A, report on held-out half B

Example:
    python3 evaluateDetection.py allclass_customwide_swa.pth \
        configs/crossval_v2_rot_customwide.json --split-frames --fp 27
"""

import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from calculateOptimalEnsemble import _instantiate_classifier, _load_weights, _load_dataset
from trainMagicianVisionClassifierTorch import metadata_collate_fn


def frame_index(dataset):
    """One integer per tile identifying its source FRAME (path minus the tile
    (x,y) offset), parsed from the per-sample metadata 'source' field. Returns
    None if the dataset carries no metadata."""
    md = getattr(dataset, "metadata", None)
    if md is None and hasattr(dataset, "h5") is False:
        # HDF5Dataset stores metadata lazily; try the raw dataset attribute
        pass
    try:
        raw = dataset.get_all_metadata() if hasattr(dataset, "get_all_metadata") else None
    except Exception:
        raw = None
    if raw is None:
        # fall back to iterating -- only if the dataset returns metadata tuples
        return None
    frames = []
    for r in raw:
        r = r.decode() if isinstance(r, bytes) else r
        src = json.loads(r)["source"]
        frames.append(src.rsplit("(", 1)[0])
    _, inv = np.unique(np.array(frames), return_inverse=True)
    return inv


def _frame_index_from_h5(dataset_dir):
    """Robust frame index straight from the H5 metadata dataset."""
    import h5py
    h5 = os.path.join(dataset_dir, "dataset.h5")
    if not os.path.isfile(h5):
        return None
    with h5py.File(h5, "r") as f:
        if "metadata" not in f:
            return None
        md = f["metadata"][:]
    frames = []
    for r in md:
        r = r.decode() if isinstance(r, bytes) else r
        frames.append(json.loads(r)["source"].rsplit("(", 1)[0])
    _, inv = np.unique(np.array(frames), return_inverse=True)
    return inv


def score_model(pth, config_json, classes, dataset, device="cuda"):
    """Return 1 - P(clean) per tile, and the clean-class index used."""
    clean_m = classes.index("class_clean")
    clf = _instantiate_classifier(config_json, classes)
    clf = _load_weights(clf, pth, device).to(device).eval()
    loader = DataLoader(dataset, batch_size=384, shuffle=False, num_workers=8,
                        collate_fn=metadata_collate_fn)
    out = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            p = torch.softmax(clf(x), dim=1)
            out.append((1.0 - p[:, clean_m]).float().cpu().numpy())
    return np.concatenate(out)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    pth = sys.argv[1]
    config_json = json.load(open(sys.argv[2]))
    split = "--split-frames" in sys.argv
    fp_budget = float(sys.argv[sys.argv.index("--fp") + 1]) if "--fp" in sys.argv else 27.0
    # positional dataset_dir = first arg after argv[2] that is neither a flag nor a
    # flag's value (only --fp takes a value).
    consumed = set()
    for i, a in enumerate(sys.argv):
        if a == "--fp":
            consumed.add(i); consumed.add(i + 1)
        elif a.startswith("-"):
            consumed.add(i)
    dataset_dir = next((a for i, a in enumerate(sys.argv[3:], start=3)
                        if i not in consumed),
                       config_json.get("validation_dataset"))

    # Evaluate against the ORIGINAL class labels (no merge), so per-class
    # detection is recoverable even for a binary-trained model.
    ds = _load_dataset(dataset_dir)
    ds.metadata = None  # 3-tuple batches crash the collate; labels still returned
    orig_classes = list(ds.classes)
    CLEAN = orig_classes.index("class_clean")

    # The MODEL's own class list (may be binary / merged) -- used to rebuild it.
    model_classes = config_json.get("classes") or orig_classes

    y = np.concatenate([np.asarray(b[1]) for b in
                        DataLoader(ds, batch_size=8192, num_workers=8,
                                   collate_fn=metadata_collate_fn)])
    isdef = y != CLEAN
    s = score_model(pth, config_json, model_classes, ds)

    fidx = _frame_index_from_h5(dataset_dir)

    # frame-disjoint split (select on A, report on B) or evaluate on everything
    if split and fidx is not None:
        rng = np.random.default_rng(42)
        NF = fidx.max() + 1
        half = set(rng.permutation(NF)[:NF // 2].tolist())
        inA = np.array([f in half for f in range(NF)])[fidx]
        sel_mask, rep_mask = inA, ~inA
        print(f"Frame-disjoint: threshold picked on A ({inA.sum()} tiles), "
              f"reported on B ({(~inA).sum()} tiles)")
    else:
        sel_mask = rep_mask = np.ones(len(y), bool)
        if split:
            print("WARNING: --split-frames requested but no frame metadata; "
                  "evaluating on the full set.")

    def thr_for_fp(mask, budget):
        cm = (~isdef) & mask
        return np.quantile(s[cm], 1 - budget / 100.0)

    auroc = roc_auc_score(isdef[rep_mask].astype(int), s[rep_mask])
    print(f"\nDefect-vs-clean AUROC (report set): {auroc:.4f}")

    print(f"\n{'FP budget':<11}{'TILE miss':>11}{'FRAME miss':>13}")
    for fp in (1, 2, 5, 10, 20, fp_budget):
        t = thr_for_fp(sel_mask, fp)
        fire = s >= t
        tm = 100 * (~fire[isdef & rep_mask]).mean()
        if fidx is not None:
            caught = np.bincount(fidx, weights=(fire & isdef).astype(float)) > 0
            anyf = np.bincount(fidx, weights=fire.astype(float)) > 0
            defframe = np.bincount(fidx, weights=isdef.astype(float)) > 0
            repframe = np.zeros(len(defframe), bool)
            repframe[fidx[rep_mask]] = True
            dm = defframe & repframe
            cm = (~defframe) & repframe
            fm = 100 * (~caught[dm]).mean()
        else:
            fm = float("nan")
        tag = f"FP<={fp:g}%"
        print(f"{tag:<11}{tm:>10.1f}%{fm:>12.1f}%")

    # per-class detection at the chosen FP, against ORIGINAL labels
    t = thr_for_fp(sel_mask, fp_budget)
    fire = s >= t
    print(f"\nPer-class detection @ FP={fp_budget:g}% (original labels):")
    for c, cn in enumerate(orig_classes):
        if c == CLEAN:
            continue
        m = (y == c) & rep_mask
        if m.sum() == 0:
            continue
        print(f"  {cn.replace('class_',''):<16} n={int(m.sum()):>6}   detected {100*fire[m].mean():5.1f}%")


if __name__ == "__main__":
    main()
