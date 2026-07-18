#!/usr/bin/python3
"""
swaCheckpoints.py

Stochastic Weight Averaging (SWA) over the per-epoch checkpoints of a training
run: average the WEIGHTS of every checkpoint into one model, write it out as a
single .pth. One model, 1x inference, and it beats picking any single epoch.

Why this exists (measured 2026-07-17/18 on the cross-site customwide run):
  - The checkpoint that minimises val_loss is NOT the best cross-site detector --
    val_loss is uncorrelated with defect-vs-clean AUROC (pearson ~ -0.09). So
    "keep the best val_loss epoch" (ModelCheckpoint save_top_k=1) selects an
    essentially random epoch w.r.t. the KPI.
  - The cross-site metric oscillates epoch-to-epoch by +/-0.02..0.04 AUROC -- the
    same size as the gaps BETWEEN models -- so any single checkpoint is a lottery.
  - Averaging the weights of ALL epochs removes the lottery: it finds a flatter,
    more central solution in weight space. On customwide this lifted held-out
    AUROC 0.79 (best single epoch) -> 0.83, and PositiveDent detection to its best
    of the whole campaign, at 1x inference.

Two rules that came out of the measurements:
  1. Average ALL epochs, not just the strong late ones -- the "weak" early epochs
     contribute weight-space diversity, and averaging the tail alone is WORSE
     (all-18 AUROC 0.829 vs last-8 0.782).
  2. Valid only WITHOUT an LR scheduler (constant-LR AdamW here) so the
     checkpoints share one loss basin. A scheduler / warm restarts would break it.

BatchNorm/InstanceNorm running stats are averaged naively along with the weights
(integer counters like num_batches_tracked are taken from the first checkpoint,
not averaged). For CustomCNN (InstanceNorm, affine only) this is exact; for a
BatchNorm backbone the textbook step is to recompute BN stats with one pass over
the training data (--recompute-bn is a TODO hook, not yet implemented) -- without
it the averaged BN stats slightly UNDER-state performance.

Usage:
    python3 swaCheckpoints.py <ckpt_dir> <config.json> <out.pth> [--last N]

    ckpt_dir    directory of *.ckpt written by the trainer (ModelCheckpoint with
                checkpoint_save_top_k=-1 -- i.e. every epoch kept)
    config.json the run's config, or any config with the same hparams/model +
                a "classes" list (needed to rebuild the architecture)
    out.pth     where to write the averaged state_dict
    --last N    average only the last N checkpoints (default: all). All is
                recommended; this is here for ablation only.

Example:
    python3 swaCheckpoints.py \
        /media/ammar/games2/Datasets/Models/customwide_epochs_ckpts \
        configs/crossval_v2_rot_customwide.json  allclass_customwide_swa.pth
"""

import glob
import os
import re
import sys
import json

import torch

from evaluateClassifierNew import get_state_dict_from_checkpoint
from calculateOptimalEnsemble import _instantiate_classifier


def _epoch_of(path):
    m = re.search(r"epoch=(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def average_checkpoints(ckpt_paths, device="cpu"):
    """Average the weights of the given checkpoints into one state_dict.

    Floating-point tensors are averaged; non-float tensors (integer counters such
    as num_batches_tracked) are copied from the first checkpoint unchanged.
    """
    sds = [get_state_dict_from_checkpoint(p, device) for p in ckpt_paths]
    keys = sds[0].keys()
    avg = {}
    for k in keys:
        v0 = sds[0][k]
        if torch.is_floating_point(v0):
            avg[k] = torch.stack([sd[k].float() for sd in sds]).mean(0).to(v0.dtype)
        else:
            avg[k] = v0
    return avg


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    ckpt_dir = sys.argv[1]
    config_path = sys.argv[2]
    out_path = sys.argv[3]
    last_n = None
    if "--last" in sys.argv:
        last_n = int(sys.argv[sys.argv.index("--last") + 1])

    cks = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=_epoch_of)
    if not cks:
        print(f"[ERROR] no *.ckpt found in {ckpt_dir}")
        sys.exit(1)
    if last_n is not None:
        cks = cks[-last_n:]
    print(f"Averaging {len(cks)} checkpoints (epochs {[_epoch_of(p) for p in cks]})")

    config_json = json.load(open(config_path))
    classes = config_json.get("classes")
    if not classes:
        print("[ERROR] config has no 'classes' list -- needed to rebuild the model. "
              "Use the config the trainer wrote out (it embeds 'classes'), or add one.")
        sys.exit(1)

    avg = average_checkpoints(cks, device="cpu")
    model = _instantiate_classifier(config_json, classes)
    missing, unexpected = model.load_state_dict(avg, strict=False)
    if missing:
        print(f"  WARNING missing keys: {len(missing)} (first: {missing[0]})")
    if unexpected:
        print(f"  WARNING unexpected keys: {len(unexpected)} (first: {unexpected[0]})")

    # Save a plain state_dict .pth (matches what the loaders expect via
    # get_state_dict_from_checkpoint, which unwraps 'state_dict' if present).
    torch.save({"state_dict": model.state_dict()}, out_path)
    print(f"Wrote SWA model -> {out_path}  ({len(classes)} classes)")
    print("Reminder: pair it with the run's .json sidecar (same hparams/model + classes) "
          "for calculateOptimalEnsemble / the live path to rebuild it.")


if __name__ == "__main__":
    main()
