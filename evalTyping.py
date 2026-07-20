#!/usr/bin/python3
"""
evalTyping.py -- domain-separated per-TYPE evaluation of an alldefect typer.

The mixed FORTH+Altinay runs train on a frame-disjoint split of the combined
pool. This script rebuilds the trainer's EXACT held-out val split (same
reconstruction + same seed), runs the model on those val tiles only (no
leakage), and reports per-class recall SPLIT BY DOMAIN (Altinay vs FORTH) --
the number the cross-site campaign actually cares about.

To compare a severity model (12 fine classes) against a non-severity model
(6 base classes) on one axis, it also collapses both predictions and truth to
BASE type (class_XClassY -> class_X) and reports base-type recall per domain.

Usage:
    python3 evalTyping.py <model.pth> <sidecar_config.json>

    sidecar_config.json = the '<name>_custom.json' the trainer writes out (it
    embeds 'classes' and the fully-resolved strip/merge/drop settings).
"""
import json
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from DatasetConverter import HDF5Dataset
from trainMagicianVisionClassifierTorch import (
    strip_severity_classes, merge_dataset_classes, drop_dataset_classes,
    align_dataset_to_classes, CombinedDataset, frame_disjoint_split,
    _dataset_source_frames, filter_dataset_classes, metadata_collate_fn,
)
from calculateOptimalEnsemble import _instantiate_classifier, _load_weights


def _base(cn):
    """class_WeldingClassA -> class_Welding (base defect type)."""
    return re.sub(r'Class[A-Z]$', '', cn)


def _load_one(d, cfg):
    """Mirror the trainer's _load_one: same filter/strip/merge/drop order."""
    ds = HDF5Dataset(f"{d}/dataset.h5")
    ds.metadata = None
    if cfg.get('selected_classes') and len(cfg['selected_classes']) > 1:
        filter_dataset_classes(ds, cfg['selected_classes'])
    if cfg.get('strip_severity') or cfg.get('severity') is False:
        strip_severity_classes(ds)
    if cfg.get('class_merges'):
        merge_dataset_classes(ds, cfg['class_merges'])
    if cfg.get('drop_classes'):
        drop_dataset_classes(ds, cfg['drop_classes'])
    return ds


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    pth, cfgpath = sys.argv[1], sys.argv[2]
    cfg = json.load(open(cfgpath))

    dirs = cfg['training_dataset']
    dirs = dirs if isinstance(dirs, list) else [dirs]
    subs = [_load_one(d, cfg) for d in dirs]
    if len(subs) == 1:
        dataset = subs[0]
    else:
        canon = cfg.get('canonical_classes') or list(subs[0].classes)
        for ds in subs:
            align_dataset_to_classes(ds, canon)
        dataset = CombinedDataset(subs)
    classes = list(dataset.classes)

    seed = cfg['hparams']['seed']
    vsplit = cfg['dataloader']['validation_split']
    _, va = frame_disjoint_split(dataset, vsplit, seed)
    va = np.asarray(va)

    srcs = np.array(_dataset_source_frames(dataset))
    dom = np.array(['Altinay' if 'altinay' in s.lower() else 'FORTH' for s in srcs])

    clf = _instantiate_classifier(cfg, classes)
    clf = _load_weights(clf, pth, 'cuda').cuda().eval()

    loader = DataLoader(Subset(dataset, va.tolist()), batch_size=512,
                        num_workers=8, shuffle=False, collate_fn=metadata_collate_fn)
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].cuda()
            preds.append(clf(x).argmax(1).cpu().numpy())
            ys.append(np.asarray(batch[1]))
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    vdom = dom[va]

    print(f"\nModel: {pth}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Val tiles: {len(ys)}  (FORTH {int((vdom=='FORTH').sum())} / "
          f"Altinay {int((vdom=='Altinay').sum())})")

    # ---- per-class recall, per domain (native class space) ----
    for D in ('FORTH', 'Altinay'):
        dm = vdom == D
        print(f"\n[{D}] per-type recall (native {len(classes)}-class space):")
        recs = []
        for c, cn in enumerate(classes):
            m = dm & (ys == c)
            n = int(m.sum())
            if n == 0:
                continue
            r = 100.0 * (preds[m] == c).mean()
            recs.append(r)
            print(f"   {cn.replace('class_',''):<22} n={n:>6}  recall {r:5.1f}%")
        if recs:
            print(f"   {'MACRO':<22}          recall {np.mean(recs):5.1f}%")

    # ---- base-type recall, per domain (comparable across sev / non-sev) ----
    base_names = sorted({_base(c) for c in classes})
    base_of = np.array([base_names.index(_base(c)) for c in classes])
    ys_b = base_of[ys]
    preds_b = base_of[preds]
    print("\n=== BASE-TYPE recall (severity collapsed -- comparable across models) ===")
    for D in ('FORTH', 'Altinay'):
        dm = vdom == D
        print(f"[{D}]:")
        recs = []
        for c, cn in enumerate(base_names):
            m = dm & (ys_b == c)
            n = int(m.sum())
            if n == 0:
                continue
            r = 100.0 * (preds_b[m] == c).mean()
            recs.append(r)
            print(f"   {cn.replace('class_',''):<22} n={n:>6}  recall {r:5.1f}%")
        if recs:
            print(f"   {'MACRO':<22}          recall {np.mean(recs):5.1f}%")


if __name__ == "__main__":
    main()
