#!/usr/bin/python3
"""
eval_domain_split.py  (mix_* campaign, 2026-07-26)

Second deliverable of the in-distribution backbone sweep: does the winner win on
the TARGET site (Altinay) specifically, or only on the FORTH-heavy aggregate?

Reproduces the EXACT held-out validation set each mix_* model was validated on
(CombinedDataset[train_nonaltinay_v2 + val_altinay_granular], frame_disjoint 0.1,
seed 42, 13-class granular, class_merges Seal->Welding, NO drops), then partitions
the held-out tiles by source domain (FORTH = non-Altinay vs Altinay) using the
CombinedDataset constituent offsets. For each backbone it reports, PER DOMAIN:
  * defect-vs-clean AUROC (threshold-free)
  * tile miss @ matched false-alarm (FA5 / FA10), threshold picked per-domain.

Score = 1 - P(clean) (defect_mass), identical to evaluateDetection.py / the trainer.

Usage:
    python3 eval_domain_split.py                       # the mix_* campaign above
    python3 eval_domain_split.py a.pth b.pth [...]     # arbitrary checkpoints

With explicit checkpoints the dataset dirs, class scheme and split parameters are
read from each model's own sidecar json (training_dataset / class_merges / drops /
classes), so a model trained on a different pool or class scheme is evaluated on the
split IT was validated on. Domains are the CombinedDataset constituents, named after
their directories, so a 3-dir pool reports 3 columns instead of FORTH/Altinay.
The no-argument path is unchanged and still reproduces the 13-class mix_* numbers.
"""
import json, os, sys, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from DatasetConverter import HDF5Dataset
from trainMagicianVisionClassifierTorch import (
    merge_dataset_classes, drop_dataset_classes, resolve_auto_drops,
    align_dataset_to_classes, CombinedDataset,
    frame_disjoint_split, metadata_collate_fn)
from calculateOptimalEnsemble import _instantiate_classifier, _load_weights

DIRS = ["/home/ammar/Documents/Programming/magician_datasets/train_nonaltinay_v2",
        "/home/ammar/Documents/Programming/magician_datasets/val_altinay_granular"]
BACKBONES = ["convnext_tiny", "efficientnet_b0", "regnet_y_800mf", "resnet18",
             "mobilenet_v3_large", "shufflenet_v2_x1_0", "custom"]
SEED, VAL_SPLIT = 42, 0.1


def load_combined():
    subs = []
    for d in DIRS:
        ds = HDF5Dataset(f"{d}/dataset.h5")
        ds.metadata = None
        merge_dataset_classes(ds, {"class_Seal": "class_Welding"})  # 13-class, no drops
        subs.append(ds)
    canon = list(subs[0].classes)
    for ds in subs:
        align_dataset_to_classes(ds, canon)
    return CombinedDataset(subs)


def load_combined_from_cfg(cfg):
    """Rebuild the combined dataset a given model was trained on, from its sidecar.

    Mirrors the trainer's per-dataset transform chain (merge -> drops -> align to the
    canonical list) so the frame-disjoint split below lands on exactly the tiles that
    model saw as validation. Returns (dataset, dir_list).
    """
    dirs = list(cfg["training_dataset"])
    subs = []
    for d in dirs:
        ds = HDF5Dataset(f"{d}/dataset.h5")
        ds.metadata = None
        if cfg.get("class_merges"):
            merge_dataset_classes(ds, cfg["class_merges"])
        drops = list(cfg.get("drop_classes") or []) + resolve_auto_drops(ds.classes, cfg)
        if drops:
            drop_dataset_classes(ds, drops)
        subs.append(ds)
    # the sidecar 'classes' is what the trained head actually emits -- align to it.
    canon = cfg.get("classes") or list(subs[0].classes)
    for ds in subs:
        align_dataset_to_classes(ds, canon)
    return CombinedDataset(subs), dirs


def _score(clf, val, CLEAN):
    loader = DataLoader(val, batch_size=384, shuffle=False, num_workers=8,
                        collate_fn=metadata_collate_fn)
    out = []
    with torch.no_grad():
        for b in loader:
            p = torch.softmax(clf(b[0].cuda()), dim=1)
            out.append((1.0 - p[:, CLEAN]).float().cpu().numpy())
    return np.concatenate(out)


def main_models(pths):
    """Per-domain evaluation for explicit checkpoints, each on its own split."""
    rows = {}
    for pth in pths:
        cfg = json.load(open(os.path.splitext(pth)[0] + ".json"))
        ds, dirs = load_combined_from_cfg(cfg)
        classes = list(ds.classes)
        CLEAN = classes.index("class_clean")
        split = cfg.get("dataloader", {})
        seed = split.get("seed", SEED)
        vs = split.get("validation_split", VAL_SPLIT)
        _, va_idx = frame_disjoint_split(ds, vs, seed)
        va_idx = np.asarray(va_idx)
        bounds = list(ds._offsets) + [ds._total_len]
        names = [os.path.basename(d.rstrip("/")) for d in dirs]
        dom = np.empty(len(va_idx), dtype=object)
        for i, n in enumerate(names):
            dom[(va_idx >= bounds[i]) & (va_idx < bounds[i + 1])] = n
        targets = np.asarray(ds.targets)[va_idx]
        isdef = targets != CLEAN

        clf = _load_weights(_instantiate_classifier(cfg, classes), pth, "cuda").cuda().eval()
        s = _score(clf, Subset(ds, va_idx.tolist()), CLEAN)
        del clf; torch.cuda.empty_cache()

        def miss_at_fa(mask, fa):
            clean = mask & ~isdef
            if clean.sum() == 0 or (mask & isdef).sum() == 0:
                return float("nan")
            thr = np.quantile(s[clean], 1 - fa / 100.0)
            return 100.0 * (~(s >= thr)[mask & isdef]).mean()

        name = os.path.basename(pth)
        print(f"\n=== {name}  ({len(classes)} classes, mono="
              f"{cfg['hparams'].get('monochrome')})")
        print(f"    held-out {len(va_idx)} tiles ({int(isdef.sum())} defect)")
        print(f"    {'domain':28} {'tiles':>8} {'def':>7} {'AUROC':>8} {'miss@5':>8} {'miss@10':>8}")
        r = {}
        for n in names:
            m = dom == n
            if m.sum() == 0:
                continue
            nd = int(isdef[m].sum())
            au = roc_auc_score(isdef[m].astype(int), s[m]) if 0 < nd < int(m.sum()) else float("nan")
            r[n] = (au, miss_at_fa(m, 5), miss_at_fa(m, 10), int(m.sum()), nd)
            print(f"    {n:28} {int(m.sum()):8d} {nd:7d} {au:8.4f} "
                  f"{r[n][1]:8.1f} {r[n][2]:8.1f}")
        rows[name] = r
    json.dump(rows, open("domain_split_models.json", "w"), indent=2)
    print("\nwrote domain_split_models.json")


def main():
    if len(sys.argv) > 1:
        return main_models(sys.argv[1:])
    ds = load_combined()
    classes = list(ds.classes)
    CLEAN = classes.index("class_clean")
    boundary = ds._offsets[1]  # combined idx >= boundary  ==>  Altinay
    print(f"classes ({len(classes)}): {classes}")
    print(f"combined tiles: {ds._total_len}  (FORTH<{boundary}<=Altinay)")

    _, va_idx = frame_disjoint_split(ds, VAL_SPLIT, SEED)
    va_idx = np.asarray(va_idx)
    dom = np.where(va_idx >= boundary, "altinay", "forth")   # per held-out tile
    targets = np.asarray(ds.targets)[va_idx]
    isdef = targets != CLEAN
    print(f"held-out tiles: {len(va_idx)}  | FORTH {int((dom=='forth').sum())} "
          f"(def {int(isdef[dom=='forth'].sum())})  | "
          f"Altinay {int((dom=='altinay').sum())} (def {int(isdef[dom=='altinay'].sum())})")

    val = Subset(ds, va_idx.tolist())

    def miss_at_fa(s, mask, fa):
        clean = mask & ~isdef
        thr = np.quantile(s[clean], 1 - fa / 100.0)
        fire = s >= thr
        dfk = mask & isdef
        return 100.0 * (~fire[dfk]).mean()

    print(f"\n{'backbone':20}| {'FORTH (non-Altinay)':^28}| {'Altinay (target)':^28}")
    print(f"{'':20}| {'AUROC':>7} {'miss@5':>7} {'miss@10':>7} | "
          f"{'AUROC':>7} {'miss@5':>7} {'miss@10':>7}")
    print("-" * 82)
    rows = {}
    for bb in BACKBONES:
        cfg = json.load(open(f"mix_{bb}.json"))
        mcls = cfg.get("classes") or classes
        clf = _load_weights(_instantiate_classifier(cfg, mcls), f"mix_{bb}.pth", "cuda").cuda().eval()
        loader = DataLoader(val, batch_size=384, shuffle=False, num_workers=8,
                            collate_fn=metadata_collate_fn)
        out = []
        with torch.no_grad():
            for b in loader:
                p = torch.softmax(clf(b[0].cuda()), dim=1)
                out.append((1.0 - p[:, CLEAN]).float().cpu().numpy())
        s = np.concatenate(out)
        r = {}
        for name, m in (("forth", dom == "forth"), ("altinay", dom == "altinay")):
            au = roc_auc_score(isdef[m].astype(int), s[m])
            r[name] = (au, miss_at_fa(s, m, 5), miss_at_fa(s, m, 10))
        rows[bb] = r
        f, a = r["forth"], r["altinay"]
        print(f"{bb:20}| {f[0]:7.4f} {f[1]:7.1f} {f[2]:7.1f} | "
              f"{a[0]:7.4f} {a[1]:7.1f} {a[2]:7.1f}")
        del clf
        torch.cuda.empty_cache()
    json.dump(rows, open("mix_domain_split.json", "w"), indent=2)
    print("\nwrote mix_domain_split.json")


if __name__ == "__main__":
    main()
