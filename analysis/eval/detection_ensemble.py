#!/usr/bin/python3
"""Ensemble search on the DETECTION metric (miss @ matched false-alarm), not the
13-way balanced accuracy that evaluateOptimalEnsemble.py optimizes. Uses the same
probs npz from calculateOptimalEnsemble.py. Soft ensemble = average P(clean) over
the subset; detector score = 1 - mean P(clean). Also splits by domain (FORTH vs
Altinay = the Camera V2 pilot target) using the exact seed42 held-out order."""
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
from mvc.core.metrics import miss_at_fa_from_scores

NPZ = sys.argv[1] if len(sys.argv) > 1 else None
META = NPZ.replace(".npz", ".json")
d = np.load(NPZ)
probs, y = d["probs"], d["y_true"]          # (M,N,13), (N,)
names = [m["name"] for m in json.load(open(META))["models"]]
ms = {m["name"]: m["ms_per_sample"] for m in json.load(open(META))["models"]}
classes = json.load(open(META))["classes"]
CLEAN = classes.index("class_clean")
isdef = (y != CLEAN)
M = probs.shape[0]

# per-model P(clean); ensemble clean prob = mean over subset
pclean = probs[:, :, CLEAN]                 # (M,N)

# domain tags in npz order (== held-out H5 order == seed42 va order)
from mvc.core.dataset_converter import HDF5Dataset
from mvc.core.class_scheme import merge_dataset_classes, align_dataset_to_classes
from mvc.core.datasets import CombinedDataset, frame_disjoint_split
DIRS = ["/home/ammar/Documents/Programming/magician_datasets/train_nonaltinay_v2",
        "/home/ammar/Documents/Programming/magician_datasets/val_altinay_granular"]
subs = []
for dd in DIRS:
    ds = HDF5Dataset(f"{dd}/dataset.h5"); ds.metadata = None
    merge_dataset_classes(ds, {"class_Seal": "class_Welding"}); subs.append(ds)
canon = list(subs[0].classes)
for ds in subs: align_dataset_to_classes(ds, canon)
comb = CombinedDataset(subs); boundary = comb._offsets[1]
_, va = frame_disjoint_split(comb, 0.1, 42); va = np.asarray(va)
dom_altinay = va >= boundary
assert len(va) == len(y)

def metrics(score, mask):
    au = roc_auc_score(isdef[mask].astype(int), score[mask])
    # Shared KPI (Metrics.miss_at_fa_from_scores); fa is a fraction, not a percentage.
    def miss(fa):
        return miss_at_fa_from_scores(score, mask & ~isdef, mask & isdef, fa)
    return au, miss(0.05), miss(0.10)

def ens_score(idx):                          # idx: tuple of model indices
    return 1.0 - pclean[list(idx)].mean(axis=0)

allmask = np.ones(len(y), bool)
print("=== single models (detection) ===")
print(f"{'model':20} {'AUROC':>7} {'miss@5':>7} {'miss@10':>7} {'ms':>6}")
singles = {}
for i, n in enumerate(names):
    au, m5, m10 = metrics(1.0 - pclean[i], allmask); singles[i] = m5
    print(f"{n:20} {au:7.4f} {m5:7.2f} {m10:7.2f} {ms[n]:6.3f}")

# greedy forward on miss@FA5 (aggregate)
print("\n=== greedy forward (minimize aggregate miss@FA5) ===")
chosen, best = [], 100.0
remaining = set(range(M))
while remaining:
    cand = min(remaining, key=lambda j: metrics(ens_score(tuple(chosen+[j])), allmask)[1])
    m5 = metrics(ens_score(tuple(chosen+[cand])), allmask)[1]
    if chosen and m5 >= best - 1e-9:         # no improvement
        break
    chosen.append(cand); remaining.discard(cand); best = m5
    au, m5b, m10b = metrics(ens_score(tuple(chosen)), allmask)
    par = max(ms[names[k]] for k in chosen)
    print(f"  + {names[cand]:20} -> subset={[names[k] for k in chosen]}  "
          f"AUROC {au:.4f}  miss@5 {m5b:.2f}  miss@10 {m10b:.2f}  ms_par {par:.3f}")

best_subset = tuple(chosen)
print(f"\nBEST detection ensemble: {[names[k] for k in best_subset]}")
for label, mask in (("FORTH", ~dom_altinay), ("Altinay(V2 target)", dom_altinay), ("aggregate", allmask)):
    au, m5, m10 = metrics(ens_score(best_subset), mask)
    print(f"  {label:20} AUROC {au:.4f}  miss@5 {m5:.2f}  miss@10 {m10:.2f}")
print("\n--- convnext_tiny alone, per domain (baseline) ---")
ci = names.index("mix_convnext_tiny")
for label, mask in (("FORTH", ~dom_altinay), ("Altinay(V2 target)", dom_altinay), ("aggregate", allmask)):
    au, m5, m10 = metrics(1.0 - pclean[ci], mask)
    print(f"  {label:20} AUROC {au:.4f}  miss@5 {m5:.2f}  miss@10 {m10:.2f}")
