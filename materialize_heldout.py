#!/usr/bin/python3
"""Materialize the EXACT seed42 frame-disjoint held-out val (13-class, post-merge,
post-align — identical to what the mix_* models were validated on) into a single
standalone H5, so calculateOptimalEnsemble.py can score the 7 models leakage-free
(it has no frame-disjoint split of its own; the raw training dirs would leak).
Output classes == the mix models' 13-class list, so alignment is 1:1."""
import numpy as np, h5py, json
from DatasetConverter import HDF5Dataset
from trainMagicianVisionClassifierTorch import (
    merge_dataset_classes, align_dataset_to_classes, CombinedDataset, frame_disjoint_split)

DIRS = ["/home/ammar/Documents/Programming/magician_datasets/train_nonaltinay_v2",
        "/home/ammar/Documents/Programming/magician_datasets/val_altinay_granular"]
OUT = "/home/ammar/Documents/Programming/magician_datasets/mix_heldout/dataset.h5"
SEED, VAL_SPLIT = 42, 0.1

subs = []
for d in DIRS:
    ds = HDF5Dataset(f"{d}/dataset.h5"); ds.metadata = None
    merge_dataset_classes(ds, {"class_Seal": "class_Welding"})
    subs.append(ds)
canon = list(subs[0].classes)
for ds in subs:
    align_dataset_to_classes(ds, canon)
comb = CombinedDataset(subs)
classes = list(comb.classes)
boundary = comb._offsets[1]
print("classes:", classes)

_, va = frame_disjoint_split(comb, VAL_SPLIT, SEED)
va = np.asarray(va)
targets = np.asarray(comb.targets)[va].astype(np.int64)
N = len(va)
C, H, W = subs[0].images.shape[1:]
print(f"held-out N={N}  shape=({C},{H},{W})  boundary={boundary}")

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with h5py.File(OUT, "w") as f:
    imgs = f.create_dataset("images", shape=(N, C, H, W), dtype=np.uint8,
                            chunks=(256, C, H, W))
    f.create_dataset("labels", data=targets, dtype=np.int64)
    f.attrs["class_names"] = json.dumps(classes)
    f.attrs["has_metadata"] = False
    for sub_i, src in enumerate(subs):
        off = comb._offsets[sub_i]
        sub_len = len(src)                   # subset length (post-align .indices)
        m = (va >= off) & (va < off + sub_len)
        pos = np.where(m)[0]                 # increasing output positions
        local = (va[m] - off)                # index into the (subset) dataset
        # map subset index -> RAW H5 row (align may have set an .indices subset)
        raw = np.asarray(src.indices)[local] if src.indices is not None else local
        order = np.argsort(raw)              # h5py needs increasing read index
        block = src.images[np.sort(raw)]     # bulk sorted read (raw rows)
        buf = np.empty_like(block)
        buf[order] = block                   # restore va order
        imgs[pos] = buf                      # increasing-index scatter write
        print(f"  wrote {len(pos)} tiles from sub{sub_i} (indices={'yes' if src.indices is not None else 'no'})")
        del block, buf
print("done ->", OUT)
