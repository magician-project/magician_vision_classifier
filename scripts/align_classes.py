#!/usr/bin/env python3
"""
Align a validation dataset.h5 to the class list of a training dataset.h5.

The trainer (trainMagicianVisionClassifierTorch.py) demands identical class
lists when an explicit validation_dataset is used, and BalancedBatchSampler
crashes on classes with zero training samples. So: the TRAIN class list is
canonical; validation samples whose class does not exist in training are
dropped (a model can neither learn nor be fairly scored on them), and the
remaining labels are remapped. The validation file is rewritten sequentially
(gzip chunks) and atomically replaced.

Usage: align_classes.py <train_dataset.h5> <val_dataset.h5>
"""
import json
import os
import sys

import h5py
import numpy as np


def get_classes(path):
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["class_names"])


def main():
    train_h5, val_h5 = sys.argv[1], sys.argv[2]
    train_classes = get_classes(train_h5)
    val_classes = get_classes(val_h5)
    print("Train classes:", train_classes)
    print("Val classes:  ", val_classes)

    dropped = [c for c in val_classes if c not in train_classes]
    if dropped:
        print("Dropping val-only classes (not learnable):", dropped)

    with h5py.File(val_h5, "r") as f:
        labels = f["labels"][:]
        keep = np.array([val_classes[l] in train_classes for l in labels])
        new_label = np.array(
            [train_classes.index(val_classes[l]) if k else -1
             for l, k in zip(labels, keep)], dtype=np.int64)
        keep_idx = np.nonzero(keep)[0]
        print(f"Keeping {len(keep_idx)}/{len(labels)} validation samples")

        tmp = val_h5 + ".tmp"
        with h5py.File(tmp, "w") as g:
            shape = f["images"].shape
            imgs = g.create_dataset("images", shape=(len(keep_idx),) + shape[1:],
                                    dtype=np.uint8, compression="gzip",
                                    chunks=(1,) + shape[1:])
            meta = g.create_dataset("metadata", shape=(len(keep_idx),),
                                    dtype=h5py.string_dtype(encoding="utf-8"))
            for j, i in enumerate(keep_idx):
                imgs[j] = f["images"][i]
                meta[j] = f["metadata"][i]
                if j % 20000 == 0:
                    print(f"  copied {j}/{len(keep_idx)}", flush=True)
            g.create_dataset("labels", data=new_label[keep_idx])
            g.attrs["class_names"] = json.dumps(train_classes)
            g.attrs["has_metadata"] = True
            g.attrs["metadata_format"] = "json-per-sample"

    os.replace(tmp, val_h5)

    counts = np.bincount(new_label[keep_idx], minlength=len(train_classes))
    print("Final validation distribution:")
    for c, n in zip(train_classes, counts):
        print(f"  {c}: {n}")
    zero_val = [c for c, n in zip(train_classes, counts) if n == 0]
    if zero_val:
        print("NOTE: classes with zero validation samples:", zero_val)


if __name__ == "__main__":
    main()
