#!/usr/bin/env python3
"""
Remove named classes from a dataset.h5 (sequential rewrite, atomic replace).
Usage: drop_classes.py <dataset.h5> <class_name> [<class_name> ...]
"""
import json
import os
import sys

import h5py
import numpy as np


def main():
    path = sys.argv[1]
    drop = set(sys.argv[2:])

    with h5py.File(path, "r") as f:
        classes = json.loads(f.attrs["class_names"])
        keep_classes = [c for c in classes if c not in drop]
        print("Dropping:", sorted(drop & set(classes)))
        mapping = {classes.index(c): i for i, c in enumerate(keep_classes)}
        labels = f["labels"][:]
        keep_idx = np.nonzero(np.isin(labels, list(mapping.keys())))[0]
        print(f"Keeping {len(keep_idx)}/{len(labels)} samples, "
              f"{len(keep_classes)} classes")

        tmp = path + ".tmp"
        with h5py.File(tmp, "w") as g:
            shape = f["images"].shape
            imgs = g.create_dataset("images", shape=(len(keep_idx),) + shape[1:],
                                    dtype=np.uint8, compression="gzip",
                                    chunks=(1,) + shape[1:])
            meta = g.create_dataset("metadata", shape=(len(keep_idx),),
                                    dtype=h5py.string_dtype(encoding="utf-8"))
            src_img, src_meta = f["images"], f["metadata"]
            for j, i in enumerate(keep_idx):
                imgs[j] = src_img[i]
                meta[j] = src_meta[i]
                if j % 50000 == 0:
                    print(f"  copied {j}/{len(keep_idx)}", flush=True)
            g.create_dataset(
                "labels",
                data=np.array([mapping[l] for l in labels[keep_idx]],
                              dtype=np.int64))
            g.attrs["class_names"] = json.dumps(keep_classes)
            g.attrs["has_metadata"] = True
            g.attrs["metadata_format"] = "json-per-sample"

    os.replace(tmp, path)
    print("Done:", path)


if __name__ == "__main__":
    main()
