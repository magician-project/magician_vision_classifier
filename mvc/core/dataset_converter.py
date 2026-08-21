#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

import os
import sys
import json
import h5py
import torch
import numpy as np

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from mvc.core.config import load_hyperparameters
from mvc.core.datasets import RGBAImageFolder
from mvc.paths import repo_root


class HDF5Dataset(Dataset):
    """
    PyTorch Dataset backed by an HDF5 file.

    Supports both legacy float32 images (auto-converted to uint8) and native
    uint8 images. Optionally reads per-sample JSON metadata and class names
    stored as HDF5 attributes.
    """
    def __init__(self, h5_path):
        """Open the HDF5 file in read-only mode and extract images, labels, metadata, and class names."""
        self.file = h5py.File(h5_path, "r")
        self.images = self.file["images"]
        self.labels = self.file["labels"]

        self.metadata = self.file["metadata"] if "metadata" in self.file else None

        # Recover class names if stored as attribute
        if "class_names" in self.file.attrs:
            self.classes = json.loads(self.file.attrs["class_names"])
        elif "class_names" in self.file:
            self.classes = [x.decode("utf-8") for x in self.file["class_names"][()]]
        else:
            self.classes = []

        # Optional runtime label remap (old_idx -> new_idx), e.g. class merges.
        # None = identity. Set via apply_class_merges() so both __getitem__ and
        # targets stay consistent without rewriting the H5 file.
        self.label_map = None
        # Optional row subset (list/array of kept H5 row positions), e.g. after
        # dropping a class from training. None = use every row. When set,
        # __len__/__getitem__ address rows through it so the H5 file is never
        # rewritten. Set via drop_dataset_classes().
        self.indices = None
        # Keep targets compatible with ImageFolder-like code
        self.targets = [int(x) for x in self.labels[:]]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        """Return the number of samples (respecting an optional row subset)."""
        if self.indices is not None:
            return len(self.indices)
        return len(self.labels)

    def _decode_metadata(self, idx):
        """
        Decode the JSON metadata string for sample *idx*.

        Handles bytes, str, and other types; returns an empty dict on any error.
        """
        if self.metadata is None:
            return {}

        try:
            raw = self.metadata[idx]

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            elif hasattr(raw, "decode"):
                raw = raw.decode("utf-8", errors="ignore")
            else:
                raw = str(raw)

            raw = raw.strip()
            if raw == "":
                return {}

            return json.loads(raw)
        except Exception:
            return {}

    def __getitem__(self, idx):
        """
        Return the image tensor, label, and optional metadata for sample *idx*.

        Images are returned as uint8 tensors (normalisation happens on the GPU).
        Legacy float32 images are auto-converted to uint8 for consistency.
        Returns (x, y) or (x, y, meta) depending on whether metadata is available.
        """
        # Return uint8 tensors — normalisation (/255) happens inside the model on
        # the GPU, keeping CPU→GPU transfer bandwidth 4× lower than float32.
        # HDF5 files written before this change stored float32; detect that case
        # and convert back to uint8 so downstream code stays consistent.
        # Map the caller's index through the optional row subset first.
        row = int(self.indices[idx]) if self.indices is not None else idx
        raw = self.images[row]
        if raw.dtype == np.float32:
            # Legacy float32 HDF5 (values already in [0,1]) — re-quantise to uint8.
            x = torch.from_numpy((raw * 255.0).clip(0, 255).astype(np.uint8))
        else:
            x = torch.from_numpy(raw.astype(np.uint8))
        lbl = int(self.labels[row])
        if self.label_map is not None:
            lbl = int(self.label_map[lbl])
        y = torch.tensor(lbl, dtype=torch.long)

        if self.metadata is not None:
            meta = self._decode_metadata(row)
            return x, y, meta

        return x, y

    def close(self):
        """Close the underlying HDF5 file handle."""
        try:
            if hasattr(self, "file") and self.file is not None:
                self.file.close()
        except Exception:
            pass

    def __del__(self):
        """Destructor: ensure the HDF5 file handle is closed."""
        self.close()


if __name__ == "__main__":

    configuration_file = "config.json"
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]

    print(f"Using {configuration_file} configuration for packaging dataset")

    config_json = load_hyperparameters(
        os.path.join(repo_root(), configuration_file)
    )
    directory = config_json["directory"]

    # HWC uint8 numpy → CHW uint8 tensor; no /255 (done in model on GPU).
    transform = transforms.Lambda(
        lambda img: torch.from_numpy(img).permute(2, 0, 1).contiguous()
    )

    # Important: ask the loader to also return PNG metadata
    dataset = RGBAImageFolder(root=directory, transform=transform, return_metadata=True)
    n_samples = len(dataset)
    print(f"Found {n_samples} samples in {directory}")

    # Peek one image to get shape
    sample = dataset[0]
    if len(sample) == 3:
        sample_img, _, _ = sample
    else:
        sample_img, _ = sample

    c, h, w = sample_img.shape
    print(f"Image shape: {c}x{h}x{w}")

    output_path = os.path.join(directory, "dataset.h5")

    with h5py.File(output_path, "w") as f:
        # Store as uint8 — 4× smaller file and 4× less I/O vs float32.
        # Normalisation (/255) happens inside the model on the GPU.
        img_dset = f.create_dataset(
            "images",
            shape=(n_samples, c, h, w),
            dtype=np.uint8,
            compression="gzip",
            chunks=(1, c, h, w),
        )

        lbl_dset = f.create_dataset(
            "labels",
            shape=(n_samples,),
            dtype=np.int64,
        )

        # Variable-length UTF-8 strings for JSON metadata
        str_dtype = h5py.string_dtype(encoding="utf-8")
        meta_dset = f.create_dataset(
            "metadata",
            shape=(n_samples,),
            dtype=str_dtype,
        )

        class_names = dataset.classes
        class_str = json.dumps(class_names)
        f.attrs["class_names"] = class_str

        # Optional bookkeeping
        f.attrs["has_metadata"] = True
        f.attrs["metadata_format"] = "json-per-sample"

        for idx, sample in enumerate(tqdm(dataset, desc="Saving HDF5")):
            if len(sample) == 3:
                img, lbl, meta = sample
            else:
                img, lbl = sample
                meta = {}

            img_dset[idx] = img.numpy()
            lbl_dset[idx] = int(lbl)

            if meta is None:
                meta = {}
            meta_dset[idx] = json.dumps(meta)

    print(f"Dataset successfully saved to {output_path}")
    print(f"Classes: {dataset.classes}")
    print("Done!")
