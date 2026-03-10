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
from trainClassifierTorch import load_hyperparameters, RGBAImageFolder


class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
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

        # Keep targets compatible with ImageFolder-like code
        self.targets = [int(x) for x in self.labels[:]]

    def __len__(self):
        return len(self.labels)

    def _decode_metadata(self, idx):
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
        x = torch.tensor(self.images[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.metadata is not None:
            meta = self._decode_metadata(idx)
            return x, y, meta

        return x, y

    def close(self):
        try:
            if hasattr(self, "file") and self.file is not None:
                self.file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


if __name__ == "__main__":

    configuration_file = "config.json"
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]

    print(f"Using {configuration_file} configuration for packaging dataset")

    config_json = load_hyperparameters(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), configuration_file)
    )
    directory = config_json["directory"]

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

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
        img_dset = f.create_dataset(
            "images",
            shape=(n_samples, c, h, w),
            dtype=np.float32,
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
