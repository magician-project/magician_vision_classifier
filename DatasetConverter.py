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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from trainClassifierTorch import load_hyperparameters, RGBAImageFolder
 
# Define a PyTorch Dataset that reads from it
class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.file = h5py.File(h5_path, 'r')
        self.images = self.file['images']
        self.labels = self.file['labels']

        # Recover class names if stored as attribute
        if 'class_names' in self.file.attrs:
            self.classes = json.loads(self.file.attrs['class_names'])
        elif 'class_names' in self.file:
            self.classes = [x.decode('utf-8') for x in self.file['class_names'][()]]
        else:
            self.classes = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

#Main
if __name__ == "__main__":

    configuration_file = "config.json"
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]

    print(f"Using {configuration_file} configuration for packaging dataset")

    config_json = load_hyperparameters(os.path.join(os.path.dirname(os.path.abspath(__file__)), configuration_file))
    directory = config_json['directory']

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = RGBAImageFolder(root=directory, transform=transform)
    n_samples = len(dataset)
    print(f"Found {n_samples} samples in {directory}")

    # Peek one image to get shape
    sample_img, _ = dataset[0]
    c, h, w = sample_img.shape
    print(f"Image shape: {c}x{h}x{w}")

    output_path = os.path.join(directory, "dataset.h5")

    # Create the HDF5 file
    with h5py.File(output_path, "w") as f:
        img_dset = f.create_dataset(
            "images",
            shape=(n_samples, c, h, w),
            dtype=np.float32,
            compression="gzip",
            chunks=(1, c, h, w)  # per-image chunking for efficient IO
        )
        lbl_dset = f.create_dataset(
            "labels",
            shape=(n_samples,),
            dtype=np.int64
        )

        class_names = dataset.classes
        class_str = json.dumps(class_names)
        f.attrs['class_names'] = class_str

        # Write one sample at a time (streaming, low memory)
        for idx, (img, lbl) in enumerate(tqdm(dataset, desc="Saving HDF5")):
            img_dset[idx] = img.numpy()
            lbl_dset[idx] = lbl

    print(f"Dataset successfully saved to {output_path}")

    # Print class names
    print(f"Classes: {dataset.classes}")
    print("Done!")
