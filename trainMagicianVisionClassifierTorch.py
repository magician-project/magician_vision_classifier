#!/usr/bin/python3

""" 
Author : "Nikos Vasilikopoulos, Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

import datetime
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    regnet_y_800mf, RegNet_Y_800MF_Weights,
    regnet_y_400mf, RegNet_Y_400MF_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    squeezenet1_1, SqueezeNet1_1_Weights,
    densenet121, DenseNet121_Weights,
    mnasnet0_5, MNASNet0_5_Weights,
    mnasnet1_0, MNASNet1_0_Weights,
)
from torchmetrics import Accuracy, Recall, Precision, AUROC
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def filter_dataset_classes(dataset, keep_classes):
    """
    Keeps only the specified classes in the dataset and removes the rest.

    Args:
        dataset (RGBAImageFolder): The dataset to filter.
        keep_classes (list[str]): A list of class names to keep.
    """
    # Get indices of the classes to keep
    keep_indices = [dataset.class_to_idx[c] for c in keep_classes if c in dataset.class_to_idx]

    # Filter samples and targets
    filtered_samples = []
    filtered_targets = []

    for sample, target in dataset.samples:
        if target in keep_indices:
            filtered_samples.append((sample, target))
            filtered_targets.append(target)

    dataset.samples = filtered_samples
    dataset.targets = filtered_targets

    # Update class lists
    dataset.classes = keep_classes
    dataset.class_to_idx = {cls: i for i, cls in enumerate(keep_classes)}

    print(f"Filtered dataset to {len(dataset.samples)} samples across {len(keep_classes)} classes.")


def align_dataset_to_classes(dataset, target_classes):
    """Force a dataset onto a canonical class list `target_classes` (exact order).
    Samples whose class is not in target are DROPPED; target classes absent from
    the dataset stay at zero count. Lets heterogeneous H5s (different class
    subsets/orders, e.g. base train + a granular dump collapsed to base) share one
    label space so CombinedDataset accepts them. Uses the same H5 label_map+indices
    / ImageFolder mechanism as drop_dataset_classes."""
    import numpy as _np
    old_classes = list(dataset.classes)
    if old_classes == list(target_classes):
        return dataset
    tgt_idx = {c: i for i, c in enumerate(target_classes)}
    # current class index -> target index (or -1 if the class is not in target)
    cur2tgt = _np.array([tgt_idx.get(c, -1) for c in old_classes], dtype=_np.int64)

    if hasattr(dataset, "labels") and hasattr(dataset, "images"):   # HDF5Dataset
        raw = _np.asarray(dataset.labels[:], dtype=_np.int64)
        raw2cur = (_np.asarray(dataset.label_map, dtype=_np.int64)
                   if dataset.label_map is not None
                   else _np.arange(max(int(raw.max()) + 1 if len(raw) else 0,
                                       len(old_classes)), dtype=_np.int64))
        cur = raw2cur[raw]
        keep = _np.where(cur2tgt[cur] >= 0)[0]
        if dataset.indices is not None:
            keep = _np.intersect1d(_np.asarray(dataset.indices), keep)
        dataset.label_map = _np.array([cur2tgt[c] if 0 <= c < len(cur2tgt) else -1
                                       for c in raw2cur], dtype=_np.int64)
        dataset.indices = keep
        dataset.targets = [int(dataset.label_map[int(r)]) for r in raw[keep]]
    else:
        samples = getattr(dataset, "samples", None)
        if samples is not None:
            samples = [(s, int(cur2tgt[t])) for (s, t) in samples if cur2tgt[t] >= 0]
            dataset.samples = samples
            dataset.targets = [t for (_, t) in samples]
        elif getattr(dataset, "targets", None) is not None:
            dataset.targets = [int(cur2tgt[t]) for t in dataset.targets if cur2tgt[t] >= 0]

    dataset.classes = list(target_classes)
    dataset.class_to_idx = dict(tgt_idx)
    print(f"[align_classes] -> {len(target_classes)} canonical classes: {list(target_classes)}")
    return dataset


def strip_severity_classes(dataset):
    """Collapse severity-tagged class names to their base at runtime, e.g.
    class_WeldingClassA / class_WeldingClassB -> class_Welding. This is the
    "non-severity" view of a granular H5: dumps store the full severity/class
    truth, and the config decides at load time whether to keep or drop severity.
    A no-op on H5s that were dumped without severity (base names already).
    Implemented via merge_dataset_classes so labels/targets/label_map stay
    consistent. The suffix is 'Class<LETTER>' at the end of the class name."""
    import re as _re
    merges = {}
    for c in list(dataset.classes):
        base = _re.sub(r'Class[A-Z]$', '', c)
        if base != c:
            merges[c] = base
    if merges:
        merge_dataset_classes(dataset, merges)
    else:
        print("[strip_severity] no severity-tagged classes to collapse")
    return dataset


def merge_dataset_classes(dataset, merges):
    """Merge/rename classes at runtime WITHOUT rewriting the H5 file. `merges`: dict
    of {source_class: destination_class}. Source samples are relabeled to the
    destination; source class dropped, rest renumbered. The destination MAY be a
    brand-new name that is not an existing class -- that is how you rename or bucket,
    e.g. collapse every defect into one class for a binary detector:
        {"class_Deformation":"class_defect", ..., "class_Welding":"class_defect"}
        -> classes become [class_clean, class_defect].
    Chains (A->B, B->C) resolve transitively to the terminal destination. Apply
    identically to train + val so their label spaces match. Honored by
    HDF5Dataset.__getitem__ via label_map, and by sample/target-list datasets."""
    import numpy as _np
    if not merges:
        return dataset
    old_classes = list(dataset.classes)
    old_cti = {c: i for i, c in enumerate(old_classes)}
    # source must exist and differ from its destination; destination may be new.
    merges = {s: d for s, d in merges.items() if s in old_cti and s != d}
    if not merges:
        print("[merge_classes] nothing to merge"); return dataset
    def _terminal(c):
        seen = set()
        while c in merges and c not in seen:
            seen.add(c); c = merges[c]
        return c
    final = {c: _terminal(c) for c in old_classes}   # final label for every old class
    dropped = set(merges.keys())                     # every source disappears
    # survivors keep their order; then append any brand-new destination names.
    new_classes = [c for c in old_classes if c not in dropped]
    for c in old_classes:
        d = final[c]
        if d not in new_classes:
            new_classes.append(d)
    new_cti = {c: i for i, c in enumerate(new_classes)}
    remap = _np.empty(len(old_classes), dtype=_np.int64)
    for c, oi in old_cti.items():
        remap[oi] = new_cti[final[c]]
    if hasattr(dataset, "label_map"):
        # `remap` is CURRENT-index -> new-index. H5 rows always carry RAW labels,
        # so label_map must stay raw->current. When a prior merge (e.g.
        # strip_severity) already set label_map (raw->current), COMPOSE with it
        # instead of overwriting -- otherwise a second merge stores a current->new
        # map that is wrong-domain for the raw rows (IndexError on drop).
        if dataset.label_map is not None:
            prev = _np.asarray(dataset.label_map, dtype=_np.int64)  # raw -> current
            dataset.label_map = remap[prev]                          # raw -> new
        else:
            dataset.label_map = remap
    if getattr(dataset, "targets", None) is not None:
        dataset.targets = [int(remap[int(t)]) for t in dataset.targets]
    if getattr(dataset, "samples", None) is not None:
        dataset.samples = [(s, int(remap[int(t)])) for (s, t) in dataset.samples]
    dataset.classes = new_classes
    dataset.class_to_idx = new_cti
    print(f"[merge_classes] {merges} -> {len(new_classes)} classes: {new_classes}")
    return dataset



class _SkipSweep(Exception):
    """Raised to skip the defect-vs-clean threshold sweep for runs with no clean
    class (alldefect). Caught right after the sweep; the confusion matrix, which
    runs earlier in the same block, is unaffected."""
    pass


def drop_dataset_classes(dataset, drop):
    """Remove all samples of the named classes from `dataset`, renumbering the
    survivors to a contiguous 0..k-1 label space. `drop`: list of class names,
    e.g. ["class_clean"] to build an 'alldefect' typer that never sees clean.

    Operates in the CURRENT label space, so run it AFTER merge_dataset_classes.
    Works on HDF5Dataset (row subset via .indices, no H5 rewrite) and on
    ImageFolder-style datasets (.samples/.targets). Apply identically to train +
    val so their label spaces match. NOTE: dropping class_clean removes the
    clean class entirely -- val_detect_auroc / penalize_false_clean are undefined
    then; select on val_auroc and set penalize_false_clean=0 for such runs."""
    import numpy as _np
    if not drop:
        return dataset
    old_cti = getattr(dataset, "class_to_idx", None) or {c: i for i, c in enumerate(dataset.classes)}
    drop = [c for c in drop if c in old_cti]
    if not drop:
        print("[drop_classes] nothing to drop (names not present)"); return dataset
    old_classes = list(dataset.classes)
    drop_set = set(drop)
    new_classes = [c for c in old_classes if c not in drop_set]
    new_cti = {c: i for i, c in enumerate(new_classes)}
    drop_idx = {old_cti[c] for c in drop}
    # current-label-index -> new-index (or -1 if dropped)
    remap = _np.full(len(old_classes), -1, dtype=_np.int64)
    for c, oi in old_cti.items():
        if c in new_cti:
            remap[oi] = new_cti[c]

    if hasattr(dataset, "labels") and hasattr(dataset, "images"):
        # HDF5Dataset: rows carry RAW labels; label_map (if any) maps raw->current.
        raw = _np.asarray(dataset.labels[:], dtype=_np.int64)
        raw2cur = (_np.asarray(dataset.label_map, dtype=_np.int64)
                   if dataset.label_map is not None
                   else _np.arange(max(int(raw.max()) + 1 if len(raw) else 0,
                                       len(old_classes)), dtype=_np.int64))
        cur = raw2cur[raw]
        keep = _np.where(~_np.isin(cur, list(drop_idx)))[0]
        if dataset.indices is not None:                    # compose with any prior subset
            keep = _np.intersect1d(_np.asarray(dataset.indices), keep)
        # new raw->new-index map (dropped raws map to -1 but their rows are excluded)
        dataset.label_map = _np.array([remap[c] if 0 <= c < len(remap) else -1
                                       for c in raw2cur], dtype=_np.int64)
        dataset.indices = keep
        dataset.targets = [int(dataset.label_map[int(r)]) for r in raw[keep]]
    else:
        samples = getattr(dataset, "samples", None)
        if samples is not None:
            samples = [(s, int(remap[t])) for (s, t) in samples if t not in drop_idx]
            dataset.samples = samples
            dataset.targets = [t for (_, t) in samples]
        elif getattr(dataset, "targets", None) is not None:
            dataset.targets = [int(remap[t]) for t in dataset.targets if t not in drop_idx]

    dataset.classes = new_classes
    dataset.class_to_idx = new_cti
    print(f"[drop_classes] dropped {sorted(drop_set)} -> {len(new_classes)} classes: {new_classes}")
    return dataset


def resolve_auto_drops(classes, config_json, clean_name='class_clean'):
    """Extra class NAMES to drop, computed from two convention toggles so callers
    don't have to enumerate exact granular names in drop_classes. Operates in the
    CURRENT label space (run alongside/after strip_severity + merge):

      drop_severityless_defects (bool): drop every non-clean class that lacks a
          severity suffix (Class A/B/C) -- e.g. the bare 'class_PositiveDent'
          bucket and stray 'class_Suspicious'/'class_Unknown'. 'class_clean' (the
          only legitimately severity-less class) is always kept.
      drop_class_families (list): drop every class whose base name (severity
          stripped) matches an entry, regardless of severity -- e.g.
          'class_Dust' drops class_Dust, class_DustClassA, class_DustClassB, ...

    Returns a sorted list; names not present are ignored by drop_dataset_classes.
    """
    import re as _re
    suffix = _re.compile(r'Class[ABC]$')
    drop = set()
    if config_json.get('drop_severityless_defects'):
        for c in classes:
            if c != clean_name and not suffix.search(c):
                drop.add(c)
    families = set(config_json.get('drop_class_families') or [])
    if families:
        for c in classes:
            if c in families or suffix.sub('', c) in families:
                drop.add(c)
    return sorted(drop)


def _dataset_source_frames(dataset):
    """Per-sample source-FRAME string for every sample (aligned with dataset[i],
    0..len-1). Frame = metadata 'source' path minus the tile (x,y) offset. Works
    on HDF5Dataset (reads its open .file, respects a prior .indices row subset)
    and CombinedDataset (concatenates constituents in the same order it indexes).
    'source' paths are globally unique across dumps, so combined frame ids stay
    distinct across FORTH / Altinay / etc. without extra offsetting."""
    import json as _json
    def _h5_frames(ds):
        raw = ds.file["metadata"][:]
        rows = ds.indices if getattr(ds, "indices", None) is not None else range(len(ds))
        out = []
        for r in rows:
            m = raw[int(r)]
            m = m.decode() if isinstance(m, bytes) else m
            out.append(_json.loads(m)["source"].rsplit("(", 1)[0])
        return out
    if hasattr(dataset, "datasets"):          # CombinedDataset
        out = []
        for ds in dataset.datasets:
            out.extend(_h5_frames(ds))
        return out
    if hasattr(dataset, "file"):              # HDF5Dataset
        return _h5_frames(dataset)
    raise ValueError("frame_disjoint_split requires H5 'source' metadata; this "
                     "dataset exposes none (PNG ImageFolder is not frame-aware).")


def frame_disjoint_split(dataset, val_split, seed):
    """Split a dataset into (train_idx, val_idx) SAMPLE-index lists by FRAME, not
    by tile — whole source frames go entirely to train or entirely to val, so
    tiles of the same frame/point never straddle the split. Essential when there
    are many tiles per point (v2 has ~16), where a tile-level random_split leaks
    near-duplicate siblings across train/val and inflates the val metric. Works
    across combined datasets (mixed domains) since frame ids are global. Returns
    indices INTO `dataset` (0..len(dataset)-1)."""
    import numpy as _np
    srcs = _np.array(_dataset_source_frames(dataset))
    _, frame_id = _np.unique(srcs, return_inverse=True)            # per SAMPLE
    uf = _np.unique(frame_id)
    rng = _np.random.default_rng(seed)
    n_val = max(1, int(round(len(uf) * val_split)))
    val_frames = set(rng.choice(uf, n_val, replace=False).tolist())
    is_val = _np.array([fr in val_frames for fr in frame_id])
    train_idx = _np.where(~is_val)[0].tolist()
    val_idx = _np.where(is_val)[0].tolist()
    print(f"[frame_disjoint_split] {len(uf)} frames -> {len(uf)-n_val} train / {n_val} val "
          f"({len(train_idx)} / {len(val_idx)} tiles); frames do not straddle the split")
    return train_idx, val_idx


def evaluate_dumped_tiles(model, tiles_dir, classes, device='cuda', batch_size=16):
    """
    Evaluates dumped PNG tiles against the current model.

    Args:
        model (torch.nn.Module): Trained classifier model.
        tiles_dir (str): Path to the directory containing dumped tiles.
        classes (list): List of class names (order must match model outputs).
        device (str): 'cuda' or 'cpu'.
        batch_size (int): Batch size for inference.

    Returns:
        dict: metrics containing accuracy, confusion matrix, and report.
    """

    import re
    from tqdm import tqdm
    from sklearn.metrics import confusion_matrix, classification_report
    model.eval()
    model.to(device)

    # HWC uint8 → CHW uint8 tensor; /255 normalisation is done inside the model.
    transform = transforms.Lambda(
        lambda img: torch.from_numpy(img).permute(2, 0, 1).contiguous()
    )

    # Collect all .png tiles
    all_files = [os.path.join(tiles_dir, f) for f in os.listdir(tiles_dir) if f.lower().endswith('.png')]
    if not all_files:
        print(f"No PNG files found in {tiles_dir}")
        return None

    y_true = []
    y_pred = []

    batch = []
    batch_gt = []

    print(f"Evaluating {len(all_files)} dumped tiles from '{tiles_dir}'...")

    for fpath in tqdm(all_files):
        # Parse filename pattern: tile_000001_y0_x0_cls2_dust.png
        match = re.search(r"_cls(-?\d+)_", os.path.basename(fpath))
        if not match:
            print(f"Could not extract class ID from {fpath}")
            continue
        gt_cls = int(match.group(1))
        if gt_cls < 0 or gt_cls >= len(classes):
            continue  # skip unknown class

        # Load RGBA image
        img = load_rgba_image(fpath)
        img = transform(img)
        batch.append(img)
        batch_gt.append(gt_cls)

        # Process in batches
        if len(batch) == batch_size:
            inputs = torch.stack(batch).to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch_gt)
            batch.clear()
            batch_gt.clear()

    # Handle remainder
    if batch:
        inputs = torch.stack(batch).to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(batch_gt)

    # Compute metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)

    print(f"\nEvaluation complete — Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Return results as dictionary
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report
    }



#-------------------------------------------------------------------------------
def checkIfPathExists(filename):
    """Return True if the given path exists (file or directory)."""
    return os.path.exists(filename)
#-------------------------------------------------------------------------------
def checkIfPathIsDirectory(filename):
    """Return True if the given path exists and is a directory."""
    return os.path.isdir(filename)
#-------------------------------------------------------------------------------
def checkIfFileExists(filename):
    """Return True if the given path exists and is a file."""
    return os.path.isfile(filename)
#-------------------------------------------------------------------------------

def load_hyperparameters(config_file):
    """
    Load and parse a JSON configuration file containing model hyperparameters,
    dataloader settings, optimizer config, and training options.

    Args:
        config_file: Path to a .json configuration file.

    Returns:
        Parsed dict with all configuration values.

    Exits:
        sys.exit(1) if the file does not exist.
    """
    if not checkIfFileExists(config_file):
        print("Config file not found")
        sys.exit(1)
    with open(config_file) as json_file:
        data = json.load(json_file)
    # Inherit shared defaults from configs/common.json (next to the trainer), so
    # cross-cutting settings (e.g. class_merges, the discard toggles) live in ONE
    # place for every config regardless of where the config file itself sits. The
    # specific config deep-overrides the common one.
    common_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "common.json")
    if checkIfFileExists(common_path):
        with open(common_path) as cf:
            common = json.load(cf)
        def _deep_merge(base, override):
            out = dict(base)
            for k, v in override.items():
                out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
            return out
        data = _deep_merge(common, data)
        print("Merged shared defaults from", common_path)
    return data


class CategoricalFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Categorical Focal Loss for multi-class classification.

        :param gamma: Focusing parameter (higher values focus more on hard examples)
        :param alpha: Class weighting (list or tensor of shape [num_classes] or None)
        :param reduction: Reduction mode: 'none' | 'mean' | 'sum'
        """
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Computes the focal loss.
        
        :param logits: Tensor of shape [batch_size, num_classes] (raw model outputs before softmax)
        :param targets: Tensor of shape [batch_size] with class indices (integer labels)
        :return: Scalar loss value
        """
        # Convert class indices to one-hot encoding
        num_classes = logits.shape[1]
        #target_one_hot = F.one_hot(targets, num_classes).float() #<-Original 
        target_one_hot = F.one_hot(targets, num_classes).to(torch.float32)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Gather the probabilities corresponding to the true class
        pt = (probs * target_one_hot).sum(dim=1)  # Shape: [batch_size]

        # Compute focal loss term
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Apply class weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # Select alpha value for each target
            ce_loss = alpha_t * ce_loss

        # Compute final loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Return per-sample loss if 'none'



class BasicResBlock(nn.Module):
    """Stride-1 residual block (conv-bn-relu-conv-bn + identity skip, then relu).
    Same channel count in/out so it can be stacked at a fixed stage width;
    lets a deeper FROM-SCRATCH net train inside a short epoch budget."""
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class WaveletPool(nn.Module):
    """Lossless 2x downsample: one level of Haar DWT, subbands stacked on channels.

    Drop-in for MaxPool2d(2), except C -> 4C: returns [LL,LH,HL,HH] per input
    channel. The transform is orthonormal and critically sampled, so it is
    invertible -- nothing is discarded, whereas MaxPool2d(2) throws away 3 of
    every 4 samples. The following stage conv then *learns* what to keep instead
    of the pool deciding blindly.

    Scale budget for the 300um (~3px at demosaic) defect on a 48x48 tile:
      stage 1 pool 48->24, level-1 detail resolves ~2px structure  <- the defect
      stage 2 pool 24->12, level-2 detail resolves ~4px            <- borderline
      stage 3 pool 12->6 (~8px) and stage 4 6->3 (~16px)           <- past it
    So this only earns its 4x channel cost at the first pool (and marginally the
    second); leave the later stages on MaxPool. See custom_wavelet_pools.

    Filters are fixed -> zero parameters. Note the decimated DWT is not
    shift-invariant, but it runs on conv1/early_convs feature maps (RF 7), not
    raw pixels, so the 3px response is already spatially smeared before decimation.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        # 2D Haar basis from l=[1,1]/sqrt(2), h=[1,-1]/sqrt(2); each has unit norm.
        ll = torch.tensor([[ 1.,  1.], [ 1.,  1.]]) * 0.5
        lh = torch.tensor([[ 1.,  1.], [-1., -1.]]) * 0.5   # horizontal edges
        hl = torch.tensor([[ 1., -1.], [ 1., -1.]]) * 0.5   # vertical edges
        hh = torch.tensor([[ 1., -1.], [-1.,  1.]]) * 0.5   # diagonal
        w = torch.stack([ll, lh, hl, hh]).unsqueeze(1)      # (4,1,2,2)
        self.register_buffer('filters', w.repeat(self.channels, 1, 1, 1))

    def forward(self, x):
        # groups=C: input channel g -> output channels 4g..4g+3 = its LL,LH,HL,HH
        return F.conv2d(x, self.filters, stride=2, groups=self.channels)


class CustomCNN(nn.Module):
    """
    A lightweight 4-layer convolutional network with adaptive global pooling
    and a two-tier fully-connected head. Designed for small tile images.

    Architecture:
        Conv2d → BN → ReLU → MaxPool (×4 stages, channels double each stage)
        → AdaptiveAvgPool2d(1,1) → Flatten
        → Linear → InstanceNorm1d → ReLU → Dropout
        → Linear → InstanceNorm1d → ReLU → Dropout
        → Linear → num_classes logits
    """
    def __init__(self, in_channels=4, intended_tile_size=64, num_classes=4, dropout_rate=0.5, base_channels=32, final_dense_layer=512, early_convs=0, channels=None, res_blocks=None, wavelet_pools=None):
        super(CustomCNN, self).__init__()
        if res_blocks is None:
            res_blocks = [0, 0, 0, 0]
        res_blocks = [int(x) for x in res_blocks]
        # 1-based stage indices whose MaxPool2d(2) is replaced by a lossless Haar
        # WaveletPool (which also widens that stage's output 4x, for free).
        wavelet_pools = set(int(s) for s in (wavelet_pools or []))

        # channels: explicit per-stage width [c1,c2,c3,c4]. Default is the
        # narrow-early/wide-late convention [base, 2base, 4base, 4base]. A
        # wide-early/taper-late schedule (e.g. [128,96,64,64]) puts capacity
        # where pooling has not yet destroyed information (see 3px-feature note).
        if channels is None:
            channels = [base_channels, base_channels*2, base_channels*4, base_channels*4]
        c1, c2, c3, c4 = [int(x) for x in channels]
        print("Custom CNN (",base_channels,",",final_dense_layer,") constructor / early_convs",
              early_convs, "/ channels", [c1,c2,c3,c4], "/ wavelet_pools", sorted(wavelet_pools))
        self.channels  = in_channels
        self.tile_size = intended_tile_size
        self.early_convs = int(early_convs)

        def make_pool(stage, ch):
            """1-based stage -> (pool module, channels it emits)."""
            if stage in wavelet_pools:
                return WaveletPool(ch), ch * 4
            return nn.MaxPool2d(2), ch

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)
        # 3px-feature preservation: extra stride-1 3x3 convs at FULL tile
        # resolution before the first pool, so a ~3px (300um) defect is encoded
        # into channels (RF grows 3->5->7...) before any downsampling. From-scratch
        # net -> no pretrained-body confound, unlike the resnet18 stems.
        self.early = nn.ModuleList()
        for _ in range(self.early_convs):
            self.early.append(nn.Sequential(
                nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1), nn.ReLU(inplace=True)))
        self.pool1, p1 = make_pool(1, c1)

        self.conv2 = nn.Conv2d(p1, c2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool2, p2 = make_pool(2, c2)

        self.conv3 = nn.Conv2d(p2, c3, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(c3)
        self.pool3, p3 = make_pool(3, c3)

        self.conv4 = nn.Conv2d(p3, c4, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(c4)
        self.pool4, p4 = make_pool(4, c4)

        # extra representational depth per stage (residual, at stage width),
        # inserted after the stage conv and BEFORE its pool
        stage_ch = [c1, c2, c3, c4]
        self.res = nn.ModuleList([
            nn.Sequential(*[BasicResBlock(stage_ch[i]) for _ in range(res_blocks[i])])
            for i in range(4)])
        print("Custom CNN res_blocks per stage:", res_blocks)

        prefinalLayerChannels     = int(final_dense_layer * 4.5)
        intermediateLayerChannels = int(final_dense_layer * 1.5)   # new FC layer

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(p4, prefinalLayerChannels)
        self.fc2 = nn.Linear(prefinalLayerChannels, intermediateLayerChannels)  # new layer
        self.bn_dense1 = nn.InstanceNorm1d(prefinalLayerChannels)
        self.bn_dense2 = nn.InstanceNorm1d(intermediateLayerChannels)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(intermediateLayerChannels, num_classes)

    def forward(self, x):
        """
        Forward pass through the CNN backbone + FC head.
        Validates input dimensions then streams through 4 conv stages,
        global average pooling, and a two-tier FC head.

        Args:
            x: Input tensor of shape (B, channels, tile_size, tile_size).

        Returns:
            Logits tensor of shape (B, num_classes).

        Raises:
            ValueError: If input spatial/channel dimensions don't match expected.
        """
        if x.shape[1:] != (self.channels, self.tile_size, self.tile_size):  # Sanity check on desired input size
          raise ValueError(f"Input size must be {self.channels}x{self.tile_size}x{self.tile_size}, got {x.shape[1:]}")
        x = F.relu(self.bn1(self.conv1(x)))
        for blk in self.early:
            x = blk(x)                     # extract 3px feature at full res
        x = self.pool1(self.res[0](x))
        x = self.pool2(self.res[1](F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.res[2](F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.res[3](F.relu(self.bn4(self.conv4(x)))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # First FC
        x = F.relu(self.bn_dense1(self.fc1(x)))
        x = self.dropout(x)

        # Second FC
        x = F.relu(self.bn_dense2(self.fc2(x)))
        x = self.dropout(x)

        x = self.out(x)
        return x



class Classifier(pl.LightningModule):
    """
    A PyTorch Lightning module wrapping a variety of pretrained vision backbones
    (ResNet, ConvNeXt, EfficientNet, Swin, RegNet, MobileNet, ShuffleNet, SqueezeNet,
    DenseNet, MNASNet) or a custom CNN for tile-level image classification.

    Supports polarization-derived input channels (AoLP, DoLP, Unpolarized, Max/Min/Range
    Polarization) computed on-the-fly from the base 4 Stokes channels. Also supports
    input noise augmentation and false-clean penalization for imbalanced datasets.
    """
    def __init__(self,
                      model='resnet18',
                      loss='focal',
                      tile_size=64,
                      num_classes=4,
                      dropout_rate=0.1,
                      lr=1e-4,
                      AoLP=False,
                      DoLP=False,
                      Unpolarized=False,
                      MaxPolarization=False,
                      MinPolarization=False,
                      RangePolarization=False,
                      load_checkpoint=None,
                      penalize_false_clean=0.0,
                      base_channels=32,
                      final_dense_layer=512,
                      clean_class=0,
                      noise_std=0.0,
                      noise_clip=None,
                      gain_jitter=0.0,
                      polar_flip=False,
                      channel_jitter=0.0,
                      monochrome=False,
                      polar_rot=False,
                      frozen_body_start_epochs=0,
                      frozen_body_end_epochs=0,
                      custom_early_convs=0,
                      custom_channels=None,
                      custom_res_blocks=None,
                      custom_wavelet_pools=None,
                      pretrained=True
                 ):
        super(Classifier, self).__init__()
        #-----------------------------------------
        self.type              = model
        self.lr                = lr
        self.tile_size         = tile_size
        self.num_classes       = num_classes
        self.dropout_rate      = dropout_rate
        self.base_channels     = base_channels
        self.final_dense_layer = final_dense_layer
        #-----------------------------------------
        self.AoLP = AoLP
        self.DoLP = DoLP
        self.Unpolarized = Unpolarized
        self.MaxPolarization   = MaxPolarization
        self.MinPolarization   = MinPolarization
        self.RangePolarization = RangePolarization
        #-----------------------------------------
        self.clean_class  = clean_class
        self.penalize_false_clean = penalize_false_clean
        #-----------------------------------------
        self.noise_std  = noise_std
        self.noise_clip = noise_clip
        self.gain_jitter = gain_jitter   # exposure-emulating multiplicative jitter
        self.polar_flip  = polar_flip    # physics-consistent mirror augmentation
        self.channel_jitter = channel_jitter  # strobe-light-emulating per-channel jitter
        self.polar_rot = polar_rot       # physics-consistent +/-90deg rotations (0<->90, 45<->135 swap)
        # Freeze the pretrained backbone body for the first / last N epochs so a
        # freshly-initialized stem or head can align before the body co-adapts
        # (and for a clean head-only fine-tune at the end). 0 = never freeze.
        self.custom_early_convs = int(custom_early_convs)
        self.custom_channels = custom_channels
        self.custom_res_blocks = custom_res_blocks
        self.custom_wavelet_pools = custom_wavelet_pools
        self.pretrained = bool(pretrained)
        self.frozen_body_start_epochs = int(frozen_body_start_epochs)
        self.frozen_body_end_epochs   = int(frozen_body_end_epochs)
        self._body_frozen_state = None   # cache to avoid redundant requires_grad churn
        # Polarization ablation: replace the 4 channels with their mean (what a
        # regular monochrome camera would record), replicated x4 so the
        # architecture/capacity stay byte-identical. Train AND val see it.
        self.monochrome = monochrome

        # Dynamic input channels (base 4 polarization channels + optional derived channels)
        extra_channels = 0
        if self.DoLP: extra_channels += 1
        if self.AoLP: extra_channels += 1
        if self.Unpolarized: extra_channels += 1
        if self.MaxPolarization: extra_channels += 1
        if self.MinPolarization: extra_channels += 1
        if self.RangePolarization: extra_channels += 1

        self.base_input_channels = 4
        self.in_channels = self.base_input_channels + extra_channels

        #RESNEXT
        if self.type == 'resnext50':
            self.model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(2048, num_classes)
        elif self.type == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT if getattr(self, 'pretrained', True) else None)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512, num_classes)
        elif self.type == 'resnet18_stem':
            # Deeper stem, same downsampling: more early capacity for the small
            # (~3x3) polarization micro-features the stock 7x7/2 conv blurs away.
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.model.fc = nn.Linear(512, num_classes)
        elif self.type == 'resnet18_fullres':
            # Maximum feature preservation for ~3px (300um) defects: the stem does
            # ZERO spatial downsampling. Three 3x3 stride-1 convs (RF 7, ch
            # 4->32->48->64) extract the feature at full 48x48 resolution and the
            # resnet body's own maxpool is removed, so layer1 runs at 48x48 and the
            # first downsample is layer2 (stride 2). Final map 6x6 (vs 1.5x1.5 stock).
            # ~16x the layer1 FLOPs of stock resnet18 — acceptable: inference cost is
            # traded back at deploy time via a larger tile step. Best fidelity for recall.
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
            self.model.maxpool = nn.Identity()  # NO downsampling before the residual body
            self.model.fc = nn.Linear(512, num_classes)
        elif self.type == 'resnet18_hires':
            # Feature-preserving stem for ~3px (300um) defects. Nyquist: a 3px
            # feature tolerates <1.5x downsampling BEFORE it is extracted, so the
            # stem runs THREE 3x3 stride-1 convs at full tile resolution
            # (receptive field 7, channels 4->32->48->64) to encode the feature
            # into the channel dimension, THEN a single maxpool downsamples the
            # now-redundant spatial grid. Contrast: stock/stem strides 2 on the
            # first conv (3px -> 0.75px at layer1, unrecoverable).
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),   # only downsample: 48->24
            )
            self.model.maxpool = nn.Identity()  # resnet's own maxpool folded into the stem above
            self.model.fc = nn.Linear(512, num_classes)
        elif self.type == 'resnet18_fine':
            # Deeper stem AND stride 1: layer1 runs at 24x24 instead of 12x12 for
            # 48px tiles (only the maxpool downsamples). ~4x compute of resnet18.
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.model.fc = nn.Linear(512, num_classes)
        elif self.type == 'convnext_tiny':
            self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 96, kernel_size=(4, 4), stride=(4, 4))
            self.model.classifier[2]  = nn.Linear(768, num_classes)
        elif self.type == 'efficientnet_v2_s':
            self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1]  = nn.Linear(1280, num_classes, bias=True)
        elif self.type == 'swin_v2_t':
            self.model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 96, kernel_size=(4, 4), stride=(4, 4))
            self.model.head = nn.Linear(768, num_classes)
        elif self.type == 'regnet_y_800mf':
            self.model = regnet_y_800mf(weights=RegNet_Y_800MF_Weights.DEFAULT)
            self.model.stem[0] = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.fc = nn.Linear(784, num_classes)
        elif self.type == 'regnet_y_400mf':
            self.model = torchvision.models.regnet_y_400mf(weights=RegNet_Y_400MF_Weights.DEFAULT)
            self.model.stem[0] = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.fc = nn.Linear(440, num_classes)
        elif self.type == 'mobilenet_v3_small':
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[3] = nn.Linear(1024, num_classes)
        elif self.type == 'mobilenet_v3_large':
            self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[3] = nn.Linear(1280, num_classes)
        elif self.type == 'shufflenet_v2_x0_5':
            self.model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
            self.model.conv1[0] = nn.Conv2d(self.in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.fc = nn.Linear(1024, num_classes)
        elif self.type == 'shufflenet_v2_x1_0':
            self.model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
            self.model.conv1[0] = nn.Conv2d(self.in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.fc = nn.Linear(1024, num_classes)
        elif self.type == 'squeezenet1_1':
            self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
            self.model.features[0] = nn.Conv2d(self.in_channels, 64, kernel_size=(3, 3), stride=(2, 2))
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
            self.model.num_classes = num_classes
        elif self.type == 'efficientnet_b0':
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif self.type == 'densenet121':
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            self.model.features.conv0 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier = nn.Linear(1024, num_classes)
        elif self.type == 'mnasnet0_5':
            self.model = mnasnet0_5(weights=MNASNet0_5_Weights.DEFAULT)
            self.model.layers[0] = nn.Conv2d(self.in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif self.type == 'mnasnet1_0':
            self.model = mnasnet1_0(weights=MNASNet1_0_Weights.DEFAULT)
            self.model.layers[0] = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif ('custom' in self.type) or ('cnn' in self.type):
            self.model = CustomCNN(
                                   in_channels=self.in_channels,
                                   intended_tile_size=tile_size,
                                   num_classes=num_classes,
                                   dropout_rate=dropout_rate,
                                   base_channels=self.base_channels,
                                   final_dense_layer=self.final_dense_layer,
                                   early_convs=getattr(self, 'custom_early_convs', 0),
                                   channels=getattr(self, 'custom_channels', None),
                                   res_blocks=getattr(self, 'custom_res_blocks', None),
                                   wavelet_pools=getattr(self, 'custom_wavelet_pools', None)
                                  )
        else:
            raise ValueError(f"Unsupported model type: {model}. Supported types are 'resnext50', 'resnet18', 'convnext_tiny', 'efficientnet_v2_s', 'swin_v2_t', 'regnet_y_800mf'.")

        if load_checkpoint is not None:
            # load_from_checkpoint returns a full Classifier (LightningModule), not a bare
            # backbone. Copy only the backbone weights so self.model keeps the correct type.
            loaded = Classifier.load_from_checkpoint(load_checkpoint)
            self.model.load_state_dict(loaded.model.state_dict())

        if loss == 'focal':
            self.criterion = CategoricalFocalLoss(gamma=2.0, alpha=None)
        elif loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss}. Supported types are 'focal' and 'cross_entropy'.")

        self.accuracy  = Accuracy(task='MULTICLASS',  num_classes=num_classes)
        self.recall    = Recall(task='MULTICLASS',    num_classes=num_classes)
        self.precision = Precision(task='MULTICLASS', num_classes=num_classes)
        self.auroc     = AUROC(task='MULTICLASS',     num_classes=num_classes)
        # Binary defect-vs-clean detector AUROC — the metric the KPI cares about,
        # and the one ModelCheckpoint should select on (see validation_step).
        self.detect_auroc = AUROC(task='BINARY')

    def add_input_noise(self, x):
        """
        Add Gaussian noise to the input during training for regularization.
        Only active when self.training is True and noise_std > 0.

        Args:
            x: Input tensor.

        Returns:
            Input tensor with noise added.
        """
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(x) * self.noise_std
            if self.noise_clip is not None:
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            x = x + noise
        return x

    def augment_train_batch(self, x):
        """
        Training-only augmentation on the raw 4-channel polarization batch.

        gain_jitter: per-sample multiplicative gain, log-uniform in
        [1/(1+j), 1+j], identical on all 4 channels — emulates the exposure
        differences between recording sessions (dataset names carry exposures
        150..4500) without breaking polarization ratios.

        polar_flip: random horizontal/vertical mirror. A mirror maps the angle
        of linear polarization theta -> -theta, i.e. S2 = I45-I135 negates, so
        the 45deg and 135deg channels (1 and 3) MUST be swapped along with the
        pixel flip. Verified empirically on the static-camera
        measure65mmheight_* sets: corr(S2, mirror(S2)) = -0.86..-0.92 while the
        invariant S1 control stays positive.

        Returns float32 in [0,1] (dequantizes uint8 first so downstream
        build_input_features skips its own dequantization consistently).
        """
        if not self.training or (self.gain_jitter <= 0.0 and not self.polar_flip
                                 and self.channel_jitter <= 0.0 and not self.polar_rot):
            return x
        if x.dtype == torch.uint8:
            x = x.float() * (1.0 / 255.0)

        if self.polar_flip:
            swap = [0, 3, 2, 1]  # 45deg <-> 135deg
            for dim in (-1, -2):  # horizontal, then vertical mirror
                sel = torch.rand(x.shape[0], device=x.device) < 0.5
                if sel.any():
                    x[sel] = x[sel].flip(dim)[:, swap, :, :]

        if self.polar_rot:
            # +/-90deg rotations complete the dihedral group the mirrors start
            # (180deg = H+V mirror is already covered). Stokes rotation by 90deg
            # negates S1 and S2 -> swap I0<->I90 AND I45<->I135. In the H5/model
            # channel order [p90, p45, p0, p135] that is the permutation [2,3,0,1].
            rot_swap = [2, 3, 0, 1]
            for k in (1, 3):  # 90 and 270 degrees, each on a random 1/4 of the batch
                sel = torch.rand(x.shape[0], device=x.device) < 0.25
                if sel.any():
                    x[sel] = torch.rot90(x[sel], k, dims=(-2, -1))[:, rot_swap, :, :]

        if self.gain_jitter > 0.0:
            j = float(self.gain_jitter)
            lo = torch.log(torch.tensor(1.0 / (1.0 + j)))
            hi = torch.log(torch.tensor(1.0 + j))
            g = torch.exp(torch.empty(x.shape[0], 1, 1, 1, device=x.device)
                          .uniform_(float(lo), float(hi)))
            x = torch.clamp(x * g, 0.0, 1.0)

        if self.channel_jitter > 0.0:
            # INDEPENDENT per-channel gains: emulates the 6 strobed scene lights,
            # whose changes shift the polarization channel proportions (measured
            # signature swing between lights: L1 up to ~0.30 on the normalized
            # 4-vector, i.e. per-channel relative changes up to ~+/-40%). This is
            # the augmentation counterpart of the annotator's canonical-light
            # remap: instead of normalizing lights away at inference, train the
            # model to be invariant to them.
            j = float(self.channel_jitter)
            lo = torch.log(torch.tensor(1.0 / (1.0 + j)))
            hi = torch.log(torch.tensor(1.0 + j))
            g = torch.exp(torch.empty(x.shape[0], x.shape[1], 1, 1, device=x.device)
                          .uniform_(float(lo), float(hi)))
            x = torch.clamp(x * g, 0.0, 1.0)

        return x

    def build_input_features(self, x):
        """
        Build model input by appending derived channels.
        All polarization-derived features are computed from the original 4 channels only.
        Expects x shape: [B, >=4, H, W] at input (normally [B,4,H,W]).
        Returns x shape: [B, self.in_channels, H, W]

        Accepts uint8 input (0-255) and normalises to float32 [0,1] on the GPU.
        This keeps PCIe transfer bandwidth 4× lower than sending float32 tensors.
        """
        # Dequantize uint8 → float32 on the device where x already lives.
        # The multiplication is a single fused kernel; cost is negligible vs. the
        # conv ops that follow.  float() preserves the current device and layout.
        if x.dtype == torch.uint8:
            x = x.float() * (1.0 / 255.0)

        if x.shape[1] < 4:
            raise ValueError(f"Expected at least 4 channels for polarization input, got {x.shape[1]}")

        if self.monochrome:
            # simulate a regular monochrome camera: intensity only, no polarization
            x = x[:, 0:4, :, :].mean(dim=1, keepdim=True).expand(-1, 4, -1, -1).contiguous()

        pol = x[:, 0:4, :, :]  # original polarization channels only

        # AoLP / DoLP from Stokes computed on the original channels
        if self.AoLP or self.DoLP:
            stokes = self.calculate_stokes(pol)

            if self.DoLP:
                dolp = self.calculate_DoLP(stokes).unsqueeze(1)  # [B,1,H,W]
                x = torch.cat((x, dolp), dim=1)

            if self.AoLP:
                aolp = self.calculate_AoLP(stokes).unsqueeze(1)  # [B,1,H,W]
                x = torch.cat((x, aolp), dim=1)

        # Unpolarized = mean over original 4 channels
        if self.Unpolarized:
            mon = pol.mean(dim=1, keepdim=True)
            x = torch.cat((x, mon), dim=1)

        # Max / Min / Range over original 4 channels
        if self.MaxPolarization:
            max_pol = pol.max(dim=1, keepdim=True)[0]
            x = torch.cat((x, max_pol), dim=1)

        if self.MinPolarization:
            min_pol = pol.min(dim=1, keepdim=True)[0]
            x = torch.cat((x, min_pol), dim=1)

        if self.RangePolarization:
            max_pol = pol.max(dim=1, keepdim=True)[0]
            min_pol = pol.min(dim=1, keepdim=True)[0]
            range_pol = max_pol - min_pol
            x = torch.cat((x, range_pol), dim=1)

        # Final sanity: ensure model sees the expected channel count
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Feature builder produced {x.shape[1]} channels, expected {self.in_channels}. "
                             f"(Flags: DoLP={self.DoLP}, AoLP={self.AoLP}, Unpolarized={self.Unpolarized}, "
                             f"MaxPolarization={self.MaxPolarization}, MinPolarization={self.MinPolarization}, "
                             f"RangePolarization={self.RangePolarization})")
        return x

    def forward(self, x):
        """
        Canonical forward pass: build input features (polarization channels + noise),
        then pass through the backbone model.

        Args:
            x: Input tensor, typically uint8 [B, 4, H, W] or float [B, 4, H, W].

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.build_input_features(x)
        return self.model(x)

    def training_step(self, batch, _batch_idx):
        """
        Compute training loss with optional false-clean penalization.

        If penalize_false_clean > 0, adds a penalty term that discourages the model
        from assigning high clean-class probability to non-clean samples. The penalty
        is -log(1 - P(clean)) for each non-clean sample.

        Args:
            batch: (x, y) tensor pair from the DataLoader.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Scalar loss tensor.
        """
        x, y = batch

        x = self.augment_train_batch(x)
        x = self.add_input_noise(x)
        x = self.build_input_features(x)

        y_hat = self.model(x)
        base_loss  = self.criterion(y_hat, y)

        if self.penalize_false_clean > 0.0:
            pred_probs = F.softmax(y_hat, dim=1)
            non_clean_mask = (y != self.clean_class)
            penalty_strength = float(self.penalize_false_clean)

            if non_clean_mask.any():
                p_clean = pred_probs[non_clean_mask, self.clean_class]
                false_clean_loss = -torch.log(1.0 - p_clean + 1e-8).mean()
                loss = base_loss + penalty_strength * false_clean_loss
            else:
                loss = base_loss
        else:
            loss = base_loss

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        """
        Execute one validation batch: compute loss and update metric accumulators.
        Noise is intentionally skipped (training-only). Metric .compute() is deferred
        to on_validation_epoch_end to avoid per-batch warnings on small/imbalanced batches.

        Args:
            batch: (x, y) tensor pair from the validation DataLoader.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Scalar validation loss tensor.
        """
        x, y = batch

        # Use self(x) — goes through the canonical forward() path.
        # Note: add_input_noise is intentionally skipped here (noise is training-only).
        y_hat = self(x)
        loss  = self.criterion(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)

        # Only accumulate state per batch — do NOT call .compute() here.
        # Computing per-batch fires "no positive samples" warnings whenever a
        # small batch happens to contain only one class (common with imbalanced
        # datasets).  Epoch-level values are logged in on_validation_epoch_end.
        self.accuracy.update(y_hat, y)
        self.recall.update(y_hat, y)
        self.precision.update(y_hat, y)
        self.auroc.update(y_hat, y)

        # Detection AUROC: is this tile ANY defect, vs clean -- the question the
        # KPI (skipped defects) actually asks. val_loss does NOT answer it:
        # focal+penalize_false_clean scores 6-way class identity and calibration,
        # and measured over epochs it is UNCORRELATED with detection
        # (pearson(val_loss, detect AUROC) = -0.09 over the 24-epoch customwide
        # run), so monitoring val_loss selects a checkpoint at random w.r.t. what
        # we care about. Same for val_auroc above: that is macro one-vs-rest over
        # all classes, not defect-vs-clean. Score by 1 - P(clean), matching
        # liveClassifierTorch.gate_tiles' defect_mass gate.
        if self.clean_class is not None:
            probs = F.softmax(y_hat, dim=1)
            self.detect_auroc.update(1.0 - probs[:, self.clean_class],
                                     (y != self.clean_class).long())

        return loss

    def calculate_stokes(self, x):
        """
        Compute the 4 Stokes parameters (S0, S1, S2, S3) from the 4-channel
        polarization input. The input channels are assumed to be:
          [I_total, 0°_linear, 45°_linear, Right_circular]

        Stokes formulas:
          S0 = I_total
          S1 = I(0°) - I(90°)   → x[:,1] - x[:,2]
          S2 = I(45°) - I(135°) → x[:,1] + x[:,2]
          S3 = 2*I(right_circ) - I_total

        Args:
            x: Tensor of shape (batch_size, 4, height, width).

        Returns:
            Stokes tensor of shape (batch_size, 4, height, width).
        """
        S0 = x[:, 0, :, :]
        S1 = x[:, 1, :, :] - x[:, 2, :, :]
        S2 = x[:, 1, :, :] + x[:, 2, :, :]
        S3 = x[:, 3, :, :]
        return torch.stack((S0, S1, S2, S3), dim=1)

    def calculate_DoLP(self, x):
        """
        Compute the Degree of Linear Polarization from Stokes parameters.
        DoLP = sqrt(S1^2 + S2^2) / S0, representing the fraction of light that
        is linearly polarized. Value range: [0, 1].

        Args:
            x: Stokes tensor of shape (batch_size, 4, height, width).

        Returns:
            DoLP tensor of shape (batch_size, height, width).
        """
        S0 = x[:, 0, :, :]
        S1 = x[:, 1, :, :]
        S2 = x[:, 2, :, :]
        #S3 = x[:, 3, :, :]  # Not used in DoLP
        DoLP = torch.sqrt(S1**2 + S2**2) / (S0 + 1e-8)
        return DoLP

    def calculate_AoLP(self, x):
        """
        Compute the Angle of Linear Polarization from Stokes parameters.
        AoLP = 0.5 * atan2(S2, S1), representing the orientation angle of the
        electric field oscillation. Value range: [-pi/4, pi/4].

        Args:
            x: Stokes tensor of shape (batch_size, 4, height, width).

        Returns:
            AoLP tensor of shape (batch_size, height, width).
        """
        S1 = x[:, 1, :, :]
        S2 = x[:, 2, :, :]
        AoLP = 0.5 * torch.atan2(S2, S1)
        return AoLP

    def on_validation_epoch_end(self):
        # Compute epoch-level metrics from the accumulated state of all validation
        # batches, then reset for the next epoch.  Computing here (not per-batch)
        # avoids "no positive samples" warnings that fire when individual batches
        # happen to contain only one class.
        self.log('val_accuracy',  self.accuracy.compute(),  prog_bar=True, sync_dist=True)
        self.log('val_recall',    self.recall.compute(),    prog_bar=True, sync_dist=True)
        self.log('val_precision', self.precision.compute(), prog_bar=True, sync_dist=True)
        self.log('val_auroc',     self.auroc.compute(),     prog_bar=True, sync_dist=True)
        if self.clean_class is not None:
            # The checkpoint selector should monitor THIS (mode='max'), not val_loss.
            self.log('val_detect_auroc', self.detect_auroc.compute(), prog_bar=True, sync_dist=True)
            self.detect_auroc.reset()
        self.accuracy.reset()
        self.recall.reset()
        self.precision.reset()
        self.auroc.reset()

    # Parameter-name fragments identifying the NON-body (input adapter / stem /
    # classifier head) parameters — the ones we replaced and that should keep
    # training while the pretrained body is frozen. Everything else is "body".
    _NONBODY_KEYS = ('conv1', 'bn1', 'stem', 'fc', 'classifier', 'head', 'features.0')

    def _is_body_param(self, name):
        return not any(k in name for k in self._NONBODY_KEYS)

    def _set_body_frozen(self, frozen):
        if self._body_frozen_state is frozen:
            return
        n = 0
        for name, p in self.model.named_parameters():
            if self._is_body_param(name):
                p.requires_grad = not frozen
                n += 1
        # also stop BN running-stat drift in the frozen body
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and self._is_body_param(name + '.weight'):
                m.eval() if frozen else m.train()
        self._body_frozen_state = frozen
        print(f"[FreezeBody] body {'FROZEN' if frozen else 'UNFROZEN'} ({n} body param tensors)")

    def on_train_epoch_start(self):
        s, e = self.frozen_body_start_epochs, self.frozen_body_end_epochs
        if s == 0 and e == 0:
            return
        ep = self.current_epoch
        total = getattr(self.trainer, 'max_epochs', None) or (ep + 1)
        freeze = (ep < s) or (ep >= total - e)
        self._set_body_frozen(freeze)

    def configure_optimizers(self):
        """
        Configure the AdamW optimizer for all parameters in the Classifier module.
        Using self.parameters() (not self.model.parameters()) ensures that any new
        learnable parameters added directly to Classifier (not the backbone) are included.
        """
        # Optimize self.parameters() (all module params) rather than just self.model.parameters(),
        # so any future additions to Classifier itself are covered automatically.
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def load_png_comment_metadata(image_path):
    """
    Read JSON metadata stored in PNG text/comment fields.

    Returns:
        dict: parsed metadata dictionary, or {} if unavailable / invalid.
    """
    try:
        from PIL import Image
        import json

        with Image.open(image_path) as img:
            candidates = []

            # Classic Pillow info dict
            if hasattr(img, "info") and isinstance(img.info, dict):
                for key in ("comment", "Comment", "description", "Description"):
                    if key in img.info and img.info[key] is not None:
                        candidates.append(img.info[key])

            # PNG text chunks
            if hasattr(img, "text") and isinstance(img.text, dict):
                for key in ("comment", "Comment", "description", "Description"):
                    if key in img.text and img.text[key] is not None:
                        candidates.append(img.text[key])

                # Also try every text chunk, in case the metadata was stored under another key
                for key, value in img.text.items():
                    if value is not None:
                        candidates.append(value)

            # Deduplicate while preserving order
            seen = set()
            unique_candidates = []
            for c in candidates:
                if isinstance(c, bytes):
                    c = c.decode("utf-8", errors="ignore")
                elif not isinstance(c, str):
                    c = str(c)

                c = c.strip()
                if c and c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)

            # Try to parse any candidate as JSON
            for c in unique_candidates:
                try:
                    parsed = json.loads(c)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

            return {}

    except Exception:
        return {}

def metadata_collate_fn(batch):
    """
    Collate function for DataLoader that handles batches with or without metadata.
    Stacks image tensors, converts labels to long, and collects metadata dicts.

    Args:
        batch: List of (x, y) or (x, y, metadata) tuples from RGBAImageFolder.

    Returns:
        (xs, ys, metas) where xs is stacked tensor, ys is label tensor,
        and metas is a list of metadata dicts.
    """
    xs = []
    ys = []
    metas = []

    for item in batch:
        if len(item) == 3:
            x, y, meta = item
        else:
            x, y = item
            meta = {}

        xs.append(x)
        ys.append(y)
        metas.append(meta)

    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys, metas

def load_rgba_image_pil(path):
    """
    Open a PNG/JPG image with PIL and convert it to RGBA mode.

    Args:
        path: File path to the image.

    Returns:
        A PIL Image in RGBA mode.

    Raises:
        ValueError: If conversion to RGBA fails.
    """
    with Image.open(path) as img:
        try:
            img = img.convert('RGBA')  # Convert to RGBA
        except Exception as e:
            raise ValueError(f"Error converting image {path} to RGBA: {e}")
        return img



def load_rgba_image(image_path):
    """
    Load an image using OpenCV as RGBA uint8 in HWC order.
    OpenCV loads in BGRA order, so we convert to BGRA to preserve correct channel
    semantics (the rest of the pipeline expects standard RGBA ordering).

    The image stays as uint8 to minimize PCIe transfer bandwidth — normalization
    to float32 [0, 1] happens inside Classifier.build_input_features() on GPU.

    Args:
        image_path: Path to the image file.

    Returns:
        Numpy array of shape (H, W, 4) with dtype uint8, values 0-255.
    """
    import cv2
    rgba_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)  # undo OpenCV channel ordering
    # Keep as uint8 — normalization (/255) happens inside the model on GPU to
    # reduce CPU→GPU transfer bandwidth by 4× (1 byte vs 4 bytes per pixel).
    return rgba_image  # HWC uint8, values 0–255


class RGBAImageFolder(datasets.DatasetFolder):
    """
    ImageFolder-style dataset that loads RGBA images via OpenCV.
    Supports optional PNG comment/metadata extraction from embedded text chunks.
    """
    def __init__(self, root, transform=None, return_metadata=False):
        """
        Args:
            root: Root directory with class subdirectories.
            transform: Optional transform to apply to loaded images (e.g. tensor conversion).
            return_metadata: If True, also load PNG comment metadata and return (x, y, meta).
        """
        super(RGBAImageFolder, self).__init__(
            root,
            loader=load_rgba_image,
            extensions=('png', 'jpg', 'jpeg'),
            transform=transform
        )
        self.return_metadata = return_metadata

    def __getitem__(self, index):
        """
        Load a single image, apply transforms, and optionally extract metadata.

        Args:
            index: Sample index.

        Returns:
            (sample, target) or (sample, target, metadata) if return_metadata is True.
        """
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_metadata:
            metadata = load_png_comment_metadata(path)
            return sample, target, metadata

        return sample, target


def _human_bytes(num_bytes: int) -> str:
    """Return a human-readable representation of a byte count (e.g. '1.23 GiB')."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    n = float(num_bytes)
    for u in units:
        if n < 1024.0 or u == units[-1]:
            return f"{n:.2f} {u}"
        n /= 1024.0


def _get_available_ram_bytes() -> int:
    """
    Get the amount of available system RAM in bytes.
    Tries psutil first, then falls back to /proc/meminfo on Linux.

    Returns:
        Available RAM in bytes, or 0 if detection fails.
    """
    # Prefer psutil if installed
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    # Fallback: Linux /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    # kB -> bytes
                    return int(parts[1]) * 1024
    except Exception:
        pass

    return 0


def _get_path_size_bytes(path: str) -> int:
    """
    Calculate the total size in bytes of all files under a path (directory or file).
    Skips files that raise OSError on access (permission issues, etc.).

    Args:
        path: File or directory path.

    Returns:
        Total size in bytes, or 0 if calculation fails.
    """
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, _, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total
    except Exception:
        return 0


def _estimate_dataset_ram_bytes(dataset: Dataset, sample_count: int = 32) -> int:
    """Estimate RAM footprint (bytes) of caching dataset[i] objects.

    We sample a few items, compute their tensor payload sizes, and extrapolate.
    This is only a heuristic but is much better than relying on on-disk size.
    """
    n = len(dataset)
    if n == 0:
        return 0

    k = max(1, min(sample_count, n))
    # Spread samples across the dataset to avoid bias
    indices = np.linspace(0, n - 1, num=k, dtype=int)

    total_bytes = 0
    for idx in indices:
        item = dataset[int(idx)]
        x = item[0]  # always the tensor; ignore label and optional metadata tuple
        item_bytes = 0
        # x may be a Tensor, numpy array, PIL image, or a tuple/list thereof
        def _payload_bytes(obj):
            if torch.is_tensor(obj):
                return int(obj.element_size() * obj.numel())
            if isinstance(obj, np.ndarray):
                return int(obj.nbytes)
            if isinstance(obj, Image.Image):
                # Approximate: width*height*channels*1 byte (before ToTensor). Conservative.
                bands = len(obj.getbands()) if hasattr(obj, "getbands") else 3
                return int(obj.size[0] * obj.size[1] * bands)
            if isinstance(obj, (list, tuple)):
                return sum(_payload_bytes(o) for o in obj)
            if isinstance(obj, dict):
                return sum(_payload_bytes(v) for v in obj.values())
            return 0

        item_bytes += _payload_bytes(x)
        # label bytes are negligible, but include a small constant overhead
        item_bytes += 64
        total_bytes += item_bytes

    avg = total_bytes / float(k)
    # Add overhead multiplier for Python objects / list storage
    overhead_multiplier = 1.25
    return int(avg * n * overhead_multiplier)

class RAMPreloadedDataset(Dataset):
    """
    Preloads an entire dataset into RAM (samples are cached as returned by the wrapped dataset).

    Use this to eliminate disk I/O bottlenecks during training at the cost of RAM.
    Ideal when the dataset fits comfortably in memory and training runs many epochs.

    Notes:
    - Increases startup time and RAM usage proportionally to dataset size.
    - With DataLoader(num_workers>0), each worker may hold its own copy depending on
      the multiprocessing start method. For true single-copy behavior, use num_workers=0.
    """
    def __init__(self, dataset, show_progress=True):
        """
        Args:
            dataset: The wrapped dataset to preload.
            show_progress: If True, show a tqdm progress bar during caching.
        """
        super().__init__()
        self._dataset = dataset
        self.classes = getattr(dataset, 'classes', None)
        self.targets = getattr(dataset, 'targets', None)

        n = len(dataset)
        print("\n[WARNING] cacheAllDataToRAM=True -> Preloading ALL dataset samples into RAM...")
        print("          This will take some time up front, but training batches will not hit the HDD.")
        print(f"          Samples to cache: {n}\n")

        self._cached = []

        iterator = range(n)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Caching dataset to RAM", unit="sample")
            except Exception:
                pass

        for i in iterator:
            self._cached.append(dataset[i])

        # If wrapped dataset doesn't expose targets, infer them from cached samples
        if self.targets is None:
            self.targets = [y for (_, y) in self._cached]

        print("[OK] Dataset cached to RAM. Starting training...\n")

    def __len__(self):
        """Return the number of cached samples."""
        return len(self._cached)

    def __getitem__(self, idx):
        """Return a cached sample by index."""
        return self._cached[idx]


class CombinedDataset(Dataset):
    """
    Concatenates multiple datasets into a single Dataset, exposing:
      - classes / class_to_idx (must be consistent across all sub-datasets)
      - targets (concatenated from all sub-datasets)

    Indexing is O(number of sub-datasets) via a linear scan through cumulative offsets.
    This is efficient for typical use cases (a handful of datasets).
    """
    def __init__(self, datasets):
        """
        Args:
            datasets: List of dataset objects, all sharing the same classes/class_to_idx.

        Raises:
            ValueError: If no datasets provided or class mappings are inconsistent.
        """
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("CombinedDataset received 0 datasets")

        # Verify consistent class mapping
        base_classes = getattr(datasets[0], "classes", None)
        base_cti     = getattr(datasets[0], "class_to_idx", None)
        if base_classes is None or base_cti is None:
            raise ValueError("Sub-dataset does not expose classes/class_to_idx")

        for i, ds in enumerate(datasets[1:], start=1):
            if getattr(ds, "classes", None) != base_classes:
                raise ValueError(f"Dataset #{i} has different classes. "
                                 f"Expected {base_classes}, got {getattr(ds,'classes',None)}")
            if getattr(ds, "class_to_idx", None) != base_cti:
                raise ValueError(f"Dataset #{i} has different class_to_idx mapping.")

        self.datasets = datasets
        self.classes = base_classes
        self.class_to_idx = base_cti

        # Build cumulative sizes for fast indexing
        self._lengths = [len(ds) for ds in datasets]
        self._offsets = []
        s = 0
        for L in self._lengths:
            self._offsets.append(s)
            s += L
        self._total_len = s

        # Concatenate targets (so your class distribution + weights still work)
        self.targets = []
        for ds in datasets:
            t = getattr(ds, "targets", None)
            if t is None:
                # Fallback if dataset doesn't expose .targets
                self.targets.extend([ds[i][1] for i in range(len(ds))])
            else:
                self.targets.extend(list(t))

    def __len__(self):
        """Return the total number of samples across all sub-datasets."""
        return self._total_len

    def __getitem__(self, idx):
        """
        Forward an index to the correct sub-dataset and return the sample.

        Args:
            idx: Global index into the combined dataset.

        Returns:
            The sample tuple from the appropriate sub-dataset.

        Raises:
            IndexError: If idx is out of bounds.
            RuntimeError: If no sub-dataset contains the index (shouldn't happen).
        """
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)

        # Find which dataset this index belongs to
        # (linear scan is fine for small number of datasets; can binary search if needed)
        for ds_i in range(len(self.datasets)-1, -1, -1):
            if idx >= self._offsets[ds_i]:
                local_idx = idx - self._offsets[ds_i]
                return self.datasets[ds_i][local_idx]

        raise RuntimeError("CombinedDataset indexing error")

class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Yields batches where every class is represented at least once.

    Mechanism:
      slots_per_class = batch_size // num_classes   (guaranteed per-class slots)
      extra_slots     = batch_size % num_classes     (filled randomly from all indices)

    One epoch produces ceil(largest_class_size / slots_per_class) batches.
    The majority class is seen roughly once per epoch; minority classes are
    oversampled proportionally. This prevents any batch from being entirely
    dominated by the clean/majority class.

    Usage:
        Pass as `batch_sampler=` to DataLoader (do NOT set shuffle=True).
    """
    def __init__(self, targets, num_classes: int, batch_size: int):
        """
        Args:
            targets: List of integer class labels for all samples.
            num_classes: Total number of distinct classes.
            batch_size: Desired batch size (must be >= num_classes).

        Raises:
            ValueError: If batch_size < num_classes or any class has zero samples.
        """
        if batch_size < num_classes:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_classes ({num_classes}) "
                f"for balanced sampling."
            )
        self.num_classes    = num_classes
        self.batch_size     = batch_size
        self.slots_per_class = batch_size // num_classes
        self.extra_slots    = batch_size % num_classes

        targets_np = np.asarray(targets)
        self.class_indices = [
            np.where(targets_np == c)[0].tolist()
            for c in range(num_classes)
        ]

        empty = [c for c, ci in enumerate(self.class_indices) if len(ci) == 0]
        if empty:
            raise ValueError(f"Classes {empty} have zero samples — cannot balance.")

        self._len = max(len(ci) for ci in self.class_indices) // self.slots_per_class

    def __len__(self):
        """Return the number of batches per epoch."""
        return self._len

    def __iter__(self):
        """
        Generate one epoch of balanced batches.

        Each class index pool is shuffled at the start of every epoch. When a pool
        is exhausted, it is re-shuffled and replayed from the beginning. Extra slots
        (batch_size % num_classes) are filled by random sampling from all indices.
        Each batch is finally shuffled to avoid class-order bias.
        """
        pools = [random.sample(ci, len(ci)) for ci in self.class_indices]
        ptrs  = [0] * self.num_classes
        all_flat = [idx for ci in self.class_indices for idx in ci]

        for _ in range(self._len):
            batch = []

            for c in range(self.num_classes):
                for _ in range(self.slots_per_class):
                    if ptrs[c] >= len(pools[c]):
                        pools[c] = random.sample(self.class_indices[c], len(self.class_indices[c]))
                        ptrs[c]  = 0
                    batch.append(pools[c][ptrs[c]])
                    ptrs[c] += 1

            if self.extra_slots:
                batch.extend(random.sample(all_flat, self.extra_slots))

            random.shuffle(batch)
            yield batch


def print_class_distribution(dataset, title="Dataset"):
    """
    Prints number of samples per class.

    Works with:
    - RGBAImageFolder
    - HDF5Dataset (if it exposes .targets)
    - random_split subsets
    """
    print(f"\n--- {title} Class Distribution ---")

    # Handle Subset (from random_split)
    if isinstance(dataset, torch.utils.data.Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
        classes = dataset.dataset.classes
    else:
        targets = dataset.targets
        classes = dataset.classes

    counter = Counter(int(t) for t in targets)

    # Iterate ALL classes (not just present ones) so zero-sample classes are
    # visible -- they crash BalancedBatchSampler and make a class unlearnable.
    total = sum(counter.values()) or 1
    for class_idx in range(len(classes)):
        count = counter.get(class_idx, 0)
        flag = "   <-- ZERO SAMPLES" if count == 0 else ""
        print(f"Class {class_idx} ({classes[class_idx]}): {count} samples "
              f"({100.0*count/total:5.1f}%){flag}")

    present = [c for c in counter.values() if c > 0]
    if present and max(present) / max(1, min(present)) > 50:
        print(f"WARNING: class imbalance {max(present)}:{min(present)} = "
              f"{max(present)/min(present):.0f}x (BalancedBatchSampler mitigates, "
              f"but rare classes stay hard cross-site)")
    print(f"Total samples: {total}")
    print("----------------------------------\n")

#Main
if __name__ == "__main__":
    # Enable TF32 for faster matmul on Ampere+ GPUs without losing meaningful precision.
    torch.set_float32_matmul_precision('high')

    configuration_file = "config.json"
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]

    print("Using ",configuration_file," configuration for training")
    config_json       = load_hyperparameters(os.path.dirname(os.path.abspath(__file__))+'/'+ configuration_file)
    print(config_json)

    if len(sys.argv) > 2:
        overwrite_model = sys.argv[2]
        config_json['model'] = overwrite_model
        print("Using model configuration provided by commandline parameter (",overwrite_model,")") 

    #Parse JSON Values
    #-----------------------------------------------------------------
    batch_size        = config_json['hparams']['batch_size']
    seed              = config_json['hparams']['seed']
    dropout_rate      = config_json['hparams']['dropout_rate']
    epochs            = config_json['hparams']['training_epochs']
    tile_size         = config_json['hparams']['tile_size']
    val_split         = config_json['dataloader']['validation_split']
    num_workers       = config_json['dataloader']['num_workers']
    # 0 in the config means "auto": use available cores, capped at 16 to avoid
    # oversubscription.  Note: RAMPreloadedDataset overrides this to 1 later to
    # prevent each worker from holding its own copy of the in-RAM dataset.
    if num_workers == 0:
        num_workers = min(os.cpu_count() or 1, 16)
    balanced_sampling = config_json['dataloader'].get('balanced_sampling', False)
    gradient_clip_val = config_json['hparams']['gradient_clip_value']
    class_weight      = config_json['class_weight']
    lr                = config_json['optimizer']['learning_rate']
    use_wandb         = config_json['wandb']['use_wandb']
    loss              = config_json['loss']
    penalize_false_clean = float(config_json['penalize_false_clean'])
    #-----------------------------------------------------------------
    base_channels     = 32
    if  'base_channels' in config_json['hparams']:
          base_channels = config_json['hparams']['base_channels']
    print("Base channels ", base_channels)
    #-----------------------------------------------------------------
    final_dense_layer=512
    if  'final_dense_layer' in config_json['hparams']:
          final_dense_layer = config_json['hparams']['final_dense_layer']
    print("Final Dense Layer ", final_dense_layer)
    #-----------------------------------------------------------------
    model_name = config_json['model']
    if "name" in config_json:
             model_name = "%s_%s" % (config_json['name'],config_json['model'])
    #-----------------------------------------------------------------
    noise_std = 0.0        
    noise_clip = None
    if  'noise_std' in config_json['hparams']:
          noise_std = config_json['hparams']['noise_std']
    if  'noise_clip' in config_json['hparams']:
          noise_clip = config_json['hparams']['noise_clip']
    print("Simulated Noise STD ",noise_std,"/ CLIP ",noise_clip)
    #-----------------------------------------------------------------
    gain_jitter = float(config_json['hparams'].get('gain_jitter', 0.0))
    polar_flip  = bool(config_json['hparams'].get('polar_flip', False))
    channel_jitter = float(config_json['hparams'].get('channel_jitter', 0.0))
    monochrome = bool(config_json['hparams'].get('monochrome', False))
    polar_rot = bool(config_json['hparams'].get('polar_rot', False))
    frozen_body_start_epochs = int(config_json['hparams'].get('frozen_body_start_epochs', 0))
    frozen_body_end_epochs   = int(config_json['hparams'].get('frozen_body_end_epochs', 0))
    custom_early_convs       = int(config_json['hparams'].get('custom_early_convs', 0))
    custom_channels          = config_json['hparams'].get('custom_channels', None)
    custom_res_blocks        = config_json['hparams'].get('custom_res_blocks', None)
    custom_wavelet_pools     = config_json['hparams'].get('custom_wavelet_pools', None)
    pretrained_backbone      = bool(config_json['hparams'].get('pretrained', True))
    print("Augmentation: gain_jitter ", gain_jitter, "/ polar_flip ", polar_flip,
          "/ channel_jitter ", channel_jitter, "/ monochrome ", monochrome,
          "/ polar_rot ", polar_rot)
    print("Frozen body epochs: start ", frozen_body_start_epochs, "/ end ", frozen_body_end_epochs)
    #-----------------------------------------------------------------
    # Derived polarization input channels (computed on-GPU from the 4 raw ones)
    use_AoLP  = bool(config_json['hparams'].get('AoLP', False))
    use_DoLP  = bool(config_json['hparams'].get('DoLP', False))
    use_unpol = bool(config_json['hparams'].get('Unpolarized',
                     config_json['hparams'].get('unpolarized', False)))
    use_maxp  = bool(config_json['hparams'].get('MaxPolarization', False))
    use_minp  = bool(config_json['hparams'].get('MinPolarization', False))
    use_rngp  = bool(config_json['hparams'].get('RangePolarization', False))
    print("Derived channels: AoLP", use_AoLP, "/ DoLP", use_DoLP, "/ Unpolarized", use_unpol,
          "/ Max", use_maxp, "/ Min", use_minp, "/ Range", use_rngp)
    #-----------------------------------------------------------------



    #Let's Go..
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    directory =  config_json['training_dataset']

    # Rearrange HWC uint8 numpy array → CHW uint8 tensor.
    # We do NOT divide by 255 here; normalisation happens inside Classifier.forward()
    # on the GPU, so the DataLoader transfers 4× less data over PCIe.
    transform = transforms.Lambda(
        lambda img: torch.from_numpy(img).permute(2, 0, 1).contiguous()
    )

    # training_dataset may be a single dir OR a LIST of dirs to combine (e.g.
    # FORTH + Altinay for a mixed-domain train, then frame_disjoint_split). Each
    # is loaded and gets the SAME class scheme (filter/merge/drop) BEFORE combining
    # so class spaces stay consistent.
    directories = directory if isinstance(directory, list) else [directory]

    def _load_one(d):
        h5 = '%s/dataset.h5' % d
        if checkIfFileExists(h5):
            from DatasetConverter import HDF5Dataset
            print("Using H5 dataset loader ", h5)
            ds = HDF5Dataset(h5)
            ds.metadata = None  # training/validation steps unpack (x, y) batches only
        else:
            print("Using Normal PNG dataset loader ", d)
            ds = RGBAImageFolder(root=d, transform=transform)
        # Keep some classes / merge / drop -- applied per dataset so the combined
        # label space is consistent across all constituents.
        if config_json.get('selected_classes') and len(config_json['selected_classes']) > 1:
            print("Only selecting the classes ", config_json['selected_classes'])
            filter_dataset_classes(ds, config_json['selected_classes'])
        # Non-severity view first (collapse class_XClassY -> class_X), so the
        # merge/drop below operate on base names. Set "severity": false (or
        # "strip_severity": true) in the config. Granular H5 + this flag = the
        # runtime "non-severity" mode; omit it to train on the full severity truth.
        if config_json.get('strip_severity') or config_json.get('severity') is False:
            strip_severity_classes(ds)
        if config_json.get('class_merges'):
            merge_dataset_classes(ds, config_json['class_merges'])
        drops = list(config_json.get('drop_classes') or [])
        drops += resolve_auto_drops(ds.classes, config_json)
        if drops:
            drop_dataset_classes(ds, drops)
        return ds

    subs = [_load_one(d) for d in directories]
    if len(subs) == 1:
        dataset = subs[0]
    else:
        # Combining multiple H5s that may have different class SUBSETS/orders
        # (e.g. base train + a granular dump collapsed to base): force all onto
        # one canonical list so CombinedDataset accepts them. canonical_classes
        # in the config, else the primary (first) dataset's post-transform list.
        canon = config_json.get('canonical_classes') or list(subs[0].classes)
        for ds in subs:
            align_dataset_to_classes(ds, canon)
        dataset = CombinedDataset(subs)   # now identical class spaces
        print(f"Combined {len(subs)} datasets -> {len(dataset)} tiles, classes {dataset.classes}")
    H5PYFilename = '%s/dataset.h5' % directories[0]   # legacy single-file references

    # Sanity-check the final training label space before we train on it.
    print_class_distribution(dataset, title="TRAIN (final label space, pre-training)")

    # Optionally preload all samples to RAM (slow startup, fast epoch iteration)
    if ('cacheAllDataToRAM' in config_json['dataloader']) and (config_json['dataloader']['cacheAllDataToRAM']):
      # --- Sanity check: ensure there is enough available RAM ---
      available_ram = _get_available_ram_bytes()
      # On-disk footprint (quick lower bound)
      disk_bytes = _get_path_size_bytes(directory)
      # Heuristic RAM estimate based on sampling decoded items
      est_ram_bytes = _estimate_dataset_ram_bytes(dataset, sample_count=32)
    
      print("\n[RAM CHECK] Dataset on disk: ", _human_bytes(disk_bytes))
      if available_ram > 0:
            print("[RAM CHECK] System available RAM: ", _human_bytes(available_ram))
      else:
            print("[RAM CHECK] Could not determine available RAM (continuing without hard check).")
    
      # If we can measure available RAM, enforce a safety margin.
      if available_ram > 0:
            # Use the larger of the decoded estimate and 2x on-disk as a conservative requirement
            conservative_required = max(est_ram_bytes, int(disk_bytes * 2.0))
            safety_margin = 0.90  # do not consume more than 90% of available RAM
            limit = int(available_ram * safety_margin)
    
            print("[RAM CHECK] Estimated RAM needed to cache: ", _human_bytes(conservative_required))
            if conservative_required > limit:
                  print("\n[ERROR] Not enough available RAM to safely cache the entire dataset.")
                  print("        Required (est.): ", _human_bytes(conservative_required))
                  print("        Available:       ", _human_bytes(available_ram))
                  print("        Safety limit:    ", _human_bytes(limit), f" ({int(safety_margin*100)}% of available)")
                  print("\n        Tip: disable cacheAllDataToRAM or reduce dataset / tile size / dtype, or run on a machine with more RAM.\n")
                  sys.exit(1)
    
      dataset = RAMPreloadedDataset(dataset)
      num_workers = 1 #Set workers to 1 to avoid RAM duplication 


    # Calculate the sizes for training and validation sets
    dataset_size    = len(dataset)
    validation_size = int(val_split * dataset_size)
    train_size      = dataset_size - validation_size
    
    #Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    

    # Print class distribution BEFORE splitting
    print_class_distribution(dataset, title="Full Dataset")


    """
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(
                                              dataset,
                                              [train_size, validation_size],
                                              generator=torch.Generator().manual_seed(seed),
                                             )

    print_class_distribution(train_dataset, title="Training Set")
    print_class_distribution(val_dataset, title="Validation Set")
    """




    # -----------------------------------------------------------
    # Optional external validation dataset
    # -----------------------------------------------------------
    val_directory = None
    if "validation_dataset" in config_json and config_json["validation_dataset"]:
        val_directory = config_json["validation_dataset"]

    if val_directory is not None:
        print("Using explicit validation dataset:", val_directory)

        H5PYValFilename = f"{val_directory}/dataset.h5"

        if checkIfFileExists(H5PYValFilename):
            from DatasetConverter import HDF5Dataset
            val_dataset = HDF5Dataset(H5PYValFilename)
            val_dataset.metadata = None  # training/validation steps unpack (x, y) batches only
        else:
            val_dataset = RGBAImageFolder(root=val_directory, transform=transform)

        if ('selected_classes' in config_json) and config_json['selected_classes']:
            filter_dataset_classes(val_dataset, config_json['selected_classes'])

        if config_json.get('class_merges'):
            merge_dataset_classes(val_dataset, config_json['class_merges'])

        val_drops = list(config_json.get('drop_classes') or [])
        val_drops += resolve_auto_drops(val_dataset.classes, config_json)
        if val_drops:
            drop_dataset_classes(val_dataset, val_drops)

        print_class_distribution(val_dataset, title="VAL (final label space)")

        if dataset.classes != val_dataset.classes:
            raise ValueError(f"Training/validation class mismatch: {dataset.classes} vs {val_dataset.classes}")

        if dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError("Training/validation class_to_idx mismatch")

        train_dataset = dataset
    elif config_json['dataloader'].get('frame_disjoint_split'):
        # Train/test split of the (possibly combined) training set BY FRAME (no
        # tile leakage). Used for the within-domain ceiling (FORTH->FORTH) and for
        # mixed-domain runs (FORTH+Altinay -> held-out frames from both).
        print("No validation_dataset → FRAME-DISJOINT split of the training set")
        from torch.utils.data import Subset
        tr_idx, va_idx = frame_disjoint_split(dataset, val_split, seed)
        train_dataset = Subset(dataset, tr_idx)
        val_dataset   = Subset(dataset, va_idx)
    else:
        print("No validation_dataset provided → using validation_split (TILE-level; "
              "may leak sibling tiles — set dataloader.frame_disjoint_split for a clean split)")

        dataset_size    = len(dataset)
        validation_size = int(val_split * dataset_size)
        train_size      = dataset_size - validation_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(seed),
        )

    # -----------------------------------------------------------

    print_class_distribution(train_dataset, title="Training Set")
    print_class_distribution(val_dataset, title="Validation Set")





    # Create DataLoaders
    if balanced_sampling:
        if isinstance(train_dataset, torch.utils.data.Subset):
            train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
        else:
            train_targets = train_dataset.targets
        print(f"Using BalancedBatchSampler (batch_size={batch_size}, num_classes={len(dataset.classes)})")
        balanced_sampler = BalancedBatchSampler(
            targets     = train_targets,
            num_classes = len(dataset.classes),
            batch_size  = batch_size,
        )
        train_loader = DataLoader(
                                  train_dataset,
                                  batch_sampler=balanced_sampler,
                                  num_workers=num_workers,
                                 )
    else:
        train_loader = DataLoader(
                                  train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                 )

    val_loader  = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False,  # never discard validation samples
                           )

    #Print class names as a sanity check
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    #Get clean class which could be needed for extra penalization. None when the
    #dataset has no clean class (e.g. an 'alldefect' run that dropped class_clean)
    #-- val_detect_auroc / penalize_false_clean / the threshold sweep all skip.
    cleanClassID = None
    for i in range(len(dataset.classes)):
           if (dataset.classes[i] == "class_clean") or (dataset.classes[i] == "Clean"):
              cleanClassID = i
    print(f"Clean class ID is : {cleanClassID}")

    """
    if class_weight:
        class_counts = Counter(train_dataset.dataset.targets)
        alpha = torch.tensor([1 / class_counts[i] for i in range(len(class_counts))])
        alpha = alpha / alpha.sum() 
        alpha = alpha.to(device)
    else:
        alpha = None
    """
    if class_weight:
        if isinstance(train_dataset, torch.utils.data.Subset):
            train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
        else:
            train_targets = train_dataset.targets

        class_counts = Counter(train_targets)
        alpha = torch.tensor(
            [1 / class_counts[i] for i in range(len(class_names))],
            dtype=torch.float32
        )
        alpha = alpha / alpha.sum()
        alpha = alpha.to(device)
    else:
        alpha = None


    # Initialize the classifier
    classifier = Classifier(
                            model=config_json['model'],
                            lr=lr,
                            num_classes=len(class_names),
                            tile_size=config_json['hparams']['tile_size'],
                            dropout_rate=dropout_rate,
                            penalize_false_clean=penalize_false_clean,
                            base_channels=base_channels,
                            final_dense_layer=final_dense_layer,
                            clean_class=cleanClassID,
                            noise_std=noise_std,
                            noise_clip=noise_clip,
                            gain_jitter=gain_jitter,
                            polar_flip=polar_flip,
                            channel_jitter=channel_jitter,
                            monochrome=monochrome,
                            polar_rot=polar_rot,
                            frozen_body_start_epochs=frozen_body_start_epochs,
                            frozen_body_end_epochs=frozen_body_end_epochs,
                            custom_early_convs=custom_early_convs,
                            custom_channels=custom_channels,
                            custom_res_blocks=custom_res_blocks,
                            custom_wavelet_pools=custom_wavelet_pools,
                            pretrained=pretrained_backbone,
                            AoLP=use_AoLP,
                            DoLP=use_DoLP,
                            Unpolarized=use_unpol,
                            MaxPolarization=use_maxp,
                            MinPolarization=use_minp,
                            RangePolarization=use_rngp)
    print(f"Learning rate: {lr}")
    
    model_type = classifier.type

    # Initialize a PyTorch Lightning trainer
    if use_wandb:
        loggers = [TensorBoardLogger("lightning_logs", name="classifier"),  WandbLogger(project=config_json['wandb']['project'], log_model=True,name=config_json['wandb']['name']+model_type+loss)]
    else:
        if checkIfPathExists("tensorboard/"):
            print("Removing previous tensorboard logs..")
            shutil.rmtree("tensorboard/", ignore_errors=True)
        loggers = [TensorBoardLogger("tensorboard", name=model_name)]

    # Keep the epoch with the best validation loss; cross-site val metrics drift
    # non-monotonically, so last-epoch weights are often not the best weights.
    #
    # save_top_k=1 (the default) DELETES every other epoch, which makes the
    # "do more epochs help?" question unanswerable: val_loss is a SOURCE-domain
    # quantity and bottoms out around epoch 0-1, so it selects an early epoch and
    # discards the later ones -- even if cross-site detection (our actual KPI)
    # keeps improving after val_loss stops. Set "checkpoint_save_top_k": -1 in the
    # config to keep every epoch, then rank them by AUROC of 1-P(clean) on the
    # held-out site instead of by val_loss. "checkpoint_dir" redirects the files
    # (they are ~30MB each and /home is nearly full).
    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_top_k  = int(config_json.get("checkpoint_save_top_k", 1))
    ckpt_dir    = config_json.get("checkpoint_dir", None)
    # Default to selecting on DETECTION AUROC, not val_loss. Measured over the
    # 24-epoch customwide run, pearson(val_loss, detect AUROC) = -0.09 and
    # pearson(val_loss, macro) = -0.00 => val_loss ranks epochs at random w.r.t.
    # the KPI, because focal+pfc scores 6-way class identity while the KPI asks
    # "defect or clean". Set "checkpoint_monitor": "val_loss" to get the old
    # behaviour back.
    ckpt_monitor = config_json.get("checkpoint_monitor", "val_detect_auroc")
    ckpt_mode    = config_json.get("checkpoint_mode",
                                   "min" if ckpt_monitor.endswith("loss") else "max")
    # Filename references only metrics that are actually logged: val_loss always,
    # plus the monitored metric (val_detect_auroc is absent on alldefect runs).
    ckpt_fname = '{epoch}-{step}-{val_loss:.4f}'
    if ckpt_monitor != 'val_loss':
        ckpt_fname += '-{' + ckpt_monitor + ':.4f}'
    best_ckpt = ModelCheckpoint(monitor=ckpt_monitor, mode=ckpt_mode,
                                save_top_k=ckpt_top_k,
                                dirpath=ckpt_dir,
                                filename=ckpt_fname)
    print(f"Checkpoints: monitor={ckpt_monitor} mode={ckpt_mode} save_top_k={ckpt_top_k} "
          f"({'ALL epochs kept' if ckpt_top_k == -1 else 'best epoch(s) only'}) "
          f"dir={ckpt_dir or '<logger default>'}")

    trainer = pl.Trainer(
                         max_epochs=epochs,
                         logger=loggers,
                         callbacks=[best_ckpt],
                         #callbacks=[EarlyStopping(monitor='val_loss')],
                         accelerator=config_json['accelerator'],
                         devices=config_json['devices'],
                         gradient_clip_val=gradient_clip_val,
                         deterministic= (noise_std==0.0), #<- When there is noise switch to non deterministic to speed up training
                       )

    #Train and log to console
    #------------------------------------------------------------------
    trainer.fit(classifier, train_loader, val_loader)

    # Restore the best-val_loss epoch before saving/validating/confusion matrix
    if best_ckpt.best_model_path:
        print("Restoring best checkpoint:", best_ckpt.best_model_path,
              "(%s = %s)" % (ckpt_monitor, str(best_ckpt.best_model_score)))
        best_state = torch.load(best_ckpt.best_model_path, weights_only=False)
        classifier.load_state_dict(best_state['state_dict'])


    #Save the model
    #------------------------------------------------------------------
    print("Saving the model")
    trainer.save_checkpoint('%s.pth' % model_name)

    # Evaluate dumped tiles (if directory exists)
    #------------------------------------------------------------------
    #tiles_dir = "tiles_dump_1760036129"  # replace or detect automatically
    #if os.path.isdir(tiles_dir):
    #    metrics = evaluate_dumped_tiles(classifier.model, tiles_dir, class_names, device=device)
    #    if metrics:
    #        with open("tile_evaluation.json", "w") as f:
    #            json.dump(metrics, f, indent=2)


    #Predictions
    #------------------------------------------------------------------
    print("Final model validation")
    classifier.eval()
    trainer.validate(classifier, val_loader)
    #------------------------------------------------------------------

    try:
        print("Removing previous confusion matrix data..")
        for stale in glob.glob(f"{model_name}_confusion.json") + glob.glob(f"{model_name}*.png"):
            try:
                os.remove(stale)
            except OSError:
                pass

        print("Generating new confusion matrix data..")
        y_true = []
        y_pred = []
        y_maxp = []   # max softmax probability      -> legacy max_prob gate sweep
        y_pcln = []   # P(clean) per tile            -> defect_mass gate sweep

        # torch.no_grad() prevents graph construction for every batch (saves memory).
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(classifier.device)
                probs = torch.softmax(classifier(x), dim=1)
                mp, pr = probs.max(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pr.cpu().numpy())
                y_maxp.extend(mp.cpu().numpy())
                if cleanClassID is not None:
                    y_pcln.extend(probs[:, cleanClassID].cpu().numpy())

        num_classes = len(dataset.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            confusion_matrix[true, pred] += 1
        print(confusion_matrix)

        # Embed confusion matrix in the main config JSON
        config_json["confusion_matrix"] = confusion_matrix.tolist()
        config_json["classes_int"] = [int(c) for c in set(y_true)]
        config_json["classes"] = dataset.classes

        # Write a separate confusion JSON and generate the plot image
        print("Generating confusion matrix plot")
        confusion_json = {
            "title":  f"{model_name} / Tile Size = {tile_size} / Epochs = {epochs}",
            "labels": dataset.classes,
            "matrix": confusion_matrix.tolist(),
        }
        with open(f"{model_name}_confusion.json", "w") as f:
            json.dump(confusion_json, f, indent=2)
        subprocess.run(
            [sys.executable, "plotTool.py", f"{model_name}_confusion.json"], check=False
        )

        # Threshold sweep -> operating curve + the gate the live path will use.
        # Swept for BOTH gates (liveClassifierTorch.gate_tiles implements them):
        #   "defect_mass" : score = 1 - P(clean), the total mass on ANY defect
        #                   class. Recommended. Flags a tile the model is sure is
        #                   a defect even when it cannot say which one.
        #   "max_prob"    : score = max_c P(c), and the tile must not argmax to
        #                   clean. Legacy, kept so older runs stay comparable. It
        #                   discards a 0.40 Welding / 0.40 Seal / 0.20 clean tile
        #                   as clean despite it being 80% likely a defect.
        # Thresholds are NOT comparable between the two modes or across models,
        # so we write the chosen one into config_json["gate"] and the live path
        # reads it from there instead of hardcoding a number.
        # The sweep is a defect-vs-clean operating curve -- meaningless without a
        # clean class (e.g. an 'alldefect' typer), so skip it there.
        if cleanClassID is None:
            print("No clean class -> skipping threshold sweep / gate calibration")
            raise _SkipSweep()
        print("Generating threshold sweep / operating curve")
        yt = np.array(y_true); yp = np.array(y_pred)
        mp = np.array(y_maxp); pcl = np.array(y_pcln)
        isdef = yt != cleanClassID
        n_def = max(1, int(isdef.sum())); n_cln = max(1, int((~isdef).sum()))

        def run_sweep(score, also_require=None):
            out = []
            for t in np.arange(0.0, 0.996, 0.005):
                flagged = score >= t
                if also_require is not None:
                    flagged = flagged & also_require
                out.append({
                    "threshold":   round(float(t), 3),
                    "detected":    float((flagged &  isdef).sum() / n_def),
                    "false_alarm": float((flagged & ~isdef).sum() / n_cln),
                })
            return out

        def pick(sweep):
            balance  = [s["detected"] - s["false_alarm"] for s in sweep]        # Youden J
            kpi_cost = [2*(1-s["detected"]) + s["false_alarm"] for s in sweep]  # misses weigh 2x
            # Deployment pick: tile rates ignore prevalence (a frame is ~99% clean,
            # so 1% FA = dozens of false crosses). Highest detection subject to a
            # false-alarm budget of 0.5% of clean tiles (~30 crosses/frame @ step 14).
            in_budget = [s for s in sweep if s["false_alarm"] <= 0.005]
            return (sweep[int(np.argmax(balance))],
                    sweep[int(np.argmin(kpi_cost))],
                    max(in_budget, key=lambda s: s["detected"]) if in_budget else sweep[-1])

        sweeps = {
            "max_prob":    run_sweep(mp, also_require=(yp != cleanClassID)),
            "defect_mass": run_sweep(1.0 - pcl),
        }
        picks = {k: pick(v) for k, v in sweeps.items()}
        for mode, (b, k, d) in picks.items():
            print(f"[{mode}] balanced t={b['threshold']:.3f} "
                  f"(detected {b['detected']:.1%}, FA {b['false_alarm']:.1%}) | "
                  f"KPI t={k['threshold']:.3f} (detected {k['detected']:.1%}, FA {k['false_alarm']:.1%}) | "
                  f"deploy t={d['threshold']:.3f} (detected {d['detected']:.1%}, FA {d['false_alarm']:.2%})")

        # Which gate ships. Override per run with "gate_mode" in the config.
        gate_mode = config_json.get("gate_mode", "defect_mass")
        best_bal, best_kpi, best_dep = picks[gate_mode]
        # Legacy keys keep their original max_prob meaning so old configs/readers
        # do not silently change semantics.
        lb, lk, ld = picks["max_prob"]
        config_json["best_threshold_balanced"]   = lb
        config_json["best_threshold_kpi"]        = lk
        config_json["best_threshold_deployment"] = ld
        config_json["gate"] = {
            "mode": gate_mode,
            "threshold": best_kpi["threshold"],      # KPI-optimal: misses weigh 2x
            "assign_best_defect_class": True,
            "calibrated_on": config_json.get("validation_dataset", ""),
            "detected": best_kpi["detected"],
            "false_alarm": best_kpi["false_alarm"],
            "alternatives": {"balanced": best_bal, "deployment": best_dep},
        }
        print(f"config['gate'] -> mode={gate_mode} threshold={best_kpi['threshold']:.3f} "
              f"(detected {best_kpi['detected']:.1%}, FA {best_kpi['false_alarm']:.1%})")
        curve_json = {
            "title": f"{model_name} / Tile Size = {tile_size} / Epochs = {epochs}",
            "sweep": sweeps[gate_mode],
            "sweeps": sweeps,
            "best_balanced": best_bal,
            "best_kpi": best_kpi,
        }
        with open(f"{model_name}_threshold_curve.json", "w") as f:
            json.dump(curve_json, f, indent=2)
        subprocess.run(
            [sys.executable, "plotTool.py", f"{model_name}_threshold_curve.json"], check=False
        )
    except _SkipSweep:
        pass   # no clean class; confusion matrix already written above
    except Exception as e:
        print("Failed generating a confusion matrix:", e)
        with open(f"{model_name}_confusion.json", "w") as f:
            f.write("Failed\n")


    #Compute MD5 of saved model for corruption detection
    #------------------------------------------------------------------
    pth_path = '%s.pth' % model_name
    md5 = hashlib.md5()
    with open(pth_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    config_json['model_md5'] = md5.hexdigest()

    #Save the JSON
    #------------------------------------------------------------------
    print("Saving the JSON")
    with open('%s.json' % model_name, 'w') as f:
       json.dump(config_json, f, indent=2)


    #Update symbolic links 
    #------------------------------------------------------------------
    #os.system("rm last.pth last.json && ln -s %s.pth last.pth && ln -s %s.json last.json" % (model_name,model_name) )
    #No longer needed

    # Build zip filename with timestamp
    #------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{model_name}_{timestamp}.zip"

    print(f"Saving everything as {zip_name} archive")
    os.makedirs("models/", exist_ok=True)
    zip_inputs = (
        [f"{model_name}.json", f"{model_name}_confusion.json", f"{model_name}.pth"]
        + glob.glob(f"{model_name}*.png")
        + glob.glob("tensorboard/*/*/*")
    )
    subprocess.run(["zip", "-r", f"models/{zip_name}"] + zip_inputs, check=False)


    print('To upload ALL models copy/paste:') 
    print("scp -P 2222 models/*.zip ammar@ammar.gr:/home/ammar/public_html/magician/ckpts2")  

    print('To upload last training results copy/paste:') 
    print("scp -P 2222 models/%s ammar@ammar.gr:/home/ammar/public_html/magician/ckpts2" % zip_name)  
    
