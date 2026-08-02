"""Dataset plumbing shared by the trainer and the analysis tools.

Loaders (RGBAImageFolder + the PNG metadata readers), the wrappers that combine and
cache datasets (CombinedDataset, RAMPreloadedDataset), the balanced sampler, and the
frame-disjoint split.

Two interactions in here are easy to get wrong and are documented at their call sites:
  * RAMPreloadedDataset wraps the dataset BEFORE the split, hiding .datasets/.file --
    _dataset_source_frames unwraps it, or frame_disjoint_split cannot see the frame
    metadata (see the unwrap note in _dataset_source_frames).
  * The RAM safety check's 2x-on-disk term silently degrades to 0 when
    training_dataset is a LIST of dirs (os.path.isfile on a list raises and is
    swallowed), leaving only the sampled estimate as a guard.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

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

def _dataset_source_frames(dataset):
    """Per-sample source-FRAME string for every sample (aligned with dataset[i],
    0..len-1). Frame = metadata 'source' path minus the tile (x,y) offset. Works
    on HDF5Dataset (reads its open .file, respects a prior .indices row subset)
    and CombinedDataset (concatenates constituents in the same order it indexes).
    'source' paths are globally unique across dumps, so combined frame ids stay
    distinct across FORTH / Altinay / etc. without extra offsetting."""
    import json as _json
    # cacheAllDataToRAM wraps the dataset in RAMPreloadedDataset BEFORE the split,
    # and that wrapper forwards only classes/targets -- so .datasets/.file (the
    # frame metadata this needs) become invisible and the split raises below.
    # Unwrap to the cached-from dataset: RAMPreloadedDataset caches dataset[i] for
    # i in 0..n-1 in order, so frame i still corresponds to cached sample i, and
    # _h5_frames honours the .indices row subsets apply_class_scheme installed.
    if not hasattr(dataset, "datasets") and not hasattr(dataset, "file"):
        inner = getattr(dataset, "_dataset", None)
        if inner is not None:
            dataset = inner
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
    return rgba_image

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
