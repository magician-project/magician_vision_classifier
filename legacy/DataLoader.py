from torch.utils.data import Dataset

class LabelRemapDataset(Dataset):
    """
    Wraps a dataset and remaps its labels to a canonical class_to_idx mapping.
    Exposes:
      - classes (canonical)
      - class_to_idx (canonical)
      - targets (remapped)
    """
    def __init__(self, dataset: Dataset, canonical_classes: list[str], canonical_class_to_idx: dict[str, int]):
        super().__init__()
        self.dataset = dataset
        self.classes = canonical_classes
        self.class_to_idx = canonical_class_to_idx

        # Try to get source mapping
        src_classes = getattr(dataset, "classes", None)
        src_cti = getattr(dataset, "class_to_idx", None)

        if src_cti is None and src_classes is not None:
            src_cti = {c: i for i, c in enumerate(src_classes)}

        if src_cti is None:
            raise ValueError("Dataset does not expose classes/class_to_idx, cannot remap safely.")

        self._src_idx_to_name = {i: name for name, i in src_cti.items()}

        # Build remapped targets (needed by your code: distribution + class_weight) :contentReference[oaicite:3]{index=3}
        src_targets = getattr(dataset, "targets", None)
        if src_targets is None:
            # Fallback: infer targets by indexing (slower at init)
            src_targets = [dataset[i][1] for i in range(len(dataset))]

        remapped = []
        for t in src_targets:
            name = self._src_idx_to_name.get(int(t), None)
            if name is None:
                raise ValueError(f"Found target idx {t} not present in src mapping.")
            if name not in canonical_class_to_idx:
                # If you want to "skip unknown classes" instead, we can implement filtering;
                # but safest is to error loudly.
                raise ValueError(f"Class '{name}' not in canonical mapping.")
            remapped.append(canonical_class_to_idx[name])

        self.targets = remapped

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # y is src index -> map to name -> canonical index
        name = self._src_idx_to_name[int(y)]
        return x, self.class_to_idx[name]


class CombinedDataset(Dataset):
    """Concatenate multiple datasets and expose .classes/.targets like your code expects."""
    def __init__(self, datasets: list[Dataset], classes: list[str], class_to_idx: dict[str, int]):
        super().__init__()
        self.datasets = datasets
        self.classes = classes
        self.class_to_idx = class_to_idx

        self._lengths = [len(d) for d in datasets]
        self._offsets = []
        s = 0
        for L in self._lengths:
            self._offsets.append(s)
            s += L
        self._total_len = s

        # Concatenate targets for your print_class_distribution + class_weight path 
        self.targets = []
        for d in datasets:
            t = getattr(d, "targets", None)
            if t is None:
                self.targets.extend([d[i][1] for i in range(len(d))])
            else:
                self.targets.extend(list(t))

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)
        for j in range(len(self.datasets) - 1, -1, -1):
            if idx >= self._offsets[j]:
                return self.datasets[j][idx - self._offsets[j]]
        raise RuntimeError("CombinedDataset indexing error")
