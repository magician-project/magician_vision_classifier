"""Class-scheme transforms shared by the trainer and the analysis tools.

filter -> strip_severity -> merge -> drop, applied identically to the training and
validation paths via apply_class_scheme(). PLAN.md records two historical bugs that
came from having two copies of this logic (the validation path omitting
strip_severity, and non-severity runs collapsing to 2 classes), so it lives in ONE
place now.
"""

from collections import Counter

import torch

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
    # Samples of a class absent from `target_classes` are DISCARDED. That is often
    # intended, but it used to happen without a word -- a whole constituent could
    # contribute nothing but clean tiles and nobody would notice. Report it.
    _n_before = len(dataset)
    _lost_classes = [c for c in old_classes if c not in tgt_idx]

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
    if _lost_classes:
        _lost = _n_before - len(dataset)
        print(f"[align_classes] WARNING discarded {_lost:,}/{_n_before:,} samples "
              f"({100.0 * _lost / max(1, _n_before):.1f}%) belonging to classes absent "
              f"from the canonical list: {_lost_classes}")
        if _lost >= _n_before:
            print("[align_classes] WARNING this dataset now contributes NOTHING — "
                  "its class names almost certainly do not match the canonical scheme "
                  "(e.g. a base-named dump aligned against a granular canon)")
    return dataset

def _expand_family_merges(merges, classes):
    """Let a BARE family name in class_merges stand for all of its severity
    variants, matching the convention drop_class_families already uses.

    Configs are written as {"class_Seal": "class_Welding"}, but a granular dataset
    holds class_SealClassA / class_SealClassB and no bare class_Seal -- so the
    exact-match lookup below found nothing and the merge silently no-opped. Here a
    source that is not itself a class is expanded over every class sharing its base
    name:
        class_Seal -> class_Welding   becomes  class_SealClassA -> class_WeldingClassA
                                              class_SealClassB -> class_WeldingClassB
    Severity is carried across only when the destination actually has that variant
    (class_WeldingClassA exists); otherwise every variant collapses onto the plain
    destination, which is what a binary bucket wants:
        class_Seal -> class_defect    becomes  class_SealClassA -> class_defect
    Sources that ARE exact class names are passed through untouched, so datasets
    with bare class names behave exactly as before.
    """
    import re as _re
    suffix = _re.compile(r'(Class[ABC])$')
    have = set(classes)
    out = {}
    for src, dst in merges.items():
        matched = False
        for c in classes:
            if suffix.sub('', c) != src:
                continue
            # Covers the bare class itself (base == src, no suffix) AND every
            # severity variant, so a family that also has a severityless bucket
            # -- class_PositiveDent alongside class_PositiveDentClassA -- merges
            # in full rather than only the exact-name hit.
            m = suffix.search(c)
            sev_dst = dst + m.group(1) if m else dst
            out[c] = sev_dst if sev_dst in have else dst
            matched = True
        if not matched:
            out[src] = dst          # absent either way; filtered out below
    return out

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
    # A bare family name ('class_Seal') stands for its severity variants; see
    # _expand_family_merges. No-op when the sources are already exact class names.
    merges = _expand_family_merges(merges, old_classes)
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
    # Safety net: a drop rule that eats most of the label space is almost always a
    # config interaction rather than intent (see resolve_auto_drops' strip_severity
    # guard). Fail on a degenerate result, warn on a suspicious one, so the next
    # variant of that mistake cannot pass silently.
    if len(new_classes) < 2:
        raise ValueError(
            f"[drop_classes] dropping {sorted(drop_set)} leaves {new_classes} — "
            f"a classifier needs at least 2 classes. Check drop_classes / "
            f"drop_severityless_defects / drop_class_families in the config.")
    if len(drop_set) > len(new_classes):
        print(f"[drop_classes] WARNING dropping {len(drop_set)} classes "
              f"{sorted(drop_set)} leaves only {len(new_classes)} "
              f"({new_classes}) — verify this is intended")
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

def apply_class_scheme(dataset, config_json, label="dataset"):
    """THE single place the config's class scheme is applied: filter -> strip_severity
    -> merge -> drop. Training and validation MUST go through this same function.

    They used to be written out separately, and the validation copy was missing
    `strip_severity` (and the `align` below). With "strip_severity": true the training
    set collapsed severities to base names while validation kept its own ordering, so
    the two ended up with the SAME class set in a DIFFERENT order and the consistency
    check further down raised "Training/validation class mismatch" -- i.e. every
    strip_severity config that also names a validation_dataset failed at startup
    (crossval_v2_rot_customwide, crossval_v2_rot_alldefect).
    """
    if config_json.get('selected_classes') and len(config_json['selected_classes']) > 1:
        print(f"[class_scheme:{label}] selecting classes", config_json['selected_classes'])
        filter_dataset_classes(dataset, config_json['selected_classes'])
    # Non-severity view first (collapse class_XClassY -> class_X), so the merge/drop
    # below operate on base names. Set "severity": false (or "strip_severity": true).
    if config_json.get('strip_severity') or config_json.get('severity') is False:
        strip_severity_classes(dataset)
    if config_json.get('class_merges'):
        merge_dataset_classes(dataset, config_json['class_merges'])
    drops = list(config_json.get('drop_classes') or [])
    drops += resolve_auto_drops(dataset.classes, config_json)
    if drops:
        drop_dataset_classes(dataset, drops)
    print(f"[class_scheme:{label}] -> {len(dataset.classes)} classes: {dataset.classes}")
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
        # GUARD: strip_severity has already removed every Class[ABC] suffix, so this
        # rule would match EVERY defect and leave the run with class_clean plus
        # whatever the merges happen to protect. Measured on train_nonaltinay_v2 with
        # crossval_v2_rot_customwide.json: 7 classes -> ['class_clean',
        # 'class_Welding'], discarding 354,480 defect tiles (73% of all defects)
        # without a word. Warn and skip instead.
        #
        # NOTE this keys off the CONFIG FLAG, not off "no class carries a suffix".
        # Those are different situations with opposite correct answers: a natively
        # base-named dump (val_altinay, train_nonaltinay, val_canonical) also has no
        # suffixes, and there dropping IS intended -- admitting a severity-less
        # class_Welding beside class_WeldingClassA would create a bogus fourth
        # "no severity" category. Only the strip_severity case is the mistake.
        if config_json.get('strip_severity') or config_json.get('severity') is False:
            print("[auto_drops] WARNING drop_severityless_defects IGNORED: "
                  "strip_severity removed every Class[ABC] suffix, so this rule would "
                  "match every defect and collapse the label space. Use "
                  "drop_class_families to drop specific families in a non-severity run.")
        else:
            # A merge DESTINATION is created deliberately by the config, so it is not
            # "a defect whose severity was never labelled". Without this guard a binary
            # bucket -- {"class_Welding": "class_defect", ...} -- would be built by the
            # merge and then immediately auto-dropped for lacking a Class[ABC] suffix,
            # leaving the run with class_clean alone.
            protected = set((config_json.get('class_merges') or {}).values())
            for c in classes:
                if c != clean_name and c not in protected and not suffix.search(c):
                    drop.add(c)
    families = set(config_json.get('drop_class_families') or [])
    if families:
        for c in classes:
            if c in families or suffix.sub('', c) in families:
                drop.add(c)
    return sorted(drop)

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
