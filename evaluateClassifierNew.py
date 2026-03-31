#!/usr/bin/python3
"""
Evaluate a trained classifier checkpoint on one or more datasets.

Usage:
    python3 evaluateClassifierNew.py model.ckpt config.json /path/to/evaluation_dir
    python3 evaluateClassifierNew.py model.pth  config.json /path/to/evaluation_dir
    python3 evaluateClassifierNew.py model.pth  config.json /path/to/evaluation_dir 16
    python3 evaluateClassifierNew.py model.pth  config.json /path/to/evaluation_dir 16 Distance1,Distance2,Distance3,Light1
    python3 evaluateClassifierNew.py model.pth  config.json /path/to/evaluation_dir 16 Distance1,Distance2,Distance3,Light1 25

e.g.
    python3 evaluateClassifierNew.py v1/allclass_resnet18.pth v1/allclass_resnet18.json /media/ammar/games2/Datasets/MagicianWeldings/ 16 Distance1,Distance2,Distance3,DistanceAverage,Light1,Light2,Light3,Light4,Light5,Light6 25 
    python3 evaluateClassifierNew.py v1/allclass_resnet18.pth v1/allclass_resnet18.json /media/ammar/games2/Datasets/MagicianWeldings/ 16 DistanceAverage,Light1,Light2,Light3,Light4,Light5,Light6 25 



What it does:
- Loads the JSON config
- Reconstructs the Classifier exactly like training
- Loads the checkpoint weights
- Resolves model outputs by trained class names
- Scans the evaluation directory:
    * If the directory itself is a dataset, evaluates it
    * Otherwise evaluates every valid dataset subdirectory inside it
- Aligns evaluation labels to model outputs by class name
- Prints detailed metrics
- Saves JSON reports per dataset
- Saves confusion JSON + PNG plots in two spaces:
    * full model-output space
    * compact evaluation-dataset space
- Saves grouped evaluation reports per metadata property
- Shows a progress bar during evaluation
"""

import json
import os
import random
import math
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from trainMagicianVisionClassifierTorch import (
    Classifier,
    RGBAImageFolder,
    checkIfFileExists,
    checkIfPathIsDirectory,
    filter_dataset_classes,
    load_hyperparameters,
    metadata_collate_fn,
)

try:
    from DatasetConverter import HDF5Dataset
    HAS_HDF5 = True
except Exception:
    HAS_HDF5 = False

OTHER_TRAINED_CLASSES_LABEL = "OTHER_TRAINED_CLASSES"
DEFAULT_METADATA_KEYS = ["Distance1", "Distance2", "Distance3", "DistanceAverage", "Light1", "Light2", "Light3", "Light4", "Light5", "Light6"]
DEFAULT_BINNED_KEYS = {"Distance1", "Distance2", "Distance3", "DistanceAverage"}
DEFAULT_BIN_SIZE = 25


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_imagefolder_dataset(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    valid_ext = (".png", ".jpg", ".jpeg")
    subdirs = [
        os.path.join(path, x)
        for x in sorted(os.listdir(path))
        if os.path.isdir(os.path.join(path, x))
    ]
    for sd in subdirs:
        try:
            for f in os.listdir(sd):
                if f.lower().endswith(valid_ext):
                    return True
        except Exception:
            pass
    return False


def is_hdf5_dataset(path: str) -> bool:
    return HAS_HDF5 and os.path.isfile(os.path.join(path, "dataset.h5"))


def is_valid_dataset_dir(path: str) -> bool:
    return is_hdf5_dataset(path) or is_imagefolder_dataset(path)


def discover_eval_datasets(eval_root: str):
    datasets = []
    if is_valid_dataset_dir(eval_root):
        datasets.append(eval_root)
        return datasets
    if not os.path.isdir(eval_root):
        return datasets
    for name in sorted(os.listdir(eval_root)):
        full = os.path.join(eval_root, name)
        if os.path.isdir(full) and is_valid_dataset_dir(full):
            datasets.append(full)
    return datasets


def make_transform():
    return transforms.Compose([transforms.ToTensor()])


def load_dataset(dataset_dir: str, config_json: dict):
    transform = make_transform()
    h5_file = os.path.join(dataset_dir, "dataset.h5")
    if os.path.isfile(h5_file):
        if not HAS_HDF5:
            raise RuntimeError(
                f"Found HDF5 dataset at {h5_file}, but DatasetConverter.HDF5Dataset is not available."
            )
        dataset = HDF5Dataset(h5_file)
        print(f"Dataset type: {type(dataset).__name__}")
        if hasattr(dataset, "metadata"):
            print("Dataset has metadata attribute:", dataset.metadata is not None)
    else:
        print("Loading PNG dataset with comment metadata") 
        dataset = RGBAImageFolder(root=dataset_dir, transform=transform, return_metadata=True)

    if (
        "selected_classes" in config_json
        and config_json["selected_classes"] is not None
        and len(config_json["selected_classes"]) > 1
    ):
        filter_dataset_classes(dataset, config_json["selected_classes"])
    return dataset


def print_class_distribution(dataset, title="Dataset"):
    print(f"\n--- {title} Class Distribution ---")
    targets = dataset.targets
    classes = dataset.classes
    counter = Counter(targets)
    total = 0
    for class_idx in sorted(counter.keys()):
        class_name = classes[class_idx]
        count = counter[class_idx]
        total += count
        print(f"Class {class_idx:2d} ({class_name}): {count}")
    print(f"Total samples: {total}")
    print("----------------------------------")


def get_clean_class_id(class_names):
    for i, name in enumerate(class_names):
        if name in ("class_clean", "Clean"):
            return i
    return 0


def instantiate_classifier_from_config(config_json: dict, class_names):
    dropout_rate = config_json["hparams"]["dropout_rate"]
    tile_size = config_json["hparams"]["tile_size"]
    lr = config_json["optimizer"]["learning_rate"]
    penalize_false_clean = float(config_json.get("penalize_false_clean", 0.0))
    base_channels = config_json["hparams"].get("base_channels", 32)
    final_dense_layer = config_json["hparams"].get("final_dense_layer", 512)
    noise_std = config_json["hparams"].get("noise_std", 0.0)
    noise_clip = config_json["hparams"].get("noise_clip", None)

    return Classifier(
        model=config_json["model"],
        loss=config_json.get("loss", "focal"),
        tile_size=tile_size,
        num_classes=len(class_names),
        dropout_rate=dropout_rate,
        lr=lr,
        AoLP=config_json.get("AoLP", False),
        DoLP=config_json.get("DoLP", False),
        Unpolarized=config_json.get("Unpolarized", False),
        MaxPolarization=config_json.get("MaxPolarization", False),
        MinPolarization=config_json.get("MinPolarization", False),
        RangePolarization=config_json.get("RangePolarization", False),
        penalize_false_clean=penalize_false_clean,
        base_channels=base_channels,
        final_dense_layer=final_dense_layer,
        clean_class=get_clean_class_id(class_names),
        noise_std=noise_std,
        noise_clip=noise_clip,
    )


def get_state_dict_from_checkpoint(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def infer_num_outputs_from_state_dict(state_dict: dict):
    candidate_keys = ["model.fc.weight", "model.classifier.weight", "fc.weight", "classifier.weight"]
    for key in candidate_keys:
        if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) >= 2:
            return int(state_dict[key].shape[0]), key
    raise RuntimeError(
        "Could not infer classifier output dimension from checkpoint. "
        f"Tried keys: {candidate_keys}"
    )


def load_checkpoint_weights(classifier: Classifier, checkpoint_path: str, device: str):
    state_dict = get_state_dict_from_checkpoint(checkpoint_path, device)
    missing, unexpected = classifier.load_state_dict(state_dict, strict=False)
    print("\nCheckpoint load summary:")
    print(f"  missing keys   : {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    if missing:
        print("  First missing keys:", missing[:10])
    if unexpected:
        print("  First unexpected keys:", unexpected[:10])
    return classifier


def build_eval_to_model_mapping(eval_class_names, model_class_names):
    model_name_to_idx = {name: i for i, name in enumerate(model_class_names)}
    eval_to_model = {}
    shared_class_names = []
    missing_in_model = []
    for eval_idx, cls_name in enumerate(eval_class_names):
        if cls_name in model_name_to_idx:
            eval_to_model[eval_idx] = model_name_to_idx[cls_name]
            shared_class_names.append(cls_name)
        else:
            missing_in_model.append(cls_name)
    missing_in_eval = [name for name in model_class_names if name not in set(eval_class_names)]
    return eval_to_model, shared_class_names, missing_in_model, missing_in_eval


@torch.no_grad()
def predict_dataset_aligned(
    classifier,
    dataset,
    eval_to_model_class,
    batch_size=16,
    num_workers=0,
    device="cpu",
    progress_desc="Evaluating",
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=metadata_collate_fn,
    )

    classifier.eval()
    classifier.to(device)

    y_true_model_space = []
    y_pred_model_space = []
    y_prob = []
    metadata_rows = []
    skipped_samples = 0

    progress_bar = tqdm(
        loader,
        desc=progress_desc,
        unit="batch",
        leave=True,
        dynamic_ncols=True,
    )

    for x, y, meta_list in progress_bar:
        x = x.to(device)

        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
        y_eval = y.cpu().numpy().tolist()

        batch_used = 0
        batch_skipped = 0

        for i, eval_idx in enumerate(y_eval):
            eval_idx = int(eval_idx)
            if eval_idx not in eval_to_model_class:
                skipped_samples += 1
                batch_skipped += 1
                continue

            gt_model_idx = eval_to_model_class[eval_idx]
            y_true_model_space.append(gt_model_idx)
            y_pred_model_space.append(int(preds[i]))
            y_prob.append(probs[i].detach().cpu().numpy().tolist())

            if i < len(meta_list) and isinstance(meta_list[i], dict):
                metadata_rows.append(meta_list[i])
            else:
                metadata_rows.append({})

            batch_used += 1

        progress_bar.set_postfix(
            used=len(y_true_model_space),
            skipped=skipped_samples,
            batch_used=batch_used,
            batch_skipped=batch_skipped,
        )

    prob_dim = len(y_prob[0]) if y_prob else 0
    return (
        np.array(y_true_model_space, dtype=np.int64),
        np.array(y_pred_model_space, dtype=np.int64),
        np.array(y_prob, dtype=np.float32) if y_prob else np.zeros((0, prob_dim), dtype=np.float32),
        metadata_rows,
        skipped_samples,
    )


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def build_detailed_stats(y_true, y_pred, model_class_names, shared_class_names=None):
    if shared_class_names is None:
        shared_class_names = list(model_class_names)

    labels = [model_class_names.index(name) for name in shared_class_names]

    if len(y_true) == 0:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        cm_norm = np.zeros((len(labels), len(labels)), dtype=float)
        report_text = "No aligned samples available for evaluation."
        per_class = []
        for i, name in enumerate(shared_class_names):
            per_class.append({
                "class_id": int(labels[i]),
                "class_name": name,
                "support": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            })
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "num_samples": 0,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "classification_report": report_text,
            "per_class": per_class,
            "labels": shared_class_names,
            "label_model_indices": labels,
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    acc = accuracy_score(y_true, y_pred)
    try:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        _, recall_tmp, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        bal_acc = float(np.mean(recall_tmp)) if len(recall_tmp) else 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = []
    for i, name in enumerate(shared_class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        class_acc = safe_div(tp, int(cm[i, :].sum()))
        per_class.append({
            "class_id": int(labels[i]),
            "class_name": name,
            "support": int(support[i]),
            "accuracy": class_acc,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })

    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=shared_class_names, digits=4, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "num_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "classification_report": report_text,
        "per_class": per_class,
        "labels": shared_class_names,
        "label_model_indices": labels,
    }


def print_metrics(title, metrics, class_names):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"Samples            : {metrics['num_samples']}")
    print(f"Accuracy           : {metrics['accuracy']:.6f}")
    print(f"Balanced Accuracy  : {metrics['balanced_accuracy']:.6f}")

    print("\nLabels:")
    for i, name in enumerate(class_names):
        model_idx = None
        if "label_model_indices" in metrics and i < len(metrics["label_model_indices"]):
            model_idx = metrics["label_model_indices"][i]
        print(f"  {i:2d}: {name}" + (f" (model index {model_idx})" if model_idx is not None else ""))

    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"], dtype=int))

    print("\nNormalized Confusion Matrix (rows=true class):")
    with np.printoptions(precision=4, suppress=True):
        print(np.array(metrics["confusion_matrix_normalized"], dtype=float))

    print("\nDetailed Classification Report:")
    print(metrics["classification_report"])

    print("Per-class statistics:")
    for row in metrics["per_class"]:
        print(
            f"  [{row['class_id']:2d}] {row['class_name']:<30} "
            f"support={row['support']:5d} "
            f"acc={row['accuracy']:.4f} "
            f"prec={row['precision']:.4f} "
            f"rec={row['recall']:.4f} "
            f"f1={row['f1']:.4f} "
            f"tp={row['tp']:5d} fp={row['fp']:5d} fn={row['fn']:5d} tn={row['tn']:5d}"
        )


def save_json_report(out_path, dataset_path, class_names, metrics, extra=None):
    payload = {"dataset": dataset_path, "classes": class_names, "metrics": metrics}
    if extra is not None:
        payload["extra"] = extra
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved report: {out_path}")


def build_full_model_confusion_matrix(y_true, y_pred, model_class_names):
    num_classes = len(model_class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_idx, pred_idx in zip(y_true, y_pred):
        if 0 <= int(true_idx) < num_classes and 0 <= int(pred_idx) < num_classes:
            matrix[int(true_idx), int(pred_idx)] += 1
    return matrix


def build_eval_dataset_confusion_matrix(y_true, y_pred, eval_class_names, model_class_names):
    eval_model_indices = [model_class_names.index(name) for name in eval_class_names]
    model_to_eval_col = {model_idx: col_idx for col_idx, model_idx in enumerate(eval_model_indices)}
    other_col = len(eval_class_names)
    size = len(eval_class_names) + 1
    matrix = np.zeros((size, size), dtype=int)

    true_model_to_row = {model_idx: row_idx for row_idx, model_idx in enumerate(eval_model_indices)}
    for true_idx, pred_idx in zip(y_true, y_pred):
        true_idx = int(true_idx)
        pred_idx = int(pred_idx)
        if true_idx not in true_model_to_row:
            continue
        row = true_model_to_row[true_idx]
        col = model_to_eval_col.get(pred_idx, other_col)
        matrix[row, col] += 1

    labels = list(eval_class_names) + [OTHER_TRAINED_CLASSES_LABEL]
    return matrix, labels


def write_confusion_json(json_path, title, labels, matrix):
    payload = {"title": title, "labels": labels, "matrix": np.asarray(matrix, dtype=int).tolist()}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved confusion JSON: {json_path}")


def plot_confusion_matrices(base_path_without_ext, title, labels, matrix):
    matrix = np.array(matrix, dtype=float)

    def _save(fig_name, heat_matrix, annot, fmt, plot_title):
        plt.figure(figsize=(max(8, len(labels) * 0.9), max(6, len(labels) * 0.7)))
        sns.heatmap(
            heat_matrix,
            annot=annot,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(plot_title)
        plt.tight_layout()
        plt.savefig(f"{base_path_without_ext}_{fig_name}.png")
        plt.close()

    matrix_int = matrix.astype(int)
    _save("raw", matrix_int, matrix_int, "d", f"{title} - Raw Counts")

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_by_row = np.divide(100.0 * matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    annot_row = np.char.mod("%.1f%%", normalized_by_row)
    _save("row_normalized", normalized_by_row, annot_row, "", f"{title} - Row Normalized (%)")

    total_sum = matrix.sum()
    normalized_total = np.zeros_like(matrix) if total_sum == 0 else (100.0 * matrix) / total_sum
    annot_total = np.char.mod("%.1f%%", normalized_total)
    _save("total_normalized", normalized_total, annot_total, "", f"{title} - Total Normalized (%)")

    hybrid = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    _save("hybrid_row_normalized", hybrid, matrix_int, "d", f"{title} - Raw Counts w/ Row-Normalized Color")


def save_confusion_artifacts(report_root, safe_name, title, suffix, labels, matrix):
    json_path = os.path.join(report_root, f"{safe_name}{suffix}.json")
    base_path = os.path.join(report_root, f"{safe_name}{suffix}")
    write_confusion_json(json_path, title, labels, matrix)
    plot_confusion_matrices(base_path, title, labels, matrix)
    return json_path


def normalize_metadata_value(value):
    if value is None:
        return "MISSING"
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return "MISSING"
        return value
    return str(value)



def bucket_numeric_metadata(value, bin_size=25):
    value = normalize_metadata_value(value)
    try:
        v = float(value)
        lo = int(math.floor(v / bin_size) * bin_size)
        hi = lo + bin_size - 1
        return f"{lo:04d}-{hi:04d}"
    except Exception:
        return str(value)


def transform_metadata_value(key, value, binned_keys=None, bin_size=25):
    if binned_keys is None:
        binned_keys = set(DEFAULT_BINNED_KEYS)
    if key in binned_keys:
        return bucket_numeric_metadata(value, bin_size=bin_size)
    return normalize_metadata_value(value)


def build_metadata_group_reports(
    y_true,
    y_pred,
    metadata_rows,
    model_class_names,
    shared_class_names,
    metadata_keys,
    binned_keys=None,
    bin_size=25,
    min_samples=1,
):
    reports = {}

    if binned_keys is None:
        binned_keys = set(DEFAULT_BINNED_KEYS)

    for key in metadata_keys:
        grouped_indices = defaultdict(list)

        for i, meta in enumerate(metadata_rows):
            meta = meta if isinstance(meta, dict) else {}
            raw_value = meta.get(key, "MISSING")
            group_value = transform_metadata_value(
                key=key,
                value=raw_value,
                binned_keys=binned_keys,
                bin_size=bin_size,
            )
            grouped_indices[str(group_value)].append(i)

        key_reports = {}
        for value in sorted(grouped_indices.keys()):
            indices = grouped_indices[value]
            if len(indices) < min_samples:
                continue

            yt = np.array([y_true[i] for i in indices], dtype=np.int64)
            yp = np.array([y_pred[i] for i in indices], dtype=np.int64)

            stats = build_detailed_stats(
                yt,
                yp,
                model_class_names,
                shared_class_names=shared_class_names,
            )

            key_reports[value] = {
                "num_samples": int(len(indices)),
                "accuracy": float(stats["accuracy"]),
                "balanced_accuracy": float(stats["balanced_accuracy"]),
                "metrics": stats,
            }

        reports[key] = key_reports

    return reports


def print_metadata_group_summary_less_verbose(metadata_group_reports):
    print("\n" + "=" * 90)
    print("METADATA GROUP SUMMARY")
    print("=" * 90)

    for key, groups in metadata_group_reports.items():
        print(f"\n--- {key} ---")
        if not groups:
            print("  No groups found.")
            continue
        for value, payload in groups.items():
            m = payload["metrics"]
            print(
                f"{key}={value:>12} | "
                f"samples={payload['num_samples']:5d} | "
                f"acc={m['accuracy']:.4f} | "
                f"bal_acc={m['balanced_accuracy']:.4f}"
            )


def print_metadata_group_summary(metadata_group_reports):
    print("\n" + "=" * 90)
    print("METADATA GROUP SUMMARY")
    print("=" * 90)

    for key, groups in metadata_group_reports.items():
        print(f"\n--- {key} ---")
        if not groups:
            print("  No groups found.")
            continue

        for value, payload in groups.items():
            m = payload["metrics"]

            print(
                f"{key}={value:>12} | "
                f"samples={payload['num_samples']:5d} | "
                f"acc={m['accuracy']:.4f} | "
                f"bal_acc={m['balanced_accuracy']:.4f}"
            )

            # Per-class stats for this metadata bucket
            for row in m["per_class"]:
                print(
                    f"    class={row['class_name']:<30} "
                    f"support={row['support']:5d} "
                    f"acc={row['accuracy']:.4f} "
                    f"prec={row['precision']:.4f} "
                    f"rec={row['recall']:.4f} "
                    f"f1={row['f1']:.4f}"
                )

def parse_metadata_keys(arg):
    if arg is None:
        return list(DEFAULT_METADATA_KEYS)
    keys = [x.strip() for x in arg.split(",") if x.strip()]
    return keys if keys else list(DEFAULT_METADATA_KEYS)


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def scan_ensemble_models(model_dir: str):
    """Return list of (pth_path, cfg_path) for all valid allclass_*.pth pairs."""
    import zipfile
    pairs = []
    for name in sorted(os.listdir(model_dir)):
        if not (name.startswith("allclass_") and name.endswith(".pth")):
            continue
        base     = name[:-4]
        pth_path = os.path.join(model_dir, f"{base}.pth")
        cfg_path = os.path.join(model_dir, f"{base}.json")
        if not os.path.isfile(cfg_path):
            print(f"[Ensemble] Skipping {name}: no matching .json")
            continue
        if not zipfile.is_zipfile(pth_path):
            print(f"[Ensemble] Skipping {name}: corrupted / not a zip")
            continue
        pairs.append((pth_path, cfg_path))
    return pairs


def load_ensemble(pairs, device):
    """Load every (pth, cfg) pair into an eval-mode Classifier. Returns list of (classifier, class_names)."""
    members = []
    for pth_path, cfg_path in pairs:
        cfg = load_hyperparameters(cfg_path)
        if "classes" not in cfg or not cfg["classes"]:
            print(f"[Ensemble] Skipping {pth_path}: no classes in config")
            continue
        cls_names = list(cfg["classes"])
        clf = instantiate_classifier_from_config(cfg, cls_names)
        clf = load_checkpoint_weights(clf, pth_path, device=device)
        clf.to(device)
        clf.eval()
        print(f"[Ensemble] Loaded {os.path.basename(pth_path)}  ({len(cls_names)} classes)")
        members.append((clf, cls_names))
    return members


@torch.no_grad()
def predict_dataset_ensemble(
    ensemble_members,        # list of (classifier, class_names)
    model_class_names,       # union / reference class list for reporting
    dataset,
    eval_to_model_class,
    batch_size=16,
    device="cpu",
    strict=True,
    progress_desc="Evaluating [ensemble]",
):
    """
    Run every ensemble member on each batch, do majority voting per sample,
    and return arrays in the same shape as predict_dataset_aligned.

    Majority vote: for each sample the winning class is the mode across members.
    If strict=True and a clean-class majority exists, it overrides to clean.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=metadata_collate_fn,
    )

    # Build a mapping from each member's class index → reference model_class index
    # Members may have different class sets.
    ref_name_to_idx = {name: i for i, name in enumerate(model_class_names)}
    clean_ref_idx   = next((i for i, n in enumerate(model_class_names)
                            if n.lower() in ("class_clean", "clean")), None)

    member_maps = []   # per-member: local_idx → ref_idx (or None)
    for _, cls_names in ensemble_members:
        m = {}
        for local_idx, name in enumerate(cls_names):
            ref_idx = ref_name_to_idx.get(name)
            if ref_idx is not None:
                m[local_idx] = ref_idx
        member_maps.append(m)

    y_true_out  = []
    y_pred_out  = []
    y_prob_out  = []
    metadata_out = []
    skipped      = 0
    num_ref      = len(model_class_names)
    M            = len(ensemble_members)

    progress_bar = tqdm(loader, desc=progress_desc, unit="batch", leave=True, dynamic_ncols=True)

    for x, y, meta_list in progress_bar:
        x     = x.to(device)
        B     = x.shape[0]
        # votes[b, ref_cls] = vote count across members
        votes = torch.zeros(B, num_ref, dtype=torch.float32, device=device)

        for (clf, _), m_map in zip(ensemble_members, member_maps):
            logits = clf(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)   # [B]
            for b in range(B):
                local_pred = int(preds[b].item())
                ref_pred   = m_map.get(local_pred)
                if ref_pred is not None:
                    votes[b, ref_pred] += 1.0

        # Majority vote per sample
        if strict and clean_ref_idx is not None:
            clean_majority = votes[:, clean_ref_idx] > (M / 2)
            voted = torch.argmax(votes, dim=1)
            voted[clean_majority] = clean_ref_idx
        else:
            voted = torch.argmax(votes, dim=1)

        voted_np = voted.cpu().numpy()
        y_eval   = y.cpu().numpy().tolist()
        # Build simple prob vector from vote fractions for compatibility
        prob_np  = (votes / votes.sum(dim=1, keepdim=True).clamp(min=1)).cpu().numpy()

        for i, eval_idx in enumerate(y_eval):
            eval_idx = int(eval_idx)
            if eval_idx not in eval_to_model_class:
                skipped += 1
                continue
            gt_ref = eval_to_model_class[eval_idx]
            y_true_out.append(gt_ref)
            y_pred_out.append(int(voted_np[i]))
            y_prob_out.append(prob_np[i].tolist())
            if i < len(meta_list) and isinstance(meta_list[i], dict):
                metadata_out.append(meta_list[i])
            else:
                metadata_out.append({})

        progress_bar.set_postfix(used=len(y_true_out), skipped=skipped)

    prob_dim = len(y_prob_out[0]) if y_prob_out else num_ref
    return (
        np.array(y_true_out,  dtype=np.int64),
        np.array(y_pred_out,  dtype=np.int64),
        np.array(y_prob_out,  dtype=np.float32) if y_prob_out else np.zeros((0, prob_dim), dtype=np.float32),
        metadata_out,
        skipped,
    )


# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier (or ensemble) on one or more datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single-model (original usage):
  python3 evaluateClassifierNew.py model.pth config.json /path/to/eval_dir [batch] [meta_csv] [bin_size]

Ensemble mode:
  python3 evaluateClassifierNew.py --ensemble /path/to/model_dir /path/to/eval_dir [batch] [meta_csv] [bin_size]
  python3 evaluateClassifierNew.py --ensemble /path/to/model_dir /path/to/eval_dir 16 Distance1,Light1 25
        """,
    )
    parser.add_argument("--ensemble", metavar="MODEL_DIR",
                        help="Directory containing allclass_*.pth + .json pairs. "
                             "Enables ensemble majority-vote mode.")
    # Positional args work for both modes; in ensemble mode pos[0] is eval_root
    parser.add_argument("args", nargs="*")
    parsed = parser.parse_args()

    ensemble_mode = parsed.ensemble is not None

    if ensemble_mode:
        # --ensemble MODEL_DIR  eval_root  [batch] [meta] [bin]
        pos = parsed.args
        if len(pos) < 1:
            parser.error("--ensemble mode requires at least: eval_root")
        model_dir      = parsed.ensemble
        eval_root      = pos[0]
        batch_size     = int(pos[1]) if len(pos) > 1 else 16
        metadata_keys  = parse_metadata_keys(pos[2]) if len(pos) > 2 else list(DEFAULT_METADATA_KEYS)
        bin_size       = int(pos[3]) if len(pos) > 3 else DEFAULT_BIN_SIZE
        checkpoint_path = None
        config_path     = None
    else:
        # Legacy positional: model.pth config.json eval_root [batch] [meta] [bin]
        pos = parsed.args
        if len(pos) < 3:
            parser.print_help()
            sys.exit(1)
        checkpoint_path = pos[0]
        config_path     = pos[1]
        eval_root       = pos[2]
        batch_size      = int(pos[3]) if len(pos) > 3 else 16
        metadata_keys   = parse_metadata_keys(pos[4]) if len(pos) > 4 else list(DEFAULT_METADATA_KEYS)
        bin_size        = int(pos[5]) if len(pos) > 5 else DEFAULT_BIN_SIZE
        model_dir       = None

    binned_keys = set(DEFAULT_BINNED_KEYS)

    # --- Validate paths ---
    if ensemble_mode:
        if not checkIfPathIsDirectory(model_dir):
            print(f"Ensemble model directory not found: {model_dir}")
            sys.exit(1)
    else:
        if not checkIfFileExists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        if not checkIfFileExists(config_path):
            print(f"Config not found: {config_path}")
            sys.exit(1)

    if not checkIfPathIsDirectory(eval_root):
        print(f"Evaluation directory not found: {eval_root}")
        sys.exit(1)

    device = default_device()
    print(f"Using device     : {device}")
    print(f"Mode             : {'ENSEMBLE' if ensemble_mode else 'single-model'}")
    print(f"Metadata keys    : {metadata_keys}")
    print(f"Binned keys      : {sorted(binned_keys)}")
    print(f"Bin size         : {bin_size}")

    # --- Load model(s) ---
    if ensemble_mode:
        pairs = scan_ensemble_models(model_dir)
        if not pairs:
            print(f"No valid allclass_*.pth models found in: {model_dir}")
            sys.exit(1)
        print(f"\nFound {len(pairs)} ensemble members:")
        for pth, cfg in pairs:
            print(f"  {os.path.basename(pth)}")

        ensemble_members = load_ensemble(pairs, device)
        if not ensemble_members:
            print("No ensemble members loaded successfully.")
            sys.exit(1)

        # Union of all class names across members, preserving order of first occurrence
        seen = {}
        for _, cls_names in ensemble_members:
            for n in cls_names:
                if n not in seen:
                    seen[n] = len(seen)
        model_class_names = list(seen.keys())
        print(f"\nUnion class set ({len(model_class_names)} classes): {model_class_names}")

        # Seed from first member's config (best effort)
        first_cfg = load_hyperparameters(pairs[0][1])
        seed_everything(int(first_cfg["hparams"].get("seed", 1234)))
        model_name = f"ensemble_{len(ensemble_members)}models"
        tile_size  = int(first_cfg["hparams"].get("tile_size", 0))
        epochs     = 0
        config_json = first_cfg   # used only for load_dataset below
    else:
        config_json = load_hyperparameters(config_path)
        seed_everything(int(config_json["hparams"].get("seed", 1234)))

        model_class_names = list(config_json["classes"])
        print(f"\nModel classes: {model_class_names}")

        state_dict = get_state_dict_from_checkpoint(checkpoint_path, device)
        checkpoint_num_outputs, checkpoint_head_key = infer_num_outputs_from_state_dict(state_dict)
        print(f"Checkpoint head: {checkpoint_head_key} -> {checkpoint_num_outputs} outputs")

        if len(model_class_names) != checkpoint_num_outputs:
            raise ValueError(
                f"Mismatch between config classes ({len(model_class_names)}) "
                f"and checkpoint outputs ({checkpoint_num_outputs})."
            )

        classifier = instantiate_classifier_from_config(config_json, model_class_names)
        classifier = load_checkpoint_weights(classifier, checkpoint_path, device=device)
        classifier.to(device)
        classifier.eval()

        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        tile_size  = int(config_json["hparams"].get("tile_size", 0))
        epochs     = int(config_json.get("epochs", 0))

    aggregate_true = []
    aggregate_pred = []
    aggregate_label_names = set()

    report_root = os.path.join(os.getcwd(), "evaluation_reports")
    os.makedirs(report_root, exist_ok=True)

    eval_datasets = discover_eval_datasets(eval_root)
    if not eval_datasets:
        print(f"No valid datasets found under: {eval_root}")
        sys.exit(1)
    print(f"\nDatasets to evaluate ({len(eval_datasets)}):")
    for d in eval_datasets:
        print(f"  {d}")

    for dataset_dir in eval_datasets:
        dataset = load_dataset(dataset_dir, config_json)
        safe_name = os.path.basename(os.path.normpath(dataset_dir))
        print_class_distribution(dataset, title=dataset_dir)

        eval_to_model_class, shared_class_names, missing_in_model, missing_in_eval = build_eval_to_model_mapping(
            dataset.classes, model_class_names
        )

        print(f"\nEval dataset classes : {dataset.classes}")
        print(f"Shared class names   : {shared_class_names}")
        print(f"Class mapping        : {eval_to_model_class}")

        if missing_in_model:
            print(f"WARNING: eval classes not found in model and skipped: {missing_in_model}")
        if missing_in_eval:
            print(f"Model-only classes not present in this eval dataset: {missing_in_eval}")
        if not shared_class_names:
            print(f"Skipping {dataset_dir}: no shared class names with trained model")
            continue

        if ensemble_mode:
            y_true, y_pred, _, metadata_rows, skipped_samples = predict_dataset_ensemble(
                ensemble_members=ensemble_members,
                model_class_names=model_class_names,
                dataset=dataset,
                eval_to_model_class=eval_to_model_class,
                batch_size=batch_size,
                device=device,
                progress_desc=f"Evaluating {safe_name}",
            )
        else:
            y_true, y_pred, _, metadata_rows, skipped_samples = predict_dataset_aligned(
                classifier=classifier,
                dataset=dataset,
                eval_to_model_class=eval_to_model_class,
                batch_size=batch_size,
                num_workers=0,
                device=device,
                progress_desc=f"Evaluating {safe_name}",
            )

        print(f"Aligned samples used : {len(y_true)}")
        print(f"Skipped samples      : {skipped_samples}")

        metrics = build_detailed_stats(y_true, y_pred, model_class_names, shared_class_names=shared_class_names)
        print_metrics(f"Evaluation on: {dataset_dir}", metrics, shared_class_names)

        metadata_group_reports = build_metadata_group_reports(
            y_true=y_true,
            y_pred=y_pred,
            metadata_rows=metadata_rows,
            model_class_names=model_class_names,
            shared_class_names=shared_class_names,
            metadata_keys=metadata_keys,
            binned_keys=binned_keys,
            bin_size=bin_size,
            min_samples=1,
        )
        print_metadata_group_summary(metadata_group_reports)

        aggregate_true.extend(y_true.tolist())
        aggregate_pred.extend(y_pred.tolist())
        aggregate_label_names.update(shared_class_names)

        full_matrix = build_full_model_confusion_matrix(y_true, y_pred, model_class_names)
        full_title = f"{model_name} / Tile Size = {tile_size} / Epochs = {epochs} / {safe_name}"
        full_confusion_json = save_confusion_artifacts(
            report_root,
            safe_name,
            full_title,
            "_confusion",
            model_class_names,
            full_matrix,
        )

        eval_matrix, eval_labels = build_eval_dataset_confusion_matrix(
            y_true, y_pred, shared_class_names, model_class_names
        )
        eval_title = f"{model_name} / Tile Size = {tile_size} / Epochs = {epochs} / {safe_name} / Eval Classes"
        eval_confusion_json = save_confusion_artifacts(
            report_root,
            safe_name,
            eval_title,
            "_eval_confusion",
            eval_labels,
            eval_matrix,
        )

        metadata_report_paths = {}
        for key, payload in metadata_group_reports.items():
            out_path = os.path.join(report_root, f"{safe_name}_metadata_{key}.json")
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            metadata_report_paths[key] = out_path
            print(f"Saved metadata report: {out_path}")

        out_json = os.path.join(report_root, f"{safe_name}_evaluation.json")
        save_json_report(
            out_json,
            dataset_dir,
            shared_class_names,
            metrics,
            extra={
                "model_classes": model_class_names,
                "eval_classes": dataset.classes,
                "shared_classes": shared_class_names,
                "missing_in_model": missing_in_model,
                "missing_in_eval": missing_in_eval,
                "skipped_samples": skipped_samples,
                "eval_to_model_class": {str(k): int(v) for k, v in eval_to_model_class.items()},
                "full_confusion_json": full_confusion_json,
                "eval_confusion_json": eval_confusion_json,
                "eval_confusion_labels": eval_labels,
                "eval_confusion_matrix": eval_matrix.tolist(),
                "metadata_keys": metadata_keys,
                "binned_metadata_keys": sorted(list(binned_keys)),
                "metadata_bin_size": bin_size,
                "metadata_group_reports": metadata_group_reports,
                "metadata_report_paths": metadata_report_paths,
            },
        )

    if not aggregate_true:
        print("\nNo aligned evaluation samples were found across all datasets.")
        sys.exit(0)

    if len(eval_datasets) > 1:
        aggregate_shared_class_names = [name for name in model_class_names if name in aggregate_label_names]
        agg_metrics = build_detailed_stats(
            np.array(aggregate_true, dtype=np.int64),
            np.array(aggregate_pred, dtype=np.int64),
            model_class_names,
            shared_class_names=aggregate_shared_class_names,
        )
        print_metrics("AGGREGATE EVALUATION (all datasets)", agg_metrics, aggregate_shared_class_names)
        out_json = os.path.join(report_root, "aggregate_evaluation.json")
        save_json_report(
            out_json,
            eval_root,
            aggregate_shared_class_names,
            agg_metrics,
            extra={
                "model_classes": model_class_names,
                "aggregated_shared_classes": aggregate_shared_class_names,
            },
        )


if __name__ == "__main__":
    main()
