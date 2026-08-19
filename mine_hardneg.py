#!/usr/bin/env python3
"""Hard-negative mining on guaranteed-clean folders.

Run the best model over certified, defect-free folders that were EXCLUDED from the
training set (calibration patterns, chessboards, clean car passes) and harvest every
48x48x4 tile the model scores as a defect. Those folders contain no real defects, so
every such tile is a FALSE POSITIVE -- exactly the hard negative that lowers the
false-alarm half of the KPI. The mined tiles are written as class_clean and added as a
second training_dataset entry (no re-dump of the main set).

SAFETY GATES (a tile is only ever produced from a provably clean frame):
  folder gate : info.json must have total_defects == 0 and empty defect_counts.
  frame  gate : the image must have a sibling .json, and that json must declare NO
                defect points (empty pointClicks AND regionClicks). Any point -> the
                whole frame is skipped and the classifier is NOT run on it.

Tiles are built byte-identically to the H5 dump (readPolarPNMToRGBA -> RGBA->BGRA[2,1,0,3]
-> CHW, kept uint8; /255 + channel derivation happen inside the model on the GPU), so the
model sees exactly what it saw in training.

Usage:
  python mine_hardneg.py <config.json> [--checkpoint X.ckpt] [--out hardneg.h5]
      [--threshold 0.9] [--step 24] [--min-value 30] [--batch 512]
      [--data-root DIR] [--folders A B C ...]
The config supplies the model + channel flags + checkpoint_dir + training_dataset (for the
class space), exactly like eval_coverage.py.
"""
import os, sys, json, glob, argparse
import numpy as np
import torch

ANNOTATOR = "/home/ammar/Documents/Programming/magician_grabber_annotator"
sys.path.insert(0, ANNOTATOR)
import wx  # noqa: E402  (datasetCreator imports it)
wx.CallAfter = lambda *a, **k: None
import readDataAnnotator as rda          # noqa: E402
from datasetCreator import H5DatasetWriter  # noqa: E402

from Config import load_hyperparameters   # noqa: E402
from ClassScheme import apply_class_scheme  # noqa: E402
from DatasetConverter import HDF5Dataset  # noqa: E402
from LitClassifier import Classifier      # noqa: E402

# Verified certified, defect-free, operational-exposure folders excluded from Aug26_78K.
DEFAULT_FOLDERS = [
    "AltinayCarSlideFromPattern",         # real car surface, clean (exp 450)
    "AltinayUniquePattern750",
    "AltinayUniquePattern750_2",
    "AltinayemulateCalibration",
    "AltinayCleanOnlyemulateCalibration",
    "Altinaychessboardbase",              # exp 350 (a bit dim)
    "Altinayuniquepatternmirror",
]


def folder_is_clean(folder):
    """Folder gate: info.json says total_defects == 0 and no defect_counts."""
    info = os.path.join(folder, "info.json")
    if not os.path.isfile(info):
        return False, "no info.json"
    try:
        j = json.load(open(info))
    except Exception as e:
        return False, f"unreadable info.json ({e})"
    if j.get("total_defects", 0) != 0 or j.get("defect_counts"):
        return False, f"total_defects={j.get('total_defects')} defect_counts={j.get('defect_counts')}"
    return True, "clean"


def frame_is_clean(png):
    """Frame gate: sibling json exists and declares no defect points. Returns
    (ok, reason). The classifier must NOT run on a frame that fails this."""
    jf = os.path.splitext(png)[0] + ".json"
    if not os.path.isfile(jf):
        return False, "no json"
    try:
        j = json.load(open(jf))
    except Exception:
        return False, "bad json"
    if j.get("pointClicks") or j.get("regionClicks"):
        return False, "has defect points"
    return True, "clean"


def build_model(cfg, ckpt):
    """Mirror eval_coverage.py: class space from the training h5 + run's scheme,
    Classifier rebuilt with the run's channel flags, checkpoint loaded."""
    train_dir = cfg["training_dataset"]
    train_dir = train_dir[0] if isinstance(train_dir, list) else train_dir
    ds = HDF5Dataset(f"{train_dir}/dataset.h5")
    ds.metadata = None
    apply_class_scheme(ds, cfg, label="mine")
    classes = list(ds.classes)
    ds.file.close()
    # Exact match: 'class_clean' -- NOT a substring test, which would wrongly match
    # 'class_WeldingClassAClean' (a vestigial class the model never predicts, so every
    # tile would look like a defect and 100% would be "mined").
    clean_id = next((i for i, c in enumerate(classes)
                     if c == "class_clean" or c.lower() == "clean"), None)
    if clean_id is None:
        raise SystemExit(f"no exact class_clean in class space: {classes}")

    # Single source of truth (LitClassifier.Classifier.from_config). This block had all
    # seven derived-channel flags and timm_stem_stride, but not the CustomCNN ladder
    # (custom_channels/custom_res_blocks/custom_wavelet_pools/custom_wavelet_stem), so
    # mining with a custom-CNN run died in load_state_dict on a conv1 shape mismatch.
    # pretrained=False: strict load_state_dict below overwrites every parameter, so
    # fetching ImageNet weights first is a download for nothing.
    model = Classifier.from_config(cfg, num_classes=len(classes),
                                   clean_class=clean_id, pretrained=False)
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    return model, dev, clean_id, classes


def slide(rgba, ts, step, min_value):
    """Every ts x ts window (top-left grid, step) that isn't essentially background."""
    H, W = rgba.shape[:2]
    wins, coords = [], []
    for y in range(0, H - ts + 1, step):
        for x in range(0, W - ts + 1, step):
            t = rgba[y:y + ts, x:x + ts, :]
            if int(t.max()) < min_value:      # background/black -> skip (as tileImages does)
                continue
            wins.append(t)
            coords.append((x, y))
    return wins, coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="hardneg.h5")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="mine a tile if defect_mass = 1 - P(clean) >= this (default 0.95)")
    ap.add_argument("--step", type=int, default=24)
    ap.add_argument("--min-value", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--data-root", default="/media/ammar/games2/Datasets/Magician")
    ap.add_argument("--folders", nargs="*", default=None,
                    help="folder names under data-root (default: the verified clean pool)")
    args = ap.parse_args()

    cfg = load_hyperparameters(args.config)
    ts = cfg["hparams"]["tile_size"]
    ckpt = args.checkpoint
    if ckpt is None:
        cks = sorted(glob.glob(os.path.join(cfg["checkpoint_dir"], "*.ckpt")))
        assert cks, f"no checkpoint in {cfg['checkpoint_dir']}"
        ckpt = cks[-1]
    print(f"[mine] model={cfg['model']} ckpt={os.path.basename(ckpt)} "
          f"tile={ts} step={args.step} threshold={args.threshold}")

    model, dev, clean_id, classes = build_model(cfg, ckpt)
    print(f"[mine] {len(classes)} classes, clean_id={clean_id} "
          f"(mono={cfg['hparams'].get('monochrome')} DoLP={cfg['hparams'].get('DoLP')} "
          f"AoLP={cfg['hparams'].get('AoLP')})")

    folders = args.folders or DEFAULT_FOLDERS
    writer = H5DatasetWriter(args.out)

    n_frames = n_skip_frame = n_skip_folder = n_windows = n_mined = 0
    for name in folders:
        folder = os.path.join(args.data_root, name)
        ok, why = folder_is_clean(folder)
        if not ok:
            print(f"[skip folder] {name}: {why}")
            n_skip_folder += 1
            continue
        pngs = sorted(glob.glob(os.path.join(folder, "*.png")) +
                      glob.glob(os.path.join(folder, "*.jpg")))
        mined_here = 0
        for png in pngs:
            fok, _ = frame_is_clean(png)
            if not fok:
                n_skip_frame += 1
                continue
            rgba = rda.readPolarPNMToRGBA(png)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                n_skip_frame += 1
                continue
            n_frames += 1
            wins, coords = slide(rgba, ts, args.step, args.min_value)
            n_windows += len(wins)
            for i in range(0, len(wins), args.batch):
                chunk = wins[i:i + args.batch]
                cc = coords[i:i + args.batch]
                # RGBA HWC -> BGRA CHW uint8, exactly as H5DatasetWriter.add stores it.
                arr = np.stack([np.ascontiguousarray(
                    np.transpose(w[:, :, [2, 1, 0, 3]], (2, 0, 1))) for w in chunk])
                x = torch.from_numpy(arr).to(dev)
                with torch.no_grad():
                    p_clean = torch.softmax(model(x), dim=1)[:, clean_id]
                mass = (1.0 - p_clean).cpu().numpy()
                for w, (cx, cy), m in zip(chunk, cc, mass):
                    if m >= args.threshold:
                        writer.add(w, "clean",
                                   {"source": f"{png}({cx},{cy})", "defect_mass": float(m)})
                        n_mined += 1
                        mined_here += 1
        print(f"[folder] {name}: {mined_here:,} tiles mined")

    writer.close()
    print(f"\n[done] frames used={n_frames:,} (skipped {n_skip_frame:,} frames, "
          f"{n_skip_folder} folders) windows scored={n_windows:,}")
    print(f"[done] MINED {n_mined:,} hard-negative tiles -> {args.out} "
          f"(all class_clean, threshold {args.threshold})")


if __name__ == "__main__":
    main()
