#!/usr/bin/env python3
"""
Headless driver for datasetCreator.ProcessorThread (Direct H5 mode, no PNG tiles).

Usage:
  python3 headless_dump.py <target_dir> <dataset_dir> [<dataset_dir> ...] [--use-severity 0|1]

Runs the exact same pipeline as the GUI "Start" button with Direct H5 checked,
using the GUI's default processing options (tile 48, step 4, ratio 10, etc.),
but synchronously and without a display.
"""
import os
import sys

ANNOTATOR = "/home/ammar/Documents/Programming/magician_grabber_annotator"
sys.path.insert(0, ANNOTATOR)
os.chdir(ANNOTATOR)

import wx  # noqa: E402  (imported by datasetCreator anyway)
# No wx.App in headless mode: make CallAfter a no-op (only feeds GUI gauges).
wx.CallAfter = lambda *a, **k: None

import datasetCreator  # noqa: E402
from datasetCreator import ProcessorThread  # noqa: E402


def grab_opt(args, name, default, cast):
    if name in args:
        i = args.index(name)
        v = cast(args[i + 1])
        del args[i:i + 2]
        return v
    return default


def main():
    args = sys.argv[1:]
    use_severity = bool(grab_opt(args, "--use-severity", 1, int))
    ratio_clean = grab_opt(args, "--ratio-clean", 10, float)
    defect_tiles = grab_opt(args, "--defect-tiles", 32, int)
    tile_size = grab_opt(args, "--tile-size", 48, int)
    if bool(grab_opt(args, "--center-defect", 0, int)):
        import readDataAnnotator as _rda
        _rda.CENTER_DEFECT = True
        print("CENTER_DEFECT enabled: one defect tile per point, defect at tile centre")
    # Stream over-acceptance multiplier for cleans (final ratio is enforced at
    # close). Lowering it caps the pre-subsample file size on disk.
    oversample = grab_opt(args, "--clean-oversample", None, float)
    if oversample is not None:
        datasetCreator.CLEAN_STREAM_OVERSAMPLE = oversample
        print("CLEAN_STREAM_OVERSAMPLE =", oversample)

    # --canonical-light 1: remap every frame's strobed light to light #0 before
    # tiling (mirror of wxAnnotator._canonicalizeLighting, but on the 4-channel
    # image loadImageAndJSON consumes). Per-dataset exemplars come from the
    # first 6 frames = one clean strobe cycle; remap = per-channel gains
    # exemplar0/exemplarK picked by nearest normalized channel-mean signature.
    if bool(grab_opt(args, "--canonical-light", 0, int)):
        import glob as _glob
        import numpy as _np
        import readData as _rd
        _orig_read = _rd.readPolarPNMToRGBA
        _exemplars = {}

        def _bootstrap(dirpath):
            frames = sorted(_glob.glob(os.path.join(dirpath, "colorFrame_0_*.pnm")) +
                            _glob.glob(os.path.join(dirpath, "colorFrame_0_*.png")))[:6]
            ex = []
            for f in frames:
                im = _orig_read(f)
                if im is None or im.ndim != 3 or im.shape[2] != 4:
                    continue
                m = im.reshape(-1, 4).mean(axis=0).astype(_np.float64)
                ex.append({"means": _np.maximum(m, 1e-6),
                           "sig": m / max(float(m.sum()), 1e-6)})
            ok = len(ex) == 6
            print(f"[CanonicalLight] {os.path.basename(dirpath)}: "
                  f"{'6 exemplars' if ok else 'BOOTSTRAP FAILED - passthrough'}")
            return ex if ok else None

        def _canonical_read(path):
            img = _orig_read(path)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                return img
            d = os.path.dirname(path)
            if d not in _exemplars:
                _exemplars[d] = _bootstrap(d)
            ex = _exemplars[d]
            if not ex:
                return img
            m = img.reshape(-1, 4).mean(axis=0).astype(_np.float64)
            sig = m / max(float(m.sum()), 1e-6)
            k = int(_np.argmin([_np.linalg.norm(sig - e["sig"]) for e in ex]))
            if k == 0:
                return img
            gains = (ex[0]["means"] / ex[k]["means"]).astype(_np.float32)
            maxv = 65535.0 if img.dtype == _np.uint16 else 255.0
            return _np.clip(img.astype(_np.float32) * gains, 0, maxv).astype(img.dtype)

        _rd.readPolarPNMToRGBA = _canonical_read
        print("CANONICAL LIGHT REMAP ENABLED (all frames rendered as light #0)")

    target_dir = args[0]
    dataset_dirs = args[1:]
    for d in dataset_dirs:
        assert os.path.isdir(d), d

    os.makedirs(target_dir, exist_ok=True)

    ui = {k: (lambda *a, **kw: None) for k in
          ("on_dataset_start", "on_progress_update", "on_file_done",
           "on_dataset_done", "on_all_done")}

    # GUI defaults (datasetCreator MainFrame initial values), Direct H5 on.
    proc = ProcessorThread(
        dataset_dirs, target_dir, ui,
        ratio_clean=ratio_clean,
        threshold=20,
        border=0,
        step=4,
        tile_size=tile_size,
        ignoreSamplesWithNoMetadata=False,   # keep datasets without controller.csv
        includeTilesNotAnnotated=False,
        includeTilesAnnotatedByAI=True,
        use_severity=use_severity,
        use_clean_class=True,
        write_h5=True,
        quarantine_px=32,
        defect_tiles_per_point=defect_tiles,
    )
    proc.run()  # synchronous
    print("HEADLESS DUMP DONE ->", os.path.join(target_dir, "dataset.h5"))


if __name__ == "__main__":
    main()
