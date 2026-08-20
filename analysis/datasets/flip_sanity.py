#!/usr/bin/env python3
"""
Sanity-check the polarization-aware flip augmentation on static-camera data
(measure65mmheight_*). Physics: mirroring a scene about a vertical axis maps
AoLP theta -> -theta, i.e. Stokes S1 = I0-I90 is invariant while S2 = I45-I135
NEGATES. So with a (roughly) left-right symmetric static scene:
  corr(S1, mirror(S1)) > 0   (control: measures how symmetric the scene is)
  corr(S2, mirror(S2)) < 0   (confirms flip must swap the 45/135 channels)
If instead corr(S2, mirror(S2)) is as positive as the control, a plain flip
without channel swap would be the right augmentation (or channels mislabeled).
Same logic applies to vertical mirroring.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ammar/Documents/Programming/magician_grabber_annotator")
from mvc.core.read_data import readPolarPNMToRGBA  # ch0=p0, ch1=p45, ch2=p90, ch3=p135


def masked_corr(a, b, mask):
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


for dset in sorted(glob.glob("/media/ammar/games2/Datasets/Magician/measure65mmheight_*")):
    frames = sorted(glob.glob(os.path.join(dset, "colorFrame_0_*.p??")))
    frames = [f for f in frames if not f.endswith(".json")][:40]
    if not frames:
        continue
    acc, n = None, 0
    for f in frames:
        img = readPolarPNMToRGBA(f)
        if img is None:
            continue
        img = img.astype(np.float32)
        acc = img if acc is None else acc + img
        n += 1
    M = acc / n                                # temporal mean, HxWx4
    p0, p45, p90, p135 = M[..., 0], M[..., 1], M[..., 2], M[..., 3]
    S0 = (p0 + p45 + p90 + p135) / 2.0
    S1 = p0 - p90
    S2 = p45 - p135
    mask = S0 > 40                             # ignore dark background

    for axis, name in ((1, "horizontal-mirror"), (0, "vertical-mirror")):
        mir = lambda x: np.flip(x, axis=axis)
        m2 = mask & mir(mask)
        c_s1 = masked_corr(S1, mir(S1), m2)
        c_s2 = masked_corr(S2, mir(S2), m2)
        c_s0 = masked_corr(S0, mir(S0), m2)
        print(f"{os.path.basename(dset):24s} {name:18s} n={n:2d} "
              f"corr(S0)={c_s0:+.3f} (scene symmetry)  "
              f"corr(S1)={c_s1:+.3f} (expect +)  corr(S2)={c_s2:+.3f} (expect -)")
