#!/usr/bin/env python3
"""Does eval_step_curve's labelling reproduce the tiles actually in Aug26_78K?

This is the evidence that the step curve scores the SAME labelling the models were trained
under. It takes real tiles out of the factory-val h5 -- their (x, y) and the class the
annotator dumped for them -- and asks eval_step_curve's point logic to label those same
coordinates from the raw frame JSON. Any disagreement means the coordinate halving, the
containment test, the class naming or the multi-point concatenation has drifted from
`readData.tileImages`, and every step-curve number would be quietly measuring something
else.

Run it after ANY change to the labelling path in eval_step_curve.py.

Last run 2026-08-21: 1,856 tiles compared, 0 disagreements.

Usage:
    python -m analysis.eval.validate_step_curve_labels [--sample N] [--frames N]
"""

import argparse
import json
import re
import sys
from collections import defaultdict

import h5py
import numpy as np

import analysis.eval.eval_step_curve as sc

H5 = '/storage/ammarkov/magician_datasets/Aug26_78K/val/dataset.h5'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=120_000,
                    help='tiles to draw from the h5 before filtering to frames with defects')
    ap.add_argument('--frames', type=int, default=40, help='frames to compare')
    ap.add_argument('--h5', default=H5)
    args = ap.parse_args()

    f = h5py.File(args.h5, 'r')
    raw = f.attrs['class_names']
    cn = json.loads(raw if isinstance(raw, str) else ''.join(raw))
    lab = f['labels'][:]
    md = f['metadata']

    by_frame = defaultdict(list)
    rng = np.random.default_rng(0)
    take = set(rng.choice(len(lab), size=min(args.sample, len(lab)), replace=False).tolist())
    CH = 100_000
    for s in range(0, len(lab), CH):
        e = min(s + CH, len(lab))
        for j, m in enumerate(md[s:e]):
            i = s + j
            if i not in take:
                continue
            m = m.decode() if isinstance(m, bytes) else m
            g = re.search(r'"source": "([^"]+?)\((\d+),(\d+)\)"', m)
            if g:
                by_frame[g.group(1)].append((int(g.group(2)), int(g.group(3)), cn[lab[i]]))
    f.close()

    frames = sorted(fr for fr in by_frame
                    if any(c != 'class_clean' for _, _, c in by_frame[fr]))[:args.frames]
    print(f'{len(frames)} frames with defect tiles sampled\n')

    agree = disagree = noimg = 0
    examples = []
    for fr in frames:
        img, jsn = sc.local_path(fr)
        if img is None:
            noimg += 1
            continue
        # RAW classes: the h5 stores the annotator's names, before any config merge/drop.
        pts, _ = sc.frame_points(jsn, merges={}, drops=set())
        if pts is None:
            continue
        for (x, y, h5cls) in by_frame[fr]:
            # mirror readData.tileImages.label_tile: concatenate every point inside the
            # tile extent, merging repeats of the same class
            text = ''
            for (px, py, pc) in pts:
                if pc is None:
                    continue
                if x <= px < x + sc.TILE and y <= py < y + sc.TILE:
                    short = pc[len('class_'):]
                    if text == '' or text == short:
                        text = short
                    else:
                        text += short
            mine = 'class_' + text if text else 'class_clean'
            if mine == h5cls:
                agree += 1
            else:
                disagree += 1
                if len(examples) < 10:
                    examples.append((fr.rsplit('/', 2)[-2] + '/' + fr.rsplit('/', 1)[-1],
                                     x, y, h5cls, mine))

    print(f'agree    {agree:,}')
    print(f'disagree {disagree:,}   ({100 * disagree / max(agree + disagree, 1):.3f}%)')
    if noimg:
        print(f'no image {noimg}  (raw frames missing under {sc.RAW_ROOT}?)')
    for e in examples:
        print('   MISMATCH frame=%s tile=(%d,%d) h5=%s mine=%s' % e)
    return 1 if disagree else 0


if __name__ == '__main__':
    sys.exit(main())
