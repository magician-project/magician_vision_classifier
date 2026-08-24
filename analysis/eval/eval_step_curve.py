#!/usr/bin/env python3
"""Detection as a function of the DEPLOYMENT TILING STEP, scored on raw frames.

WHY THIS EXISTS
---------------
Every accuracy number in this campaign is PER-TILE, measured on tiles that were already
extracted into the Aug26_78K h5. Per-tile scores are independent of the deployment step --
`eval_coverage.py` never tiles anything, it reads tiles that exist. So the campaign can say
`convnext_tiny` is +3.51 coverage over the incumbent, and it can say `convnext_tiny` reaches
23 Hz at step 18 while the incumbent reaches it at step 16, but it CANNOT say which of the
two catches more defects on a live frame. That question is the whole recommendation, and
nothing in the repo answered it.

What changes with the step is SPATIAL SAMPLING. A defect at (x, y) is seen only by tiles
whose 48 px extent contains it, and on a step-S grid there are about (48/S)^2 of those --
9 at step 16, 7 at step 18, 4 at step 24, and at step >= 48 there may be exactly one or, for
a point near a grid seam, none at all. Coarser tiling therefore costs detection in a way no
per-tile metric can see, and it costs it worst on small, weak defects: precisely the
PositiveDent B/C classes that bind the KPI.

WHY IT NEEDS RAW FRAMES AND CANNOT BE FAKED FROM THE H5
-------------------------------------------------------
The obvious cheap version -- subsample the existing h5 onto a step-S grid -- does not work.
The dump's clean tiles sit on a 48 px grid, but its DEFECT tiles are sampled densely around
each annotation point (`defect_tiles_per_point=8` at `step=4` offsets), so they land at
essentially arbitrary coordinates: measured on the factory val, only 0.5% of defect tiles
have both x and y on a step-16 grid. There is no step-S sliding window hiding inside that
data. So this reads the raw frames and tiles them itself.

THE MEASUREMENT
---------------
Per-DEFECT-POINT detection at a matched false-alarm budget. For every annotated defect
point, take the maximum `defect_mass = 1 - P(clean)` over the tiles of the step-S grid that
CONTAIN it; the point is detected if that maximum clears the threshold. A point no tile
contains is a miss with no forward pass at all -- which is exactly the density effect this
script exists to quantify, and why the metric is per point rather than per tile.

The threshold is matched to a 5% false-alarm rate on the SAME grid's clean tiles, the same
way `miss@FA5` and `eval_coverage.py` match theirs, so a step-16 number and a step-24 number
are on the same footing and both are on the same footing as the rest of the campaign.

LABELLING IS COPIED FROM THE ANNOTATOR, NOT REINVENTED
-------------------------------------------------------
`_label_tile` below mirrors `readData.tileImages` in magician_grabber_annotator, which is
what built Aug26_78K. Getting any of it subtly wrong would produce a plausible curve that
measures a different labelling than the model was trained under:

  * point coordinates are in RAW MOSAIC space and are halved (`xFull // 2`) -- the .pnm is
    2448x2048 and the debayered .png the classifier sees is 1224x1024;
  * a tile carries a defect if the point falls inside its 48 px extent (containment, not
    distance to the tile centre);
  * a tile whose rectangle-distance to a defect point is in (0, quarantine_px) is dropped
    from BOTH pools -- the dent extends past the click, so it is not trustworthy as clean;
  * a tile with no pixel above `low_value_tile_threshold` is skipped as a dark frame;
  * the class name is point class + severity with spaces removed, then `class_`-prefixed,
    then put through the run config's own `class_merges` / `drop_classes`.

Images are read with `readPolarPNMToRGBA` and tiled with `tile_and_cast_data_torch` -- the
training reader and the deployment tiler respectively, not reimplementations. The channel
order in particular is a cv2 write/read round trip and must not be second-guessed.

Usage:
    python -m analysis.eval.eval_step_curve <config.json> [checkpoint.ckpt] \\
        [--steps 16,18,20,24,32] [--frames coverage|factory|both] [--stride N] [--limit N]

    --stride N   score every Nth frame (quick look; N=1 is the full set)
    --limit N    stop after N frames
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

from mvc.core.artifact_paths import find_artifact, out_path
from mvc.core.config import load_hyperparameters
from mvc.core.datasets import clean_class_index
from mvc.core.lit_classifier import Classifier
from mvc.core.metrics import detection_at_fa, fa_threshold
from mvc.core.read_data import readPolarPNMToRGBA
from mvc.inference.classifier_pnm import tile_and_cast_data_torch

# Where the raw recordings live on this box. The frame lists carry the DEV BOX paths they
# were dumped with, so the prefix is swapped rather than the lists being rewritten -- the
# lists are frozen artifacts and two boxes must agree on them byte for byte.
RAW_ROOT = '/storage/ammarkov/MagicianRawDatasets/Magician'
DEV_ROOT = '/media/ammar/games2/Datasets/Magician'

TILE = 48
QUARANTINE_PX = 32          # headless_dump.py's value for the Aug26 dump
LOW_VALUE_THRESHOLD = 20    # headless_dump.py passes threshold=20
FA_TARGETS = (0.05, 0.10)


def local_path(frame_json):
    """Dev-box frame path -> this box, and the .json sidecar -> its image."""
    p = frame_json.replace(DEV_ROOT, RAW_ROOT)
    stem = p[:-5] if p.endswith('.json') else p
    for ext in ('.png', '.pnm'):
        if os.path.exists(stem + ext):
            return stem + ext, p
    return None, p


def point_class_name(pclass, severity, use_severity=True):
    """Annotator point class + severity -> the dataset's class name.

    'Positive Dent' + 'Class B' -> 'class_PositiveDentClassB'. The space removal is
    H5DatasetWriter.add()'s `class_name_no_space`; the concatenation order is
    readData.tileImages'. Clean points carry no severity.
    """
    if pclass in ('Clean', 'RLClean'):
        return None
    text = pclass + (severity if use_severity and severity else '')
    return 'class_' + text.replace(' ', '')


def apply_scheme(name, merges, drops):
    """Put a raw class name through the run config's merge/drop scheme."""
    if name is None:
        return None
    name = merges.get(name, name)
    return None if name in drops else name


def frame_points(json_path, merges, drops, use_severity=True):
    """(x, y, mapped_class_or_None) per annotated point, in debayered image coords.

    `mapped_class` is None for points the run config drops -- those still quarantine and
    still keep a tile out of the clean pool, exactly as they did at dump time, but they
    are not scored because the model has no output for them.
    """
    try:
        with open(json_path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None, False
    if d.get('lightDirection') == 'No Light':
        return None, True          # the dump skipped these frames outright
    clicks = d.get('pointClicks', []) or []
    classes = d.get('pointClasses', []) or []
    sevs = d.get('pointSeverities', []) or []
    out = []
    for i, (xf, yf) in enumerate(clicks):
        pc = classes[i] if i < len(classes) else ''
        sv = sevs[i] if i < len(sevs) else ''
        if pc in ('Clean', 'RLClean'):
            continue               # clean-ish points never quarantine and are not defects
        raw = point_class_name(pc, sv, use_severity)
        out.append((int(xf) // 2, int(yf) // 2, apply_scheme(raw, merges, drops)))
    return out, False


def grid_origins(size, step):
    """Top-left coordinates of the deployment grid along one axis.

    Matches `tile_and_cast_data_torch`, which uses torch.unfold: origins are
    0, step, 2*step, ... while the tile still fits, and the remainder at the far edge is
    simply not covered. Any independent range() here would drift from the tiler.
    """
    return np.arange(0, size - TILE + 1, step, dtype=np.int64)


def score_frame(model, dev, rgba, points, steps, clean_id, n_classes, acc):
    """Run one frame at every step and fold the results into `acc`."""
    h, w = rgba.shape[:2]
    for step in steps:
        xs, ys = grid_origins(w, step), grid_origins(h, step)
        if not len(xs) or not len(ys):
            continue
        tiles = tile_and_cast_data_torch(rgba, tile_size=TILE, step=step)
        tiles = tiles.permute(0, 3, 1, 2).contiguous().to(dev)
        mass = np.empty(len(tiles), np.float32)
        CH = 2048
        with torch.no_grad():
            for s in range(0, len(tiles), CH):
                chunk = tiles[s:s + CH]
                with torch.amp.autocast(device_type=dev.type, dtype=torch.float16,
                                        enabled=(dev.type == 'cuda')):
                    logits = model(chunk)
                prob = torch.softmax(logits.float(), dim=1)
                mass[s:s + len(chunk)] = (1.0 - prob[:, clean_id]).cpu().numpy()
        del tiles

        # unfold emits row-major over (y, x): index = iy * len(xs) + ix
        mass2d = mass.reshape(len(ys), len(xs))

        # --- the clean pool, mirroring tileImages' exclusions --------------------------
        # A tile is clean only if it contains no defect point AND is outside every
        # quarantine band. Dark tiles are dropped from both pools.
        keep = np.ones((len(ys), len(xs)), bool)
        for (px, py, _cls) in points:
            # tiles CONTAINING the point -> never clean
            cx = (xs <= px) & (px < xs + TILE)
            cy = (ys <= py) & (py < ys + TILE)
            keep &= ~np.outer(cy, cx)
            # quarantine band: rectangle-distance in (0, QUARANTINE_PX)
            dx = np.maximum(np.maximum(xs - px, 0), px - (xs + TILE - 1))
            dy = np.maximum(np.maximum(ys - py, 0), py - (ys + TILE - 1))
            d2 = dy[:, None] ** 2 + dx[None, :] ** 2
            keep &= ~((d2 > 0) & (d2 < QUARANTINE_PX ** 2))
        a = acc[step]
        a['clean_mass'].append(mass2d[keep])

        # --- per-point detection -------------------------------------------------------
        for (px, py, cls) in points:
            if cls is None:
                continue
            ix = np.where((xs <= px) & (px < xs + TILE))[0]
            iy = np.where((ys <= py) & (py < ys + TILE))[0]
            if not len(ix) or not len(iy):
                # No tile on this grid contains the point. A structural miss -- the whole
                # reason a step curve is not flat. Recorded as -inf so it can never clear
                # any threshold, and counted separately so the effect stays visible.
                a['point_mass'].append(-np.inf)
                a['point_class'].append(cls)
                a['uncovered'] += 1
                continue
            a['point_mass'].append(float(mass2d[np.ix_(iy, ix)].max()))
            a['point_class'].append(cls)
        a['frames'] += 1
        a['tiles'] += mass.size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('checkpoint', nargs='?')
    ap.add_argument('--steps', default='16,18,20,24,32')
    ap.add_argument('--frames', default='both', choices=('coverage', 'factory', 'both'))
    ap.add_argument('--stride', type=int, default=1, help='score every Nth frame')
    ap.add_argument('--limit', type=int, default=0, help='stop after N frames (0 = all)')
    ap.add_argument('--min-points', type=int, default=25,
                    help='a class needs this many annotated points to enter the macro. '
                         'A class with a handful of points swings the macro by tens of '
                         'points between steps and reads as a step effect when it is '
                         'sampling noise -- at --stride 40, DeformationClassA had n=2 and '
                         'moved the macro 11 points on one point flipping.')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(',')]
    cfg = load_hyperparameters(args.config)

    classes = cfg.get('classes')
    if not classes:
        sys.exit(f'{args.config} has no "classes" list -- score the run first')
    clean_id = clean_class_index(classes)
    if clean_id is None:
        sys.exit(f'no clean class among {classes}')
    merges = cfg.get('class_merges') or {}
    drops = set(cfg.get('drop_classes') or [])
    use_sev = True

    # --- frame lists ------------------------------------------------------------------
    wanted = []
    if args.frames in ('coverage', 'both'):
        p = find_artifact('val_coverage_frames.json')
        if not p:
            sys.exit('val_coverage_frames.json not found')
        wanted += json.load(open(p))['frames']
    if args.frames in ('factory', 'both'):
        p = find_artifact('val_frames_frozen.json') or \
            '/storage/ammarkov/magician_datasets/Aug26_78K/val_frames_frozen.json'
        if not os.path.exists(p):
            sys.exit('val_frames_frozen.json not found')
        wanted += json.load(open(p))['val_frames']
    wanted = sorted(set(wanted))[::args.stride]
    if args.limit:
        wanted = wanted[:args.limit]
    print(f'[step-curve] {len(wanted):,} frames requested ({args.frames})')

    # --- model ------------------------------------------------------------------------
    ckpt = args.checkpoint
    if ckpt is None:
        import glob
        cks = sorted(glob.glob(os.path.join(cfg['checkpoint_dir'], '*.ckpt')))
        if not cks:
            sys.exit(f'no checkpoints in {cfg["checkpoint_dir"]}')
        mon = cfg.get('checkpoint_monitor', 'val_detect_auroc')
        best = 'max' if cfg.get('checkpoint_mode', 'max') == 'max' else 'min'
        scored = [(float(m.group(1)), c) for c in cks
                  if (m := re.search(rf'{re.escape(mon)}=([0-9]+\.[0-9]+)', c))]
        ckpt = (max if best == 'max' else min)(scored)[1] if scored else cks[-1]
    print(f'[step-curve] {os.path.basename(ckpt)}')

    model = Classifier.from_config(cfg, num_classes=len(classes), clean_class=clean_id)
    model.load_state_dict(torch.load(ckpt, weights_only=False,
                                     map_location='cpu')['state_dict'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()

    acc = {s: {'clean_mass': [], 'point_mass': [], 'point_class': [],
               'frames': 0, 'tiles': 0, 'uncovered': 0} for s in steps}
    missing = skipped = 0
    for i, fj in enumerate(wanted):
        img_path, json_path = local_path(fj)
        if img_path is None:
            missing += 1
            continue
        pts, no_light = frame_points(json_path, merges, drops, use_sev)
        if pts is None:
            skipped += 1
            continue
        rgba = readPolarPNMToRGBA(img_path)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            missing += 1
            continue
        score_frame(model, dev, rgba, pts, steps, clean_id, len(classes), acc)
        if i % 25 == 0:
            print(f'  {i:,}/{len(wanted):,} frames', end='\r')
    print()
    if missing:
        print(f'[step-curve] WARNING {missing:,} frames unreadable or absent under {RAW_ROOT}')
    if skipped:
        print(f'[step-curve] {skipped:,} frames skipped (no-light / unreadable json)')

    # --- the macro class set, fixed once and used at every step ------------------------
    # The same annotation points are scored at every step, so support does not vary with
    # the step and the set can be resolved once. It MUST be: letting each step macro over
    # whichever classes it happened to see turns a class appearing or vanishing into a
    # fake step effect.
    counts = defaultdict(int)
    for c in acc[steps[0]]['point_class']:
        counts[c] += 1
    macro_set = sorted(c for c, n in counts.items() if n >= args.min_points)
    thin = sorted(c for c, n in counts.items() if n < args.min_points)
    tier_of = {}
    cov = find_artifact('val_coverage_frames.json')
    if cov:
        tier_of = {k: v['tier'] for k, v in json.load(open(cov))['classes'].items()}
    tier_a_set = [c for c in macro_set if tier_of.get(c) == 'TIER_A']
    print(f'\n[step-curve] macro over {len(macro_set)} classes with >= {args.min_points} '
          f'points: {", ".join(c[len("class_"):] for c in macro_set) or "(none)"}')
    if thin:
        print(f'[step-curve] excluded, too few points: '
              + ', '.join(f'{c[len("class_"):]} (n={counts[c]})' for c in thin))
    if tier_a_set:
        print(f'[step-curve] TIER_A subset ({len(tier_a_set)}): '
              + ', '.join(c[len('class_'):] for c in tier_a_set))

    # --- report -----------------------------------------------------------------------
    result = {'config': os.path.basename(args.config), 'model': cfg['model'],
              'checkpoint': os.path.basename(ckpt), 'frames_scored': acc[steps[0]]['frames'],
              'quarantine_px': QUARANTINE_PX, 'tile': TILE,
              'min_points': args.min_points, 'macro_classes': macro_set,
              'macro_classes_tier_a': tier_a_set, 'excluded_thin': thin, 'steps': {}}
    print(f'\n{"step":>5s} {"tiles/frame":>12s} {"points":>8s} {"uncov":>7s} '
          f'{"thr@FA5":>8s} {"macro@FA5":>10s} {"TIER_A@FA5":>11s} {"macro@FA10":>11s}')
    print('-' * 80)
    for s in steps:
        a = acc[s]
        if not a['point_mass']:
            continue
        clean = np.concatenate(a['clean_mass']) if a['clean_mass'] else np.array([])
        pmass = np.array(a['point_mass'], np.float64)
        pcls = np.array(a['point_class'])
        # One pooled array so the shared estimator sees exactly the shape it does
        # everywhere else: clean tiles set the threshold, defect points are scored on it.
        mass = np.concatenate([clean, pmass])
        is_clean = np.concatenate([np.ones(len(clean), bool), np.zeros(len(pmass), bool)])
        per_class, thr5 = {}, None
        for fa in FA_TARGETS:
            thr = fa_threshold(mass, is_clean, fa)
            if fa == 0.05:
                thr5 = thr
            for c in sorted(set(pcls.tolist())):
                sel = np.concatenate([np.zeros(len(clean), bool), pcls == c])
                per_class.setdefault(c, {})[f'detect_at_fa{int(fa * 100)}'] = \
                    detection_at_fa(mass, is_clean, sel, fa)
        for c in per_class:
            per_class[c]['points'] = int((pcls == c).sum())
        def macro(sel, key):
            vals = [per_class[c][key] for c in sel if c in per_class]
            return float(np.mean(vals)) if vals else float('nan')

        m5, m10 = macro(macro_set, 'detect_at_fa5'), macro(macro_set, 'detect_at_fa10')
        ta5 = macro(tier_a_set, 'detect_at_fa5')
        tpf = a['tiles'] / max(a['frames'], 1)
        print(f'{s:5d} {tpf:12,.0f} {len(pmass):8,d} {a["uncovered"]:7,d} '
              f'{thr5:8.4f} {m5:10.2f} {ta5:11.2f} {m10:11.2f}')
        result['steps'][str(s)] = {
            'tiles_per_frame': tpf, 'clean_tiles': int(len(clean)),
            'points': int(len(pmass)), 'uncovered_points': a['uncovered'],
            'fa5_threshold': float(thr5),
            'macro_detect_at_fa5': m5, 'macro_detect_at_fa10': m10,
            'macro_detect_at_fa5_tier_a': ta5,
            'per_class': per_class}

    print(f'\n{"class":32s} ' + ' '.join(f'{"s" + str(s):>8s}' for s in steps))
    print('-' * (33 + 9 * len(steps)))
    allc = sorted({c for s in steps if str(s) in result['steps']
                   for c in result['steps'][str(s)]['per_class']})
    for c in allc:
        cells = []
        for s in steps:
            pc = result['steps'].get(str(s), {}).get('per_class', {}).get(c)
            cells.append(f'{pc["detect_at_fa5"]:8.2f}' if pc else f'{"-":>8s}')
        n = result['steps'][str(steps[0])]['per_class'].get(c, {}).get('points', 0)
        mark = ('A' if c in tier_a_set else '+') if c in macro_set else ' '
        print(f'{mark} {c:30s} ' + ' '.join(cells) + f'   (n={n:,})')

    print('\nPER-POINT detection at a per-tile-matched FA. "uncov" counts annotation points'
          '\nthat NO tile of that grid contains -- structural misses that exist only because'
          '\nthe tiling is coarse, and the mechanism this curve is here to expose.'
          '\n"A" = in the TIER_A macro, "+" = in the all-class macro, blank = too few points.')

    out = args.out or out_path(f"{cfg['name']}_{cfg['model']}", '_step_curve.json')
    json.dump(result, open(out, 'w'), indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
