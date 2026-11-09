#!/usr/bin/env python3
"""Mine hard tiles -- false positives and missed/misclassified defects -- into a new,
wholly separate, ablatable `training_dataset` directory.

WHY THIS EXISTS
---------------
`mine_hardneg.py` already mines false positives, but only from 7 pre-vetted, totally
clean, mostly-flat folders (calibration patterns, chessboards) that were excluded from
Aug26_78K entirely. That experiment is a documented dead end (`knowledge/PLAN.md`:
"258k mined false positives added as class_clean... gave +3.39 paired (3/3 seeds
worse)... mined FPs are exactly the high-contrast edge structures the model reads as
defects"). The follow-up hypothesis this script tests is different: Aug26_78K's clean
tiles come from a coarse, NON-OVERLAPPING 48px grid (`ratio_clean=10`), so hard clean
regions near real structure (holes, seams, panel edges) on frames the model already
trained on may simply be under-sampled at the specific offsets that matter, not absent
from the corpus. A dense inference pass at the deployment step (16, not the dump's
48px grid) over the SAME training recordings can surface those exact offsets as new,
never-dumped tile crops -- same frames, genuinely different samples.

Both `--kind fp` (false positives -- clean-eligible tiles the model flags, labeled
class_clean) and `--kind fn` (missed/misclassified real defects, labeled with their
TRUE class -- standard hard-positive mining, a different, untried mechanism from the
fp dead end) are supported, always to SEPARATE output directories so either can be
folded into a `training_dataset` list (or dropped) independently. See
`knowledge/PLAN.md` / the 2026-09-15 plan for the full design rationale.

TEXTURE/SHARPNESS GATE -- `--min-sharpness` (default 15.0, Laplacian variance on the
tile's mean-intensity image, see `tile_sharpness()`) additionally requires a flagged
tile to carry real edge/texture content before it's mined, on top of the confidence+
vote criteria above. Motivation: those criteria alone still let through some flat/
near-blank tiles that happened to clear the confidence+vote bar (sensor noise on a dark
region, not a "learnable" example) -- confirmed visually 2026-09-17 by sorting already-
mined tiles by this score (knowledge/PLAN.md has the labelled comparison grids this
default was picked from).

OUTPUT FORMAT -- PNG folders (class_<Name>/*.png), not HDF5. `RGBAImageFolder`
auto-loads this whenever a directory has no `dataset.h5` (`mvc/core/datasets.py:
_load_one_dir`). A plain `cv2.imwrite` of a raw RGBA crop does NOT round-trip through
`load_rgba_image()` -- empirically verified 2026-09-15: `cv2.imwrite(p, tile)` then
`load_rgba_image(p)` swaps channels 0 and 2. The correct write, matching
`H5DatasetWriter.add()`'s own `[2,1,0,3]` swap, is `cv2.imwrite(p, tile[:,:,[2,1,0,3]])`
-- verified to round-trip byte-identical. Get this wrong and every mined tile silently
trains on the wrong channel semantics.

Usage:
  python -m analysis.datasets.mine_hard_tiles --kind fp --model fzcnxtiny \\
      --out-dir /storage/ammarkov/magician_datasets/hardneg_fp_20260915 \\
      --frame-limit 30   # sanity pass first, see module docstring / PLAN.md
  python -m analysis.datasets.mine_hard_tiles --kind fn --model anc \\
      --out-dir /storage/ammarkov/magician_datasets/hardpos_fn_20260915
  python -m analysis.datasets.mine_hard_tiles --pool-only --kind fp   # no GPU, just counts
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

from analysis.eval.eval_step_curve import RAW_ROOT, TILE, frame_points, grid_origins
from analysis.eval.eval_vote_curve import cache_frame, neighbor_counts
from mvc.core.artifact_paths import find_config_with_classes
from mvc.core.config import load_hyperparameters
from mvc.core.datasets import clean_class_index
from mvc.core.lit_classifier import Classifier
from mvc.core.read_data import readPolarPNMToRGBA

AUG26_ROOT = '/storage/ammarkov/magician_datasets/Aug26_78K'
MODELS = {'fzcnxtiny': 'convnext_tiny', 'anc': 'convnext_pico'}

# mine_hardneg.py's already-validated, provably-clean, excluded folders -- kept here as
# the (non-default) `--pool clean_folders` option for comparability with that experiment.
CLEAN_POOL_FOLDERS = [
    'AltinayCarSlideFromPattern', 'AltinayUniquePattern750', 'AltinayUniquePattern750_2',
    'AltinayemulateCalibration', 'AltinayCleanOnlyemulateCalibration', 'Altinaychessboardbase',
    'Altinayuniquepatternmirror',
]

# Certified, defect-bearing recordings absent from Aug26_78K/{train,val}_list.txt, hand-
# verified 2026-09-15 (see knowledge/PLAN.md): AltinayMovingCar (66 defects, exposure 350,
# 168 PNGs) and test2_400 (35 defects, exposure 400, 35 PNGs). Excluded by the same check:
# miscalibratedLight_FORTH_POSA_50 (name says why -- exposure 50, likely bad-light data)
# and test_400 (info.json claims 168 defects but 0 PNGs are present on this box).
UNSEEN_DEFECT_FOLDERS = ['AltinayMovingCar', 'test2_400']


def aug26_recordings():
    train = [os.path.basename(l.strip()) for l in open(os.path.join(AUG26_ROOT, 'train_list.txt'))]
    val = [os.path.basename(l.strip()) for l in open(os.path.join(AUG26_ROOT, 'val_list.txt'))]
    return train, val


def pool_folders(kind, pool):
    """Recording names (not full paths) making up the candidate pool for this run."""
    if pool == 'clean_folders':
        return CLEAN_POOL_FOLDERS
    if pool == 'unseen_defects':
        return UNSEEN_DEFECT_FOLDERS
    if pool == 'train_corpus':
        train, _val = aug26_recordings()
        return train
    raise ValueError(pool)


def enumerate_pool(kind, pool, stride=1, frame_limit=0):
    """[(recording, png_path), ...] for the whole pool, no GPU work -- used both for the
    real run and for --pool-only sizing."""
    folders = pool_folders(kind, pool)
    out = []
    for name in folders:
        folder = os.path.join(RAW_ROOT, name)
        if not os.path.isdir(folder):
            print(f'[pool] {name}: missing on this box, skipping')
            continue
        pngs = sorted(glob.glob(os.path.join(folder, '*.png')))[::stride]
        out.extend((name, p) for p in pngs)
    if frame_limit:
        out = out[:frame_limit]
    return out


def load_model_for_run(run):
    cfg_path = find_config_with_classes(f'{run}_{MODELS[run]}.json')
    if not cfg_path:
        sys.exit(f'no filed config found for {run}_{MODELS[run]}.json')
    cfg = load_hyperparameters(cfg_path)
    classes = cfg.get('classes')
    if not classes:
        sys.exit(f'{cfg_path} has no "classes" -- score the run first')
    clean_id = clean_class_index(classes)
    if clean_id is None:
        sys.exit(f'no clean class among {classes}')
    merges = cfg.get('class_merges') or {}
    drops = set(cfg.get('drop_classes') or [])

    cks = sorted(glob.glob(os.path.join(cfg['checkpoint_dir'], '*.ckpt')))
    if not cks:
        sys.exit(f'no checkpoints in {cfg["checkpoint_dir"]}')
    mon = cfg.get('checkpoint_monitor', 'val_detect_auroc')
    best = 'max' if cfg.get('checkpoint_mode', 'max') == 'max' else 'min'
    scored = [(float(m.group(1)), c) for c in cks
             if (m := re.search(rf'{re.escape(mon)}=([0-9]+\.[0-9]+)', c))]
    ckpt = (max if best == 'max' else min)(scored)[1] if scored else cks[-1]

    model = Classifier.from_config(cfg, num_classes=len(classes), clean_class=clean_id)
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()
    print(f'[model] {run} ({MODELS[run]}): {os.path.basename(cfg_path)}, '
         f'{os.path.basename(ckpt)}, {len(classes)} classes, clean_id={clean_id}, dev={dev}')
    return model, dev, classes, clean_id, merges, drops


def tile_sharpness(crop):
    """Laplacian-variance edge/texture strength of an RGBA tile crop, computed on its
    mean-intensity representation -- a cheap, well-established blur/flatness proxy
    (high = real high-frequency structure: a hole, seam, panel edge, embossed marking;
    low = near-uniform/flat, little content for the model to learn from even if the
    confidence+vote criteria flagged it). Calibrated 2026-09-17 by sorting 2,000 already-
    mined FP tiles by this score: values around 6 are visibly near-blank/featureless,
    values around 20 already show clear edges/gradients/texture -- see knowledge/PLAN.md
    for the labelled low/mid/high comparison grids this threshold was picked from."""
    intensity = crop.astype(np.float32).mean(axis=2).astype(np.uint8)
    return float(cv2.Laplacian(intensity, cv2.CV_64F).var())


def write_tile(out_dir, class_name, crop, x, y, recording, frame_stem, manifest_fh, extra):
    """Write one PRE-SLICED tile crop if not already present (idempotent across
    models/runs); always append a manifest line so a re-flag by a second model is still
    recorded."""
    class_dir = os.path.join(out_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    fname = f'{recording}__{frame_stem}__x{x}_y{y}.png'
    path = os.path.join(class_dir, fname)
    is_new = not os.path.exists(path)
    if is_new:
        # [2,1,0,3] swap required for a correct round-trip through load_rgba_image() --
        # see module docstring, empirically verified 2026-09-15.
        cv2.imwrite(path, crop[:, :, [2, 1, 0, 3]])
    manifest_fh.write(json.dumps({'path': os.path.relpath(path, out_dir), 'class': class_name,
                                  'recording': recording, 'frame': frame_stem, 'x': int(x),
                                  'y': int(y), 'new_file': is_new, **extra}) + '\n')
    return is_new


def mine(kind, models, pool, out_dir, step, threshold, stride, frame_limit, max_tiles,
        kernel=1, min_votes=2, min_sharpness=0.0):
    items = enumerate_pool(kind, pool, stride, frame_limit)
    print(f'[mine] kind={kind} pool={pool} step={step} threshold={threshold} '
         f'kernel={kernel} min_votes={min_votes} (fp only) min_sharpness={min_sharpness} '
         f'-> {len(items):,} (recording, frame) pairs queued, models={models}')
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'manifest.jsonl')
    counts = defaultdict(int)
    n_new = n_seen = n_frames = n_low_sharpness = 0
    t0 = time.time()
    with open(manifest_path, 'a') as mfh:
        for run in models:
            model, dev, classes, clean_id, merges, drops = load_model_for_run(run)
            class_index = {c: i for i, c in enumerate(classes)}
            for (recording, png) in items:
                json_path = os.path.splitext(png)[0] + '.json'
                pts, no_light = frame_points(json_path, merges, drops, True)
                if pts is None or no_light:
                    continue
                rgba = readPolarPNMToRGBA(png)
                if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                    continue
                cached = cache_frame(model, dev, rgba, pts, step, clean_id,
                                     return_pred=(kind == 'fn'))
                if cached is None:
                    continue
                h, w = rgba.shape[:2]
                xs, ys = grid_origins(w, step), grid_origins(h, step)
                mass2d, keep2d = cached['mass2d'], cached['keep']
                frame_stem = os.path.splitext(os.path.basename(png))[0]
                n_frames += 1

                if kind == 'fp':
                    # Raw per-tile threshold alone reproduces the exact multiple-
                    # comparisons artifact eval_frame_level_sweep.py already found and
                    # fixed: with ~4,800 tiles/frame at step 16, isolated softmax
                    # overconfidence flags dozens of scattered singleton tiles per frame
                    # that are noise, not structure -- empirically confirmed here too
                    # (209 flagged tiles in one frame before this filter was added,
                    # 2026-09-15 sanity pass). A tile only counts if enough of its
                    # spatial neighbours also fire, same primitive the live deployed
                    # path uses (classifier_pnm.process_predictions_erode) -- this
                    # keeps mining pointed at genuine clustered structure (holes, seams,
                    # panel edges) instead of single-tile flicker.
                    raw = mass2d >= threshold
                    voted = raw & (neighbor_counts(raw, kernel) >= min_votes)
                    flag = keep2d & voted
                    for iy, ix in zip(*np.where(flag)):
                        n_seen += 1
                        xg, yg = int(xs[ix]), int(ys[iy])
                        crop = rgba[yg:yg + TILE, xg:xg + TILE, :]
                        sharp = tile_sharpness(crop)
                        if min_sharpness and sharp < min_sharpness:
                            n_low_sharpness += 1
                            continue
                        is_new = write_tile(out_dir, 'class_clean', crop, xg, yg,
                                            recording, frame_stem, mfh,
                                            {'model': run, 'defect_mass': float(mass2d[iy, ix]),
                                             'sharpness': sharp})
                        n_new += int(is_new)
                        if is_new:
                            counts['class_clean'] += 1
                else:  # fn
                    pred2d = cached['pred2d']
                    # cached['points'] is (iy_array_or_None, ix_array_or_None, cls) per
                    # annotated point -- the exact same containment computation
                    # cache_frame already does internally, reused rather than redone.
                    for (iy, ix, cls) in cached['points']:
                        if cls is None or cls not in class_index or iy is None:
                            continue  # dropped class, or a structural miss (no tile contains it)
                        true_idx = class_index[cls]
                        sub_pred = pred2d[np.ix_(iy, ix)]
                        sub_mass = mass2d[np.ix_(iy, ix)]
                        wrong = np.where(sub_pred != true_idx)
                        for wy, wx in zip(*wrong):
                            n_seen += 1
                            ry, rx = iy[wy], ix[wx]
                            xg, yg = int(xs[rx]), int(ys[ry])
                            crop = rgba[yg:yg + TILE, xg:xg + TILE, :]
                            sharp = tile_sharpness(crop)
                            if min_sharpness and sharp < min_sharpness:
                                n_low_sharpness += 1
                                continue
                            is_new = write_tile(out_dir, cls, crop, xg, yg,
                                                recording, frame_stem, mfh,
                                                {'model': run, 'true_class': cls,
                                                 'predicted_class': classes[int(sub_pred[wy, wx])],
                                                 'defect_mass': float(sub_mass[wy, wx]),
                                                 'sharpness': sharp})
                            n_new += int(is_new)
                            if is_new:
                                counts[cls] += 1
                if max_tiles and n_new >= max_tiles:
                    print(f'[mine] hit --max-tiles {max_tiles:,}, stopping early '
                         f'({n_frames:,} frames scanned this run)')
                    break
                if n_frames % 25 == 0:
                    dt = time.time() - t0
                    print(f'  {run}: {n_frames:,} frames, {n_new:,} new tiles '
                         f'({n_new/max(dt,1e-6):.1f} tiles/s, {dt:.0f}s elapsed)', end='\r')
            print()
            del model
            torch.cuda.empty_cache()
            if max_tiles and n_new >= max_tiles:
                break
    dt = time.time() - t0
    print(f'\n[mine] done: {n_frames:,} frames scanned, {n_seen:,} flagged tile-events, '
         f'{n_low_sharpness:,} dropped for low sharpness/texture, '
         f'{n_new:,} new tiles written in {dt:.0f}s ({n_new/max(dt,1e-6):.2f} tiles/s)')
    for c, n in sorted(counts.items()):
        print(f'  {c}: {n:,}')
    write_manifest_md(out_dir, kind, pool, models, step, threshold, kernel, min_votes,
                      min_sharpness, counts, n_frames, n_low_sharpness, dt)
    return counts


def write_manifest_md(out_dir, kind, pool, models, step, threshold, kernel, min_votes,
                      min_sharpness, counts, n_frames, n_low_sharpness, dt):
    lines = [
        f'# Hard-tile mining manifest ({kind})', '',
        f'- kind: {kind}', f'- pool: {pool}', f'- models: {models}',
        f'- deployment step: {step}', f'- threshold (fp only): {threshold}',
        f'- spatial vote (fp only): kernel={kernel} min_votes={min_votes}',
        f'- min_sharpness (texture/edge filter, Laplacian variance): {min_sharpness}',
        f'- tiles dropped for low sharpness: {n_low_sharpness:,}',
        f'- frames scanned: {n_frames:,}', f'- wall time: {dt:.0f}s',
        f'- generated: {time.strftime("%Y-%m-%d %H:%M")}', '', '## Per-class tile counts', '',
    ]
    for c, n in sorted(counts.items()):
        lines.append(f'- {c}: {n:,}')
    lines.append('')
    lines.append('Format: PNG folders (class_<Name>/*.png), RGBAImageFolder-compatible. '
                 'Add as an extra `training_dataset` list entry to use; remove the entry to '
                 'fully ablate. See analysis/datasets/mine_hard_tiles.py and knowledge/PLAN.md.')
    with open(os.path.join(out_dir, 'MANIFEST.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=('fp', 'fn'), required=True)
    ap.add_argument('--model', choices=list(MODELS) + ['both'], default='both')
    ap.add_argument('--pool', choices=('train_corpus', 'clean_folders', 'unseen_defects'),
                    default=None, help='default: train_corpus for fp, unseen_defects for fn')
    ap.add_argument('--out-dir', required=False)
    ap.add_argument('--step', type=int, default=16, help='deployment tiling step (16=densest)')
    ap.add_argument('--threshold', type=float, default=0.999,
                    help='fp only: mine a clean-eligible tile if 1-P(clean) >= this. '
                         'Calibrated 2026-09-15 on a 255-frame train_corpus subset '
                         '(kernel=1,min_votes=2): 0.95->19.9/frame, 0.98->11.8/frame, '
                         '0.99->8.0/frame, 0.995->5.3/frame, 0.999->1.7/frame -- 0.999 '
                         'chosen as "really very strong" only, see knowledge/PLAN.md')
    ap.add_argument('--kernel', type=int, default=1,
                    help='fp only: spatial-vote neighbourhood radius (deployed default=1)')
    ap.add_argument('--min-votes', type=int, default=2,
                    help='fp only: min activated neighbours (incl. self) to count '
                         '(deployed default=2) -- filters isolated softmax-overconfidence '
                         'singletons, see mine()')
    ap.add_argument('--min-sharpness', type=float, default=15.0,
                    help='drop a flagged tile if its Laplacian-variance texture/edge score '
                         'is below this (0 disables). Concentrates mining on tiles with real '
                         'learnable structure (holes, seams, embossed marks) rather than '
                         'flat/near-blank regions that happened to clear the confidence+vote '
                         'gate. Calibrated 2026-09-17 on already-mined FP tiles: ~6 is visibly '
                         'near-blank, ~20 already shows clear edges/texture -- see '
                         'tile_sharpness() and knowledge/PLAN.md.')
    ap.add_argument('--stride', type=int, default=1, help='use every Nth frame per recording')
    ap.add_argument('--frame-limit', type=int, default=0,
                    help='stop after N (recording, frame) pairs total -- sanity-pass control')
    ap.add_argument('--max-tiles', type=int, default=0, help='stop after N new tiles (0=no cap)')
    ap.add_argument('--pool-only', action='store_true',
                    help='report pool size only, no GPU/model work')
    args = ap.parse_args()

    pool = args.pool or ('train_corpus' if args.kind == 'fp' else 'unseen_defects')

    if args.pool_only:
        items = enumerate_pool(args.kind, pool, args.stride, args.frame_limit)
        by_recording = defaultdict(int)
        for r, _p in items:
            by_recording[r] += 1
        print(f'[pool-only] kind={args.kind} pool={pool} stride={args.stride} -> '
             f'{len(items):,} frames across {len(by_recording)} recordings')
        for r, n in sorted(by_recording.items()):
            print(f'  {r}: {n:,}')
        return

    if not args.out_dir:
        sys.exit('--out-dir is required unless --pool-only')
    models = list(MODELS) if args.model == 'both' else [args.model]
    mine(args.kind, models, pool, args.out_dir, args.step, args.threshold,
        args.stride, args.frame_limit, args.max_tiles, args.kernel, args.min_votes,
        args.min_sharpness)


if __name__ == '__main__':
    main()
