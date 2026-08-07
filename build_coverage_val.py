#!/usr/bin/env python3
"""Build the COVERAGE validation set: every defect class, tiered by how honest it can be.

The problem. Aug26_78K's val is 6 held-out recordings of the integrator's car, which is
what makes it the right ship/no-ship metric -- but that car does not contain every defect
type. Five defect classes have ZERO val tiles, and they are (bar one) the model's weakest:
PositiveDentClassC 76.9%, NegativeDentClassC 82.2%, PositiveDentClassB 85.9% recall on the
legacy split. Meanwhile WeldingClassA -- its best class at 98.4% -- is 84% of val defects.
So the factory KPI cannot see a regression in exactly the classes most likely to regress.

The fix is a SECOND val set that covers the taxonomy. Two things make it honest:

  1. Its frames are REMOVED from training (dataloader.exclude_frames), not copied out of
     it. A coverage set sampled from data the model also trains on measures memorisation,
     not detection, and would pass while broken.

  2. Classes are held out at RECORDING level where the data allows, which controls
     recording leakage the same way the factory val does. Measured spread across the 98
     train recordings:
         NegativeDentClassA  23 recordings   PositiveDentClassC   8
         PositiveDentClassB  14              MaterialDefectClassA 2   <- cannot
         NegativeDentClassC   9
     With only 2 recordings, holding one out would cost ~50% of MaterialDefect training
     data, so that class falls back to frame-disjoint and is LABELLED as optimistic.

Every class carries its tier into the manifest so no one quotes a frame-disjoint number as
a generalization result:
    TIER_A  recording-disjoint  -- comparable in kind to the factory val
    TIER_B  frame-disjoint      -- optimistic, tile-sibling controlled only
    TIER_C  not evaluated       -- vestigial classes (<=500 tiles, 1 recording). These are
                                   annotation slips, not classes: class_WeldingClassAClean
                                   has 2 tiles, class_Deformation and class_Unknown 8 each.

Usage:
    python build_coverage_val.py [--out val_coverage_frames.json] [--min-tiles 1500]
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import h5py
import numpy as np

TRAIN_H5 = '/storage/ammarkov/magician_datasets/Aug26_78K/train/dataset.h5'
CACHE = 'coverage_extract_cache.npz'

# Below this the "class" is an annotation slip, not a category worth evaluating.
TIER_C_MAX_TILES = 500
# Never take so much of a class that training starves.
MAX_TRAIN_FRACTION_REMOVED = 0.35


def extract(h5_path, cache=CACHE):
    """(recording, frame) per tile. Cached -- the metadata pass is minutes, not seconds."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return (z['labels'], z['rec_id'], z['frame_id'],
                list(z['rec_names']), list(z['frame_names']), list(z['class_names']))
    f = h5py.File(h5_path, 'r')
    raw = f.attrs['class_names']
    class_names = json.loads(raw if isinstance(raw, str) else ''.join(raw))
    labels = f['labels'][:]
    md = f['metadata']
    rec_ids = np.zeros(len(labels), np.int32)
    frame_ids = np.zeros(len(labels), np.int32)
    recs, frames = {}, {}
    CH = 250_000
    for s in range(0, len(labels), CH):
        e = min(s + CH, len(labels))
        for j, m in enumerate(md[s:e]):
            m = m.decode() if isinstance(m, bytes) else m
            src = m.split('"source"', 1)[1].split('"', 2)[1] if '"source"' in m else '?'
            frame = src.rsplit('(', 1)[0]          # drop the tile (x,y) -- same rule as
            rec = frame.rsplit('/', 1)[0].rsplit('/', 1)[-1]   # _dataset_source_frames
            frame_ids[s + j] = frames.setdefault(frame, len(frames))
            rec_ids[s + j] = recs.setdefault(rec, len(recs))
        print(f'  extracted {e:,}/{len(labels):,}', end='\r')
    f.close()
    rec_names = [k for k, _ in sorted(recs.items(), key=lambda kv: kv[1])]
    frame_names = [k for k, _ in sorted(frames.items(), key=lambda kv: kv[1])]
    np.savez_compressed(cache, labels=labels, rec_id=rec_ids, frame_id=frame_ids,
                        rec_names=np.array(rec_names, object),
                        frame_names=np.array(frame_names, object),
                        class_names=np.array(class_names, object))
    print()
    return labels, rec_ids, frame_ids, rec_names, frame_names, class_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='val_coverage_frames.json')
    ap.add_argument('--min-tiles', type=int, default=1500,
                    help='target val tiles per class before a class is considered covered')
    ap.add_argument('--train-h5', default=TRAIN_H5,
                    help='train dataset.h5 to carve coverage from (default: supermicro path)')
    args = ap.parse_args()

    labels, rec_id, frame_id, rec_names, frame_names, class_names = extract(args.train_h5)
    clean = class_names.index('class_clean')
    print(f'{len(labels):,} tiles, {len(rec_names)} recordings, {len(frame_names):,} frames')

    tot = Counter(labels.tolist())
    cls_rec = defaultdict(Counter)
    for c in range(len(class_names)):
        if c == clean or not tot.get(c):
            continue
        idx = np.where(labels == c)[0]
        for r, n in Counter(rec_id[idx].tolist()).items():
            cls_rec[c][r] = n

    tier, selected_recs = {}, set()
    for c, recs in cls_rec.items():
        if tot[c] <= TIER_C_MAX_TILES or len(recs) < 2:
            tier[c] = 'TIER_C'
        elif len(recs) >= 4:
            tier[c] = 'TIER_A'
        else:
            tier[c] = 'TIER_B'

    # Which classes actually NEED coverage: the ones the factory val cannot measure.
    # Holding out recordings rich in WeldingClassA buys nothing -- it is already 84% of
    # factory val defects -- while costing training data, so purity is scored against the
    # classes that need coverage, not against raw class counts.
    factory_covered = {'class_WeldingClassA', 'class_SealClassA', 'class_SealClassB',
                       'class_PositiveDentClassA', 'class_NegativeDentClassB',
                       'class_DeformationClassA', 'class_clean'}
    needs = [c for c in cls_rec
             if tier[c] in ('TIER_A', 'TIER_B') and class_names[c] not in factory_covered]
    rec_total = Counter(rec_id.tolist())

    def violates(proj):
        """True if holding out `proj` starves ANY class -- Tier C included. A class driven
        to zero training tiles crashes BalancedBatchSampler ('Classes [...] have zero
        samples'), and class_NegativeDent lives in exactly one recording, so this must
        consider every class, not just the ones being covered."""
        for c2 in cls_rec:
            removed = sum(cls_rec[c2][x] for x in proj)
            if tot[c2] - removed <= 0:
                return True
            if removed / tot[c2] > MAX_TRAIN_FRACTION_REMOVED:
                return True
        return False

    # --- Tier A: whole recordings, scarcest needed class first, PUREST recording first.
    for c in sorted([c for c in needs if tier[c] == 'TIER_A'], key=lambda c: tot[c]):
        have = sum(cls_rec[c][r] for r in selected_recs)
        if have >= args.min_tiles:
            continue
        # purity = share of this recording that is the class we want. High purity means
        # little collateral training data is lost per coverage tile gained.
        cands = sorted(cls_rec[c].items(), key=lambda kv: -kv[1] / rec_total[kv[0]])
        for r, n in cands:
            if r in selected_recs or violates(selected_recs | {r}):
                continue
            selected_recs.add(r)
            have += n
            if have >= args.min_tiles:
                break

    in_rec = np.isin(rec_id, list(selected_recs)) if selected_recs else np.zeros(len(labels), bool)
    chosen_frames = set(frame_id[in_rec].tolist())

    # --- Tier B: whole FRAMES from within the class's own recordings (tile-sibling safe,
    # recording-leaky). Only for classes recording-level holdout cannot serve.
    for c in [c for c in needs if tier[c] == 'TIER_B']:
        idx = np.where(labels == c)[0]
        have = int(np.isin(frame_id[idx], list(chosen_frames)).sum())
        if have >= args.min_tiles:
            continue
        per_frame = Counter(frame_id[idx].tolist())
        for fr, n in sorted(per_frame.items(), key=lambda kv: -kv[1]):
            if fr in chosen_frames:
                continue
            chosen_frames.add(fr)
            have += n
            if have >= args.min_tiles:
                break

    sel = np.isin(frame_id, list(chosen_frames))
    print(f'\nheld out: {len(selected_recs)} whole recordings + frame-level top-ups')
    print(f'          {len(chosen_frames):,} frames, {int(sel.sum()):,} tiles '
          f'({sel.sum()/len(labels)*100:.2f}% of train)\n')

    print(f"{'class':30s} {'tier':7s} {'train tot':>10s} {'-> cov':>8s} {'left':>10s} {'%out':>6s}")
    print('-' * 76)
    manifest = {}
    for c in sorted(cls_rec, key=lambda c: -tot[c]):
        got = int(((labels == c) & sel).sum())
        manifest[class_names[c]] = {'tier': tier[c], 'train_total': int(tot[c]),
                                    'coverage_tiles': got, 'recordings': len(cls_rec[c])}
        note = '  <- not evaluated' if tier[c] == 'TIER_C' else ''
        print(f'{class_names[c]:30s} {tier[c]:7s} {tot[c]:10,d} {got:8,d} '
              f'{tot[c]-got:10,d} {got/tot[c]*100:5.1f}%{note}')
    cg = int(((labels == clean) & sel).sum())
    print(f'{"class_clean":30s} {"-":7s} {tot[clean]:10,d} {cg:8,d} {tot[clean]-cg:10,d} '
          f'{cg/tot[clean]*100:5.1f}%')

    json.dump({
        '_README': ('Frames carved OUT of Aug26_78K train to form the coverage val. Pass as '
                    'dataloader.exclude_frames when training and to eval_coverage.py when '
                    'scoring. TIER_A classes are recording-disjoint (honest); TIER_B are '
                    'frame-disjoint (optimistic -- their recordings also appear in train, so '
                    'never quote them as generalization); TIER_C are vestigial and not '
                    'evaluated. The factory val (Aug26_78K/val) is untouched and remains the '
                    'ship/no-ship metric.'),
        'source': args.train_h5,
        'held_out_recordings': sorted(rec_names[r] for r in selected_recs),
        'n_frames': len(chosen_frames),
        'n_tiles': int(sel.sum()),
        'classes': manifest,
        'frames': sorted(frame_names[f] for f in chosen_frames),
    }, open(args.out, 'w'), indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
