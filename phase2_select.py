#!/usr/bin/env python3
"""Pick the Phase-2 full-train candidate from the completed screens, by a rule fixed
BEFORE the Phase-1 results exist, and write the full-train config.

Why a script rather than eyeballing the table: a full train is ~8-11 h, and the whole
point of Phase 1 is that two of its outcomes (DoLP turning out to be dead weight on a
pretrained backbone, or the 30k ranking being a step-budget artifact) would change which
model deserves that time. Choosing after seeing the numbers invites picking the model
that confirms the expectation. So the rule is written down here first:

  1. Among every 30k screen (all directly comparable -- same budget, same split),
     take the lowest miss@FA5. That is the default candidate.
  2. Phase 1b ran convnext_nano AND convnext_pico at 60k. nano lost to pico at 30k
     (8.50 vs 7.51) despite 1.75x the parameters. If that was undertraining, doubling
     the budget should close the gap. Only if nano actually BEATS pico at 60k does the
     budget bias count as demonstrated, and only then does nano take the slot -- a
     nano improvement that pico matches is just "more steps", not evidence about nano.
  3. The winning model keeps whichever DoLP setting won for it at 30k.

Full-train settings: 3 epochs, no limit_train_batches, save_top_k=3. Three epochs
because epoch index 2 was the best epoch on BOTH full trains measured so far, and
save_top_k=3 so score_checkpoints.py can rank all of them on the KPI afterwards --
val_detect_auroc picked the right epoch last time, but by +0.00 margin, and val_loss
would have picked one 0.75 miss@FA5 worse.

Usage:
    python phase2_select.py            # report + write p2_<model>.json
    python phase2_select.py --dry-run  # report only
"""

import glob
import json
import os
import sys

RESULT_KEYS = ('confusion_matrix', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment')


def miss_at_fa(curve_path, targets=(0.05, 0.10)):
    """miss rate (%) at matched false alarm, linearly interpolated from the sweep."""
    d = json.load(open(curve_path))
    pts = (d.get('sweeps') or {}).get('defect_mass') or d['sweep']
    fa = [p['false_alarm'] for p in pts]
    det = [p['detected'] for p in pts]
    order = sorted(range(len(fa)), key=lambda i: fa[i])
    fa = [fa[i] for i in order]
    det = [det[i] for i in order]

    def interp(t):
        if t <= fa[0]:
            return det[0]
        if t >= fa[-1]:
            return det[-1]
        for i in range(1, len(fa)):
            if fa[i] >= t:
                span = fa[i] - fa[i - 1]
                w = 0.0 if span == 0 else (t - fa[i - 1]) / span
                return det[i - 1] + w * (det[i] - det[i - 1])
        return det[-1]

    return {t: (1.0 - interp(t)) * 100.0 for t in targets}


def collect(prefixes=('tz', 'p1n', 'p1b')):
    """Every screen that has both a config and a finished threshold curve."""
    out = []
    for pfx in prefixes:
        for cfg_path in sorted(glob.glob(f'{pfx}_*.json')):
            base = cfg_path[:-len('.json')]
            if base.endswith('_confusion') or base.endswith('_threshold_curve'):
                continue
            cfg = json.load(open(cfg_path))
            model = cfg['model']
            curve = f"{cfg['name']}_{model}_threshold_curve.json"
            if not os.path.exists(curve):
                print(f'  (skip {cfg_path}: no {curve} yet)')
                continue
            r = miss_at_fa(curve)
            out.append({
                'cfg': cfg_path, 'name': cfg['name'], 'model': model,
                'dolp': bool(cfg['hparams'].get('DoLP', False)),
                'budget': int(cfg.get('limit_train_batches') or 0),
                'm5': r[0.05], 'm10': r[0.10],
            })
    return out


def write_full_train(pick, out_path):
    cfg = json.load(open(pick['cfg']))
    for k in RESULT_KEYS:
        cfg.pop(k, None)
    cfg.pop('limit_train_batches', None)          # full data, not a screen
    cfg['name'] = 'p2'
    cfg['hparams']['training_epochs'] = 3
    cfg['checkpoint_save_top_k'] = 3               # keep every epoch for KPI scoring
    cfg['checkpoint_dir'] = f"datasets/mix_ckpts/p2_{pick['model']}"
    json.dump(cfg, open(out_path, 'w'), indent=2)
    return cfg


def main():
    rows = collect()
    if not rows:
        sys.exit('no completed screens found')

    print(f"\n{'config':28s} {'model':22s} {'DoLP':5s} {'budget':>7s} {'miss@FA5':>9s} {'miss@FA10':>10s}")
    print('-' * 86)
    for r in sorted(rows, key=lambda r: r['m5']):
        print(f"{r['cfg']:28s} {r['model']:22s} {str(r['dolp']):5s} "
              f"{r['budget']:>7d} {r['m5']:9.2f} {r['m10']:10.2f}")

    s30 = [r for r in rows if r['budget'] == 30000]
    s60 = {r['model']: r for r in rows if r['budget'] == 60000}
    if not s30:
        sys.exit('no 30k screens to rank')

    pick = min(s30, key=lambda r: r['m5'])
    reason = f"best 30k screen ({pick['m5']:.2f} miss@FA5)"

    # --- rule 2: budget-bias override -----------------------------------------
    nano, pico = s60.get('convnext_nano'), s60.get('convnext_pico')
    print('\n--- step-budget check (rule 2) ---')
    if nano and pico:
        n30 = next((r for r in s30 if r['model'] == 'convnext_nano'), None)
        p30 = next((r for r in s30 if r['model'] == 'convnext_pico'), None)
        if n30 and p30:
            print(f"  30k: nano {n30['m5']:.2f}  pico {p30['m5']:.2f}  "
                  f"(gap {n30['m5'] - p30['m5']:+.2f})")
        print(f"  60k: nano {nano['m5']:.2f}  pico {pico['m5']:.2f}  "
              f"(gap {nano['m5'] - pico['m5']:+.2f})")
        if nano['m5'] < pico['m5']:
            print('  -> nano OVERTAKES pico at 60k: the 30k ranking was a budget artifact.')
            dolp_winner = min((r for r in s30 if r['model'] == 'convnext_nano'),
                              key=lambda r: r['m5'], default=None)
            pick = dolp_winner or nano
            pick = dict(pick)
            pick['cfg'] = next(r['cfg'] for r in rows
                               if r['model'] == 'convnext_nano' and r['budget'] == 30000)
            reason = 'convnext_nano overtakes pico at 60k (rule 2 override)'
        else:
            print('  -> nano does NOT overtake pico. The 30k ranking stands; extra steps '
                  'help both, so this is not evidence for nano.')
    else:
        print('  (both 60k runs not finished -- rule 2 cannot be applied)')

    print(f"\nPHASE 2 PICK: {pick['model']}  DoLP={pick['dolp']}  <- {reason}")

    if '--dry-run' in sys.argv:
        print('(dry run: no config written)')
        return
    out_path = f"p2_{pick['model']}.json"
    cfg = write_full_train(pick, out_path)
    print(f"wrote {out_path}: full data, {cfg['hparams']['training_epochs']} epochs, "
          f"save_top_k={cfg['checkpoint_save_top_k']}, ckpts -> {cfg['checkpoint_dir']}")


if __name__ == '__main__':
    main()
