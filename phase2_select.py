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

from Metrics import miss_at_fa  # noqa: F401  (re-exported: older callers import it here)
from artifact_paths import find_artifact

RESULT_KEYS = ('confusion_matrix', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment')


# miss_at_fa now lives in Metrics.py. It used to be DEFINED here, and
# model_sweep_report.py and modifier_sweep.py imported the project's KPI from this
# one-off selection script -- which meant this file could never be moved or deleted.


def collect(prefixes=('tz', 'p1n', 'p1b', 'p2', 'p3')):
    """Every screen that has both a config and a finished threshold curve."""
    out = []
    for pfx in prefixes:
        for cfg_path in sorted(glob.glob(f'{pfx}_*.json')):
            base = cfg_path[:-len('.json')]
            if base.endswith('_confusion') or base.endswith('_threshold_curve'):
                continue
            cfg = json.load(open(cfg_path))
            model = cfg['model']
            curve_name = f"{cfg['name']}_{model}_threshold_curve.json"
            curve = find_artifact(curve_name)
            if curve is None:
                print(f'  (skip {cfg_path}: no {curve_name} yet)')
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

    already = [p for p in glob.glob('p2_*.json')
               if not p.endswith(('_confusion.json', '_threshold_curve.json'))]
    if already:
        # Phase 2 has already run. Screens that landed afterwards (p3_*) can move this
        # line, and a reader glancing at the tail of a log should not mistake it for the
        # decision that was actually taken.
        print(f"\nNOTE Phase 2 ALREADY RAN as {already[0]}. The line below is what the rule "
              f"would pick from TODAY's screens, not what was trained.")
    print(f"\nPHASE 2 PICK: {pick['model']}  DoLP={pick['dolp']}  <- {reason}")

    if '--dry-run' in sys.argv:
        print('(dry run: no config written)')
        return
    # Once a Phase-2 config exists the full train has been launched against it, and the
    # trainer writes its RESULTS back into that same file. Re-deriving it would destroy
    # them -- and the pick can legitimately move as later screens (p3_*) land.
    existing = glob.glob('p2_*.json')
    existing = [p for p in existing if not p.endswith(('_confusion.json', '_threshold_curve.json'))]
    if existing and '--force' not in sys.argv:
        sys.exit(f"refusing to overwrite an existing Phase-2 config: {existing}. "
                 f"Pass --force only if you intend to discard that run's results.")
    out_path = f"p2_{pick['model']}.json"
    cfg = write_full_train(pick, out_path)
    print(f"wrote {out_path}: full data, {cfg['hparams']['training_epochs']} epochs, "
          f"save_top_k={cfg['checkpoint_save_top_k']}, ckpts -> {cfg['checkpoint_dir']}")


if __name__ == '__main__':
    main()
