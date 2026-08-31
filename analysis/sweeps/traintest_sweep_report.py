#!/usr/bin/env python3
"""Report for the in-distribution train/test split campaign (analysis/sweeps/traintest_sweep.py).

⚠️  READ BEFORE QUOTING A NUMBER FROM THIS TABLE. The validation set here is a random,
TILE-level split of the training data (dataloader.frozen_tile_split) — leakage of sibling
tiles is EXPECTED and accepted, not a bug. This measures an IN-DISTRIBUTION FIT CEILING: how
well an architecture can fit this problem at all, independent of generalization.

TWO NUMBERS, NOT COMPARABLE THE SAME WAY:
  - miss@FA5 (score_checkpoints.py) is AGGREGATE and incidence-weighted -- dominated by
    whichever class has the most tiles. A smoke test found this barely moves relative to the
    factory val's miss@FA5 (15.08 vs 16.92 for squeezenet1_1), because the class mix differs
    between the two validation sets (53.4% welding here vs 87.6% there) enough to swamp the
    leakage effect on an incidence-weighted number. It is still useful for ranking WITHIN this
    campaign, just not for sizing the generalization-vs-underfitting gap.
  - macro_detect@FA5 (eval_traintest_split.py) is a per-class macro, the same KIND of number
    as the TIER_A coverage macro in 21-8/24-8-report.md. This IS the number to compare against
    coverage -- same smoke test: 83.57% here vs 67.79% TIER_A coverage for squeezenet1_1, a
    real +15.78 gap, which is the shape leakage is expected to produce.

Only requires the factory-style threshold_curve.json (mirrors model_sweep_report.py's
scores(), not full_zoo_report.py's, which silently drops any run missing a coverage table and
would drop every run here — there is no coverage table in this campaign at all).
macro_detect@FA5 is blank for a run that has not been scored by eval_traintest_split.py yet.

Usage:  python traintest_sweep_report.py [--all]
"""

import json
import os
import re
import sys

from mvc.core.artifact_paths import find_artifact
from mvc.core.metrics import miss_at_fa

from .traintest_sweep import model_list

TARGET_HZ = 23.0
GPU_SCALE = 1.6                  # A6000 bench -> RTX 5090, conservative


def best_epoch(ckpt_dir, max_epoch=1):
    best = None
    if not os.path.isdir(ckpt_dir):
        return None
    for b in os.listdir(ckpt_dir):
        m_ep = re.search(r'epoch=(\d+)', b)
        m_mon = re.search(r'val_detect_auroc=([0-9]+\.[0-9]+)', b)
        if not (m_ep and m_mon) or int(m_ep.group(1)) > max_epoch:
            continue
        cand = (float(m_mon.group(1)), int(m_ep.group(1)))
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1] if best else None


def scores(name, model):
    sfx = model.replace('/', '_')
    # macro5 (eval_traintest_split.py) and miss5 (score_checkpoints.py) are two
    # INDEPENDENT writers -- eval_traintest_split picks its own checkpoint by globbing
    # datasets/mix_ckpts/ directly, it does not read score_checkpoints' curve at all. A
    # bug in score_checkpoints (fixed 2026-08-31: it looked up its own curve file by an
    # unsanitised `timm/x` model name and crashed before writing the best epoch's curve
    # for 12 of 14 timm/* models in this campaign) must not hide a macro5 that scored
    # fine -- so each is looked up independently and either can be None on its own,
    # rather than one missing artifact dropping the whole row.
    ep = best_epoch(f'datasets/mix_ckpts/{name}_{sfx}')
    if ep is None:
        return None
    out = {'epoch': ep, 'miss5': None, 'macro5': None}
    curve = find_artifact(f'{name}_ep{ep}_{sfx}_threshold_curve.json')
    if curve:
        out['miss5'] = miss_at_fa(curve)[0.05]
    detect = find_artifact(f'{name}_{sfx}_traintest_detect.json')
    if detect:
        out['macro5'] = json.load(open(detect)).get('macro_detect_at_fa5')
    return out if (out['miss5'] is not None or out['macro5'] is not None) else None


def bench():
    out = {}
    for fname in ('phase4_inference_bench.json', 'zoo_inference_bench.json'):
        p = find_artifact(fname)
        if not p:
            continue
        for r in json.load(open(p))['rows']:
            key = r['model']
            if key not in out or r.get('variant', '').lower() == 'fused':
                out[key] = r
    return out


def main():
    show_all = '--all' in sys.argv
    hz = bench()
    models = model_list()

    inc_tag, inc_model = models[0]
    assert inc_model == 'convnext_pico', 'model_list() must list the incumbent first'
    inc = scores(f'tt{inc_tag}', inc_model)
    if not inc or inc['miss5'] is None:
        print('incumbent (ttcnxpico) not scored yet; run it first')
        return
    base = inc['miss5']

    base_macro = inc['macro5']

    print('\nIn-distribution train/test split — Aug26_78K, 4ch+DoLP, 10-class, RANDOM '
          'tile-level split,')
    print('seed 42, 2 epochs. Leakage expected; this is a fit-ceiling check.')
    print(f'Incumbent convnext_pico (fresh run, ttcnxpico) = {base:.2f} miss@FA5'
          + (f', {base_macro:.2f}% macro detect@FA5' if base_macro is not None else '')
          + '.\n')

    hdr = (f'{"model":40s} {"miss@FA5":>9s} {"Δ inc":>7s} {"macro5%":>8s} {"Δ inc":>7s} '
           f'{"5090 Hz":>8s} {"ships":>6s} {"params M":>9s}')
    print(hdr + '\n' + '-' * len(hdr))

    def line(tag, model, s):
        b = hz.get(model, {})
        h16 = b.get('hz_step16', 0.0)
        h5090 = h16 * GPU_SCALE
        ships = 'yes' if h5090 >= TARGET_HZ else 'NO'
        params = b.get('params_M')
        pstr = f'{params:9.2f}' if params is not None else f'{"--":>9}'
        if s is None:
            if not show_all:
                return
            print(f'{model:40s} {"--":>9s} {"--":>7s} {"--":>8s} {"--":>7s} '
                  f'{h5090:8.1f} {ships:>6s} {pstr} not run yet')
            return
        # miss5 and macro5 are independent writers (see scores()' docstring) -- a run can
        # have one without the other, most commonly a macro5 with no miss5 (the
        # score_checkpoints.py timm/* bug, fixed 2026-08-31, backfill pending on 12 runs).
        if s['miss5'] is not None:
            mi5str, midstr = f'{s["miss5"]:9.2f}', f'{s["miss5"] - base:+7.2f}'
        else:
            mi5str, midstr = f'{"--":>9s}', f'{"--":>7s}'
        m5 = s['macro5']
        if m5 is not None and base_macro is not None:
            mstr, mdstr = f'{m5:8.2f}', f'{m5 - base_macro:+7.2f}'
        else:
            mstr, mdstr = f'{"--":>8s}', f'{"--":>7s}'
        note = '' if s['miss5'] is not None else '  (miss@FA5 pending backfill)'
        print(f'{model:40s} {mi5str} {midstr} {mstr} {mdstr} '
              f'{h5090:8.1f} {ships:>6s} {pstr}{note}')

    for tag, model in models:
        s = scores(f'tt{tag}', model)
        line(tag, model, s)

    print('\nmacro5% = macro detect@FA5 from eval_traintest_split.py, the number comparable '
          'to TIER_A coverage.')
    print('ships: 5090 Hz >= 23 Hz at tiling step 16 (same gate as every other report).')
    print('Pass --all to also print rows that have not finished training yet.')


if __name__ == '__main__':
    main()
