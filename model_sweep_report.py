#!/usr/bin/env python3
"""Backbone sweep, stage 1: eliminate candidates, and cost every survivor in Hz.

TWO RULES THIS REPORT ENFORCES, BECAUSE THE NUMBERS INVITE BREAKING BOTH
-----------------------------------------------------------------------
1. STAGE 1 IS n=1 AND CANNOT RANK. Full-budget seed noise on the incumbent is ~1.8
   miss@FA5 (two anchor seeds, 2026-08-11). Anything within ~2 sd of the incumbent is
   reported as INDISTINGUISHABLE, in one undifferentiated group, deliberately not sorted
   into an order that would read as a ranking. Only candidates outside that band are
   called eliminated. The gap column is printed so the reader can see how little most of
   them differ.

2. ACCURACY WITHOUT Hz IS NOT A RESULT. The KPI is miss@FA5 subject to 23 Hz on an RTX
   5090; a backbone that wins on miss@FA5 and misses the frame rate has not won anything.
   Every row therefore carries throughput at each step size, and the verdict column is
   joint. Hz figures come from the A6000 bench (`phase4_inference_bench.json`) -- the
   architecture half of it transfers even though its accuracy half does not -- and are
   additionally shown scaled by an assumed 1.6x for the 5090.

Coverage is reported alongside because at full budget the coverage macro is 2.4x more
seed-stable than the factory KPI (range 0.76 vs 1.83 across the two anchor seeds), so a
coverage difference of a given size is stronger evidence than the same difference on the
headline number.

Usage:  python model_sweep_report.py
"""

import json
import os
import re
from statistics import mean

from artifact_paths import find_artifact
from Metrics import miss_at_fa

# tag -> (model, note); mirrors model_sweep.CANDIDATES
from model_sweep import CANDIDATES, REPARAM

INCUMBENT = ('convnext_pico', 'anc', 'incumbent')
# Measured on the two full-budget anchor seeds; see PLAN.md 2026-08-11.
SEED_SD = 1.83 / (2 ** 0.5)      # range of 2 -> sd estimate, 1 df. Deliberately crude.
TARGET_HZ = 23.0
GPU_SCALE = 1.6                  # A6000 bench -> RTX 5090, conservative


def epoch_scores(name, model, ep):
    """Scores at a SPECIFIC epoch, for the low-variance epoch-0 comparison."""
    curve = find_artifact(f'{name}_ep{ep}_{model}_threshold_curve.json')
    return miss_at_fa(curve)[0.05] if curve else None


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
    ep = best_epoch(f'datasets/mix_ckpts/{name}_{model}')
    if ep is None:
        return None
    curve = find_artifact(f'{name}_ep{ep}_{model}_threshold_curve.json')
    cov = find_artifact(f'{name}_{model}_coverage.json') or \
        find_artifact(f'epochcov_{name}_ep{ep}.json')
    out = {'epoch': ep, 'miss5': None, 'tier_a': None}
    if curve:
        out['miss5'] = miss_at_fa(curve)[0.05]
    if cov:
        rows = json.load(open(cov))['rows']
        v = [r['detect_at_fa5'] for r in rows
             if r['tier'] == 'TIER_A' and r['class'] != 'class_clean'
             and r.get('detect_at_fa5') is not None]
        out['tier_a'] = mean(v) if v else None
    return out


def bench():
    b = json.load(open('phase4_inference_bench.json'))
    out = {}
    for r in b['rows']:
        # A reparameterizable model deploys FUSED; that is the number that counts.
        key = r['model']
        if key not in out or r['variant'].lower() == 'fused':
            out[key] = r
    return out


def main():
    hz = bench()
    inc_model, inc_name, _ = INCUMBENT
    inc = scores(inc_name, inc_model)
    if not inc or inc['miss5'] is None:
        print('incumbent not scored; run the anchor first')
        return
    base = inc['miss5']

    print('\nBackbone sweep stage 1 — Aug26_78K, 4ch+DoLP, 10-class, coverage carved out,')
    print('seed 42, 2 epochs. n=1 PER CANDIDATE: this table ELIMINATES, it does not rank.')
    print(f'Incumbent convnext_pico = {base:.2f} miss@FA5 (seed 42). Full-budget seed sd '
          f'≈ {SEED_SD:.2f}, so the\nindistinguishable band is ±{2 * SEED_SD:.2f}.')
    print(f'Hz: A6000 bench; "5090" column = x{GPU_SCALE} assumed. Target {TARGET_HZ:.0f} Hz.\n')

    hdr = (f'{"model":24s} {"miss@FA5":>9s} {"Δ inc":>7s} {"TIER_A":>7s} '
           f'{"Hz@16":>7s} {"Hz@18":>7s} {"5090@16":>8s} {"verdict":>26s}')
    print(hdr + '\n' + '-' * len(hdr))

    rows = []
    for tag, model, _hz16, _prior, _note in CANDIDATES:
        s = scores(f'ms{tag}', model)
        rows.append((tag, model, s))

    def line(model, s, label=None):
        b = hz.get(model, {})
        h16, h18 = b.get('hz_step16'), b.get('hz_step18')
        if s is None or s['miss5'] is None:
            print(f'{model:24s} {"--":>9s} {"--":>7s} {"--":>7s} '
                  f'{h16 or 0:7.1f} {h18 or 0:7.1f} {(h16 or 0) * GPU_SCALE:8.1f} '
                  f'{"not run yet":>26s}')
            return
        d = s['miss5'] - base
        ta = s['tier_a']
        fast = (h16 or 0) * GPU_SCALE >= TARGET_HZ
        if label:
            v = label
        elif d > 2 * SEED_SD:
            v = 'ELIMINATED (worse)'
        elif d < -2 * SEED_SD:
            v = 'BEATS INCUMBENT — verify n=3'
        else:
            v = 'indistinguishable' + ('' if fast else ' / TOO SLOW')
        star = '*' if model in REPARAM else ' '
        print(f'{model + star:24s} {s["miss5"]:9.2f} {d:+7.2f} '
              f'{ta if ta is None else round(ta, 2):>7} '
              f'{h16 or 0:7.1f} {h18 or 0:7.1f} {(h16 or 0) * GPU_SCALE:8.1f} {v:>26s}')

    line(inc_model, inc, label='INCUMBENT')
    for _tag, model, s in rows:
        line(model, s)

    # ---- the low-variance view ----------------------------------------------------
    # Measured 2026-08-13 on the anchor arm's three seeds: seed sd is 0.37 at epoch 0 and
    # 1.62 at epoch 1 -- divergence grows 4.4x with the second epoch, while the mean moves
    # only 0.38 (9.48 -> 9.10), which is itself inside the noise. So epoch 0 is a far more
    # PRECISE instrument, and it is free: score_checkpoints.py already scores every epoch.
    #
    # It is not a strictly better one. Epoch 0 measures how fast a backbone learns in one
    # pass, not how good it is when trained -- and that distinction bites exactly here,
    # since the dev box found ConvNeXt V2's FCMAE init transfers slowly. A slow starter
    # would be unfairly eliminated on this column alone. Read the two together: epoch 0 to
    # SEPARATE candidates, monitored-best to check the separation survives training.
    print(f'\n\n{"=" * 70}\nEPOCH-0 VIEW — same runs, 4.4x lower seed variance '
          f'(sd 0.37 vs 1.62)\n'
          f'2-sd bar here is ~0.74, against ~3.2 at the monitored-best epoch.\n'
          f'Measures early convergence, NOT final quality — a slow-starting backbone\n'
          f'(e.g. an SSL init) is penalised here and must not be eliminated on this alone.\n'
          + '=' * 70)
    inc_ep0 = epoch_scores(inc_name, inc_model, 0)
    if inc_ep0 is None:
        print('incumbent epoch-0 curve missing')
        return
    print(f'{"model":24s} {"ep0 miss@FA5":>13s} {"Δ inc":>7s}   verdict (2-sd bar 0.74)')
    print('-' * 70)
    print(f'{inc_model:24s} {inc_ep0:13.2f} {0.0:+7.2f}   INCUMBENT')
    for tag, model, _s in rows:
        v = epoch_scores(f'ms{tag}', model, 0)
        if v is None:
            print(f'{model:24s} {"--":>13s} {"--":>7s}   not run yet')
            continue
        d = v - inc_ep0
        verdict = ('separated (worse)' if d > 0.74 else
                   'separated (BETTER)' if d < -0.74 else 'indistinguishable')
        print(f'{model:24s} {v:13.2f} {d:+7.2f}   {verdict}')

    print('\n* reparameterizable: Hz shown is the FUSED figure. Training runs the slow')
    print('  multi-branch graph; a deployment that forgets to fuse gets ~1/5 of that Hz.')
    print('\nSTAGE 2: seeds 1337 and 7 on everything not eliminated, plus the incumbent,')
    print('for the same n=3 paired comparison the stride-2 question is getting. Until then')
    print('no ordering among the indistinguishable group may be quoted as a result.')


if __name__ == '__main__':
    main()
