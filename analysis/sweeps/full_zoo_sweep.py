#!/usr/bin/env python3
"""One run of EVERY model the codebase can build. Breadth scan, single seed, no statistics.

This is deliberately a different kind of experiment from the 11-backbone sweep. That one
was built to support verdicts and so carried a two-stage design and an elimination-only
stage 1. This one is a MAP: one run each of all 37 remaining architectures, seed 42, to see
the shape of the field. Requested explicitly on that basis.

WHAT IT CAN AND CANNOT SUPPORT. Full-budget seed sd is 1.01 on the factory KPI and 0.43 on
coverage, so a single run locates a model to roughly +-2 and +-0.9 respectively. Gaps of a
few points are not orderings. Gaps of ten are. The previous sweep's useful findings were
all of the second kind -- lcnet_050 losing 13.9 points of weak-dent detection is not a
seed artifact -- so a breadth scan at n=1 is a reasonable instrument for finding the
large-effect outliers, which is what a map is for.

PROTOCOL IS IDENTICAL TO THE 11-BACKBONE SWEEP so the two tables merge into one: same
anchor config, 4ch+DoLP, 10-class scheme, coverage carve-out, pfc=0.5, seed 42, 2 epochs,
monitored-best checkpoint. Only `model` differs. 49 models total once merged.

ORDER IS FASTEST-FIRST, by measured inference throughput as a proxy for training cost.
Two reasons: a breadth scan is most useful when it covers the most ground per hour, and the
slowest models here are also the ones that cannot meet 23 Hz on the 5090 -- so stopping the
queue early costs only architectures that could not have shipped anyway.

All 37 were verified to build and forward at 5ch / 48px / 10-class with pretrained weights
before this file was written, and benched into `zoo_inference_bench.json`.

Usage:  python full_zoo_sweep.py [--dry-run]
"""

import copy
import json

from mvc.core.artifact_paths import find_artifact
import sys

BASE_CONFIG = 'anc_convnext_pico.json'
BENCH = 'zoo_inference_bench.json'
SEED = 42
EPOCHS = 2
TARGET_HZ = 23.0
GPU_SCALE = 1.6                 # A6000 bench -> RTX 5090, conservative
# convnext_pico trains 2 epochs in ~4.4 h at 22.5 Hz; scale by inverse throughput.
REF_HOURS, REF_HZ = 4.4, 22.5

RESULT_KEYS = ('confusion_matrix', 'classes', 'classes_int', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment',
               'training_started', 'training_finished', 'training_seconds', 'fit_seconds')

# Short tags keep artifact names readable -- the writers emit `{name}_{model}`.
TAGS = {
    'resnext50': 'rx50', 'resnet18': 'r18', 'resnet18_stem': 'r18stem',
    'resnet18_fullres': 'r18full', 'resnet18_hires': 'r18hi', 'resnet18_fine': 'r18fine',
    'convnext_tiny': 'cnxtiny', 'efficientnet_v2_s': 'effv2s', 'swin_v2_t': 'swin',
    'regnet_y_800mf': 'regy800', 'regnet_y_400mf': 'regy400',
    'mobilenet_v3_small': 'mnv3s', 'mobilenet_v3_large': 'mnv3l',
    'shufflenet_v2_x0_5': 'shuf05', 'shufflenet_v2_x1_0': 'shuf10',
    'squeezenet1_1': 'squeeze', 'efficientnet_b0': 'effb0', 'densenet121': 'dense121',
    'mnasnet0_5': 'mnas05', 'mnasnet1_0': 'mnas10', 'custom': 'customcnn',
    'timm/convnextv2_atto': 'v2atto', 'timm/convnextv2_femto': 'v2femto',
    'timm/convnextv2_pico': 'v2pico', 'timm/convnextv2_nano': 'v2nano',
    'timm/tinynet_c': 'tinyc', 'timm/tinynet_d': 'tinyd', 'timm/tinynet_e': 'tinye',
    'timm/efficientvit_b0': 'evitb0', 'timm/lcnet_100': 'lcnet100',
    'timm/mobilenetv3_small_100': 'tmnv3s', 'timm/efficientnet_lite0': 'lite0',
    'timm/hardcorenas_a': 'hcnas', 'timm/regnetx_002': 'regx002',
    'timm/semnasnet_075': 'semnas', 'timm/spnasnet_100': 'spnas',
    'timm/tf_mobilenetv3_small_minimal_100': 'tfmnv3min',
}


def main():
    dry = '--dry-run' in sys.argv
    with open(find_artifact(BASE_CONFIG) or BASE_CONFIG) as fh:
        base = json.load(fh)
    assert base['dataloader'].get('exclude_frames'), 'anchor lost its coverage carve-out'
    assert len(base['classes']) == 10, 'anchor is not on the 10-class scheme'

    # Resolved, not opened by bare name: the tidy moved this file to
    # experiments/configs_frozen/ and a raw open() here fails at the cwd. That failure is
    # SILENT where it matters -- run_full_zoo_sweep.sh fills its queue from this module via
    # `mapfile < <(python3 ...)`, and mapfile succeeds with empty input, so a resume would
    # report "queue: 0 models" and exit 0 having trained nothing.
    bench = find_artifact(BENCH)
    if not bench:
        raise SystemExit(f'{BENCH} not found: the queue is ordered by benched throughput '
                         f'and cannot be built without it')
    hz = {r['model']: r for r in json.load(open(bench))['rows']}
    models = sorted(TAGS, key=lambda m: -hz.get(m, {}).get('hz_step16', 0))

    total = 0.0
    print(f'{"model":40s} {"tag":10s} {"Hz@16":>7s} {"5090":>7s} {"ships?":>7s} {"~h":>6s}')
    for model in models:
        b = hz.get(model, {})
        h16 = b.get('hz_step16', 0.0)
        est = REF_HOURS * REF_HZ / h16 if h16 else float('nan')
        total += est if est == est else 0.0
        tag = TAGS[model]

        cfg = copy.deepcopy(base)
        for k in RESULT_KEYS:
            cfg.pop(k, None)
        cfg['model'] = model
        cfg['hparams']['seed'] = SEED
        cfg['hparams']['training_epochs'] = EPOCHS
        cfg['hparams'].pop('timm_stem_stride', None)
        name = f'fz{tag}'
        cfg['name'] = name
        cfg['checkpoint_save_top_k'] = EPOCHS
        # `timm/x` is not a legal path component.
        cfg['checkpoint_dir'] = f'datasets/mix_ckpts/{name}_{model.replace("/", "_")}'

        out = f'{name}_{model.replace("/", "_")}.json'
        ships = 'yes' if h16 * GPU_SCALE >= TARGET_HZ else 'NO'
        print(f'{model:40s} {tag:10s} {h16:7.1f} {h16 * GPU_SCALE:7.1f} {ships:>7s} {est:6.1f}')
        if not dry:
            with open(out, 'w') as fh:
                json.dump(cfg, fh, indent=2)

    print(f'\n{len(models)} configs {"planned" if dry else "written"} · seed {SEED} · '
          f'{EPOCHS} epochs · est. {total:.0f} h total')
    n_slow = sum(1 for m in models
                 if hz.get(m, {}).get('hz_step16', 0) * GPU_SCALE < TARGET_HZ)
    print(f'{n_slow} of them cannot meet {TARGET_HZ:.0f} Hz on the 5090 and are queued last, '
          f'so stopping early\ncosts only architectures that could not have shipped anyway.')


if __name__ == '__main__':
    main()
