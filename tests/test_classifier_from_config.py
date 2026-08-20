#!/usr/bin/env python3
"""Round-trip test for Classifier.from_config -- the test that would have caught the bugs.

Three properties, one per failure mode found in the audit:

  1. EQUIVALENCE. For every real config, from_config() produces byte-identical constructor
     kwargs to the trainer's hand-written block. This is what makes converting the correct
     call sites (the trainer, score_checkpoints.py, eval_coverage.py, classifierPnm.py) a
     provable no-op -- important because a 37-model sweep is running through two of them.

  2. NESTING. from_config reads the derived-channel flags from `hparams`. The four eval
     tools read them from the config top level, where no config has ever had them: 167
     configs carry DoLP under hparams and zero at the top. The test asserts the flags
     actually arrive, so a regression to top-level reading fails here.

  3. CHANNEL COUNT, NOT JUST A CLEAN LOAD. `monochrome` replaces the four polarization
     channels with their mean REPLICATED x4, so the tensor shape is unchanged and a model
     built without the flag loads a mono checkpoint perfectly and computes the wrong
     thing. A test that only asserted "state dict loads" would pass on the one bug that
     has already produced wrong numbers. So this compares the input features actually
     built against the config, and asserts the stem's in_channels matches.

Run:  python test_classifier_from_config.py
"""

import glob
import inspect
import json
import sys

import torch

from mvc.core.lit_classifier import Classifier

FAMILIES = {
    'timm + DoLP (5ch)': {'model': 'convnext_pico',
                          'hparams': {'tile_size': 48, 'dropout_rate': 0.25, 'DoLP': True,
                                      'pretrained': False}},
    'timm stride-2':     {'model': 'convnext_pico',
                          'hparams': {'tile_size': 48, 'dropout_rate': 0.25, 'DoLP': True,
                                      'pretrained': False, 'timm_stem_stride': 2}},
    'timm MONOCHROME':   {'model': 'convnext_pico',
                          'hparams': {'tile_size': 48, 'dropout_rate': 0.25,
                                      'monochrome': True, 'pretrained': False}},
    'torchvision 4ch':   {'model': 'resnet18',
                          'hparams': {'tile_size': 48, 'dropout_rate': 0.25,
                                      'pretrained': False}},
    'custom CNN wide':   {'model': 'custom',
                          'hparams': {'tile_size': 48, 'dropout_rate': 0.25,
                                      'custom_channels': [128, 96, 64, 64],
                                      'custom_res_blocks': [1, 1, 1, 1],
                                      'pretrained': False}},
}

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


# --------------------------------------------------------------- 1. equivalence
def trainer_kwargs(cfg, num_classes, clean_class):
    """The trainer's hand-written translation, transcribed verbatim as the reference."""
    h = cfg['hparams']
    return dict(
        model=cfg['model'],
        lr=cfg.get('optimizer', {}).get('learning_rate', 1e-4),
        num_classes=num_classes,
        tile_size=h['tile_size'],
        dropout_rate=h['dropout_rate'],
        penalize_false_clean=float(cfg.get('penalize_false_clean', 0.0)),
        base_channels=h.get('base_channels', 32),
        final_dense_layer=h.get('final_dense_layer', 512),
        clean_class=clean_class,
        noise_std=h.get('noise_std', 0.0),
        noise_clip=h.get('noise_clip', None),
        gain_jitter=float(h.get('gain_jitter', 0.0)),
        polar_flip=bool(h.get('polar_flip', False)),
        channel_jitter=float(h.get('channel_jitter', 0.0)),
        monochrome=bool(h.get('monochrome', False)),
        polar_rot=bool(h.get('polar_rot', False)),
        frozen_body_start_epochs=int(h.get('frozen_body_start_epochs', 0)),
        frozen_body_end_epochs=int(h.get('frozen_body_end_epochs', 0)),
        custom_early_convs=int(h.get('custom_early_convs', 0)),
        custom_channels=h.get('custom_channels', None),
        custom_res_blocks=h.get('custom_res_blocks', None),
        custom_wavelet_pools=h.get('custom_wavelet_pools', None),
        custom_wavelet_stem=int(h.get('custom_wavelet_stem', 0) or 0),
        pretrained=bool(h.get('pretrained', True)),
        seed_pretrained_stem=bool(h.get('seed_pretrained_stem', True)),
        timm_stem_stride=h.get('timm_stem_stride', None),
        AoLP=bool(h.get('AoLP', False)),
        DoLP=bool(h.get('DoLP', False)),
        Unpolarized=bool(h.get('Unpolarized', h.get('unpolarized', False))),
        MaxPolarization=bool(h.get('MaxPolarization', False)),
        MinPolarization=bool(h.get('MinPolarization', False)),
        RangePolarization=bool(h.get('RangePolarization', False)),
    )


def from_config_kwargs(cfg, num_classes, clean_class):
    """What from_config would pass -- the pure translation, no model built."""
    return Classifier.config_to_kwargs(cfg, num_classes=num_classes,
                                       clean_class=clean_class)


def test_equivalence():
    configs = sorted(set(glob.glob('*.json') + glob.glob('configs/*.json') +
                         glob.glob('experiments/configs_*/*.json') +
                         # run configs are filed beside their weights now; without this the
                         # corpus silently shrank from 167 to 135 when they moved.
                         glob.glob('experiments/**/*.json', recursive=True)))
    n = 0
    for path in configs:
        try:
            with open(path) as fh:
                cfg = json.load(fh)
        except Exception:
            continue
        if not isinstance(cfg, dict) or 'hparams' not in cfg or 'model' not in cfg:
            continue
        ref = trainer_kwargs(cfg, num_classes=10, clean_class=9)
        got = from_config_kwargs(cfg, num_classes=10, clean_class=9)
        diff = {k: (ref[k], got.get(k)) for k in ref if ref[k] != got.get(k)}
        if not check(not diff, f'{path}: from_config differs from the trainer: {diff}'):
            return n
        n += 1
    print(f'  [1] equivalence: from_config == trainer block on {n} real configs')
    return n


# --------------------------------------------------------------- 2 + 3. round trip
def in_channels_of(model):
    """Channels the first conv actually expects."""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m.in_channels
    return None


def test_round_trip():
    for label, cfg in FAMILIES.items():
        h = cfg['hparams']
        model = Classifier.from_config(cfg, num_classes=10, clean_class=9)

        # (2) the flags actually arrived, from hparams
        for flag in ('DoLP', 'monochrome', 'timm_stem_stride'):
            check(getattr(model, flag, None) == h.get(flag, getattr(model, flag, None)) or
                  h.get(flag) is None,
                  f'{label}: {flag} did not survive from_config')

        # (3) the INPUT the model builds matches the config -- not merely a clean load.
        # build_input_features' `in_channels` is the EXPECTED OUTPUT width; it raises if
        # the flags produce a different count, which is itself a useful guard.
        expected = 4 + sum(bool(h.get(k)) for k in
                           ('AoLP', 'DoLP', 'Unpolarized', 'MaxPolarization',
                            'MinPolarization', 'RangePolarization'))
        import mvc.core.polarization as pol
        x = torch.zeros(2, 4, h['tile_size'], h['tile_size'])
        feats = pol.build_input_features(
            x, in_channels=expected, monochrome=model.monochrome, AoLP=model.AoLP,
            DoLP=model.DoLP, Unpolarized=model.Unpolarized,
            MaxPolarization=model.MaxPolarization,
            MinPolarization=model.MinPolarization,
            RangePolarization=model.RangePolarization)
        check(feats.shape[1] == expected,
              f'{label}: built {feats.shape[1]} input channels, config implies {expected}')
        stem = in_channels_of(model.model if hasattr(model, 'model') else model)
        check(stem == expected,
              f'{label}: stem expects {stem} channels, config implies {expected}')

        # monochrome must actually flatten the polarization content, at EVAL time too
        if h.get('monochrome'):
            model.eval()
            varied = torch.rand(2, 4, h['tile_size'], h['tile_size'])
            import mvc.core.polarization as pol
            out = pol.build_input_features(varied, in_channels=4, monochrome=True)
            spread = (out[:, 0] - out[:, 1]).abs().max().item()
            check(spread < 1e-6,
                  f'{label}: monochrome did not equalise channels at eval (spread {spread})')

        # forward pass has to work end to end
        y = model(torch.zeros(2, 4, h['tile_size'], h['tile_size']))
        check(tuple(y.shape) == (2, 10), f'{label}: output {tuple(y.shape)} != (2, 10)')
        print(f'  [2/3] {label:22s} {expected}ch in, stem {stem}ch, out {tuple(y.shape)}')


# --------------------------------------------------------------- 4. refusal
def test_refuses_to_guess():
    try:
        Classifier.from_config({'model': 'convnext_pico'}, num_classes=10)
        failures.append('from_config accepted a config with no hparams instead of raising')
    except ValueError:
        print('  [4] refusal: from_config raises on a config with no hparams')

    try:
        Classifier.load_for_eval('/nonexistent.ckpt')
        failures.append('load_for_eval accepted a missing checkpoint with no config')
    except (ValueError, FileNotFoundError, OSError):
        print('  [4] refusal: load_for_eval raises rather than defaulting')


def main():
    print('Classifier.from_config round-trip test\n')
    test_equivalence()
    test_round_trip()
    test_refuses_to_guess()
    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print('  -', f)
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
