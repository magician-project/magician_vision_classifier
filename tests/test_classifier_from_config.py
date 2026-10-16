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

# Runnable both as `python -m tests.test_classifier_from_config` and as
# `python tests/test_classifier_from_config.py`. Run directly, sys.path[0] is tests/ rather
# than the repo root, so `mvc` is not importable -- and this file has a __main__ block, so
# direct invocation is a supported entry point.
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        penalize_false_defect=float(cfg.get('penalize_false_defect', 0.0)),
        rank_loss_weight=float(cfg.get('rank_loss_weight', 0.0)),
        rank_hard_fraction=float(cfg.get('rank_hard_fraction', 0.2)),
        distill_teacher=cfg.get('distill_teacher') or None,
        distill_weight=float(cfg.get('distill_weight', 0.0)),
        distill_temperature=float(cfg.get('distill_temperature', 2.0)),
        base_channels=h.get('base_channels', 32),
        final_dense_layer=h.get('final_dense_layer', 512),
        clean_class=clean_class,
        noise_std=h.get('noise_std', 0.0),
        noise_clip=h.get('noise_clip', None),
        gain_jitter=float(h.get('gain_jitter', 0.0)),
        polar_flip=bool(h.get('polar_flip', False)),
        channel_jitter=float(h.get('channel_jitter', 0.0)),
        scale_jitter=float(h.get('scale_jitter', 0.0)),
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
        hier_d=int(h.get('hier_d', 128)),
        hier_n=int(h.get('hier_n', 2)),
        hier_m=int(h.get('hier_m', 2)),
        hier_tokens=int(h.get('hier_tokens', 8)),
        hier_heads=int(h.get('hier_heads', 1)),
        hier_type_residual=bool(h.get('hier_type_residual', False)),
        hier_sev_weight=(None if h.get('hier_sev_weight') is None
                         else float(h['hier_sev_weight'])),
        hier_hybrid=bool(h.get('hier_hybrid', False)),
        hier_aux_weight=float(h.get('hier_aux_weight', 1.0)),
        hier_detach_branches=bool(h.get('hier_detach_branches', False)),
        hier_sev_per_type=bool(h.get('hier_sev_per_type', False)),
        timm_pad_downsample=bool(h.get('timm_pad_downsample', False)),
        **({'class_names': list(cfg['classes'])}
           if str(cfg['model']).startswith('hier_') and cfg.get('classes') else {}),
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


# --------------------------------------------------------------- 5. hier head knobs
HIER_CLASSES = ['class_DeformationClassA', 'class_NegativeDentClassA',
                'class_NegativeDentClassB', 'class_WeldingClassA', 'class_clean']


def test_hier_knobs():
    """hier_heads changes NO parameter shapes (MultiheadAttention's projections are E x E
    whatever the head count), so a 2-head checkpoint would load cleanly into a 1-head model
    and silently compute something else -- the monochrome failure mode again. Assert the
    knobs reach the module, survive a real checkpoint round trip, and that the shape-
    identical variants really do produce different outputs from the same weights."""
    import tempfile
    base = {'model': 'hier_convnext_pico', 'classes': HIER_CLASSES,
            'hparams': {'tile_size': 48, 'dropout_rate': 0.25, 'DoLP': True,
                        'pretrained': False}}
    x = torch.rand(2, 4, 48, 48) * 255
    ref = None
    for heads, res in ((1, False), (2, False), (4, False), (1, True)):
        cfg = json.loads(json.dumps(base))
        cfg['hparams'].update(hier_heads=heads, hier_type_residual=res)
        model = Classifier.from_config(cfg, num_classes=len(HIER_CLASSES), clean_class=4)
        head = model.model.head
        tag = f'heads={heads} residual={res}'
        check(head.attn.num_heads == heads and head.type_residual == res,
              f'hier {tag}: module built with heads={head.attn.num_heads} '
              f'residual={head.type_residual}')
        n_params = sum(p.numel() for p in model.parameters())
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'm.ckpt')
            torch.save({'state_dict': model.state_dict(),
                        'hyper_parameters': dict(model.hparams)}, path)
            back = Classifier.load_for_eval(path)
        check(back.model.head.attn.num_heads == heads and
              back.model.head.type_residual == res,
              f'hier {tag}: lost in checkpoint round trip')
        if ref is None:
            ref = (model.state_dict(), n_params)
        else:
            check(n_params == ref[1], f'hier {tag}: {n_params} params != baseline {ref[1]}')
            # Same weights, different head config -> must not be the same function.
            model.load_state_dict(ref[0], strict=True)
            model.eval()
            base_m = Classifier.from_config(base, num_classes=len(HIER_CLASSES), clean_class=4)
            base_m.load_state_dict(ref[0], strict=True)
            base_m.eval()
            # Compare the head's raw outputs with an explicit tolerance. At random init the
            # severity branch moves the final probabilities by only ~1e-4, which sits at the edge
            # of allclose's default rtol -- the old full-output check passed or failed on noise
            # (2026-10-12). The raw logits differ by far more than float error when a knob is live.
            with torch.no_grad():
                ft = model.build_input_features(x)
                ta, sa = model.model.forward_heads(ft)
                tb, sb = base_m.model.forward_heads(ft)
                diff = max(float((ta - tb).abs().max()), float((sa - sb).abs().max()))
            check(diff > 1e-6, f'hier {tag}: head outputs identical to the 1-head model '
                               f'(max diff {diff:.2e}) -- knob inert')
        print(f'  [5] hier {tag:24s} params {n_params:,}  round trip ok')

    # An indivisible head count must refuse, not round down.
    cfg = json.loads(json.dumps(base))
    cfg['hparams'].update(hier_heads=3)
    try:
        Classifier.from_config(cfg, num_classes=len(HIER_CLASSES), clean_class=4)
        failures.append('hier_heads=3 on 16-wide tokens was accepted instead of raising')
    except ValueError:
        print('  [5] refusal: hier_heads=3 on 16-wide tokens raises')


# --------------------------------------------------------------- 6. hier split loss
def test_hier_sev_weight():
    """The split loss focal(type) + lambda * CE(sev|type) must be an exact decomposition:
    at gamma = 0, lambda = 1 it equals cross-entropy on the flat logits to float precision.
    Also: forward() still equals log P(type) + graded * log P(sev|type), lambda survives a
    checkpoint, lowering lambda lowers the loss on graded targets, and a non-hier model
    refuses the knob instead of silently ignoring it."""
    import tempfile
    import torch.nn.functional as F
    torch.manual_seed(0)
    x = torch.rand(8, 4, 48, 48) * 255
    # graded (NegativeDent A/B) and ungraded (Deformation, Welding, clean) targets
    y = torch.tensor([0, 1, 2, 3, 4, 1, 2, 4])

    def build(lam, loss='cross_entropy'):
        cfg = {'model': 'hier_convnext_pico', 'classes': HIER_CLASSES, 'loss': loss,
               'hparams': {'tile_size': 48, 'dropout_rate': 0.0, 'DoLP': True,
                           'pretrained': False, 'hier_heads': 2, 'hier_sev_weight': lam}}
        return Classifier.from_config(cfg, num_classes=len(HIER_CLASSES), clean_class=4,
                                      loss=loss).eval()

    m = build(1.0)
    feats = m.build_input_features(x)
    with torch.no_grad():
        log_type, log_sev = m.model.log_parts(feats)
        net = m.model
        flat = m.model(feats)
        check(torch.allclose(flat, log_type[:, net.type_idx] + net.graded * log_sev),
              'hier forward() != log P(type) + graded * log P(sev|type)')
        check(torch.allclose(flat.exp().sum(1), torch.ones(len(x)), atol=1e-5),
              'hier flat logits are not normalised log-probabilities')
        _, split = m.hier_split_loss(feats, y)
        ce = F.cross_entropy(flat, y)
    check(abs(split.item() - ce.item()) < 1e-5,
          f'split loss (gamma=0, lambda=1) {split.item():.6f} != flat CE {ce.item():.6f}')
    print(f'  [6] split loss gamma=0 lambda=1 == flat CE ({split.item():.5f} vs {ce.item():.5f})')

    m0 = build(0.0)
    m0.load_state_dict(m.state_dict())
    with torch.no_grad():
        _, split0 = m0.hier_split_loss(feats, y)
    check(split0.item() < split.item(), 'lambda=0 did not lower the loss on graded targets')

    mf = build(0.5, loss='focal')
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'm.ckpt')
        torch.save({'state_dict': mf.state_dict(), 'hyper_parameters': dict(mf.hparams)}, path)
        back = Classifier.load_for_eval(path)
    check(back.hier_sev_weight == 0.5, f'hier_sev_weight lost in round trip: {back.hier_sev_weight}')
    print('  [6] hier_sev_weight survives a checkpoint round trip; lambda=0 < lambda=1')

    try:
        Classifier.from_config({'model': 'convnext_pico',
                                'hparams': {'tile_size': 48, 'dropout_rate': 0.25,
                                            'pretrained': False, 'hier_sev_weight': 0.5}},
                               num_classes=10, clean_class=9)
        failures.append('hier_sev_weight on a non-hier model was accepted instead of raising')
    except ValueError:
        print('  [6] refusal: hier_sev_weight on convnext_pico raises')


# --------------------------------------------------------------- 7. hier hybrid head
def test_hier_hybrid():
    """hier_hybrid: P(clean) must come from the flat head and the defect split from the
    hier branches, as one normalised distribution. Asserts those identities, the exact
    parameter cost (one Linear feat->C over the 1-head hier model), the training loss
    composition, a checkpoint round trip, and the refusals."""
    import tempfile
    torch.manual_seed(0)
    C = len(HIER_CLASSES)
    clean = HIER_CLASSES.index('class_clean')
    x = torch.rand(6, 4, 48, 48) * 255
    y = torch.tensor([0, 1, 2, 3, 4, 2])

    def build(**hp):
        cfg = {'model': 'hier_convnext_pico', 'classes': HIER_CLASSES,
               'hparams': {'tile_size': 48, 'dropout_rate': 0.0, 'DoLP': True,
                           'pretrained': False, **hp}}
        return Classifier.from_config(cfg, num_classes=C, clean_class=clean).eval()

    plain, hyb = build(), build(hier_hybrid=True, hier_aux_weight=0.5)
    n_plain = sum(p.numel() for p in plain.parameters())
    n_hyb = sum(p.numel() for p in hyb.parameters())
    check(n_hyb - n_plain == 512 * C + C,
          f'hybrid adds {n_hyb - n_plain} params, expected one Linear 512->{C} ({512 * C + C})')

    net = hyb.model
    feats = hyb.build_input_features(x)
    with torch.no_grad():
        comb, flat, hier = net.hybrid_parts(feats)
        p, pf, ph = comb.exp(), flat.softmax(-1), hier.exp()
        d = [i for i in range(C) if i != clean]
        check(torch.allclose(p.sum(1), torch.ones(len(x)), atol=1e-5),
              'hybrid output is not a normalised distribution')
        check(torch.allclose(p[:, clean], pf[:, clean], atol=1e-6),
              'hybrid P(clean) != flat head P(clean)')
        check(torch.allclose(p[:, d] / p[:, d].sum(1, keepdim=True),
                             ph[:, d] / ph[:, d].sum(1, keepdim=True), atol=1e-5),
              'hybrid defect split != hier P(c | defect)')
        check(torch.allclose(net(feats), comb), 'forward() != hybrid_parts()[0]')
        want = hyb.criterion(flat, y) + 0.5 * hyb.criterion(hier, y)
    # training_step's loss, without Lightning's logger: same composition as want
    hyb.log = lambda *a, **k: None
    hyb.augment_train_batch = lambda t: t
    hyb.penalize_false_clean = 0.0
    with torch.no_grad():
        got = hyb.training_step((x, y), 0)
    check(abs(got.item() - want.item()) < 1e-5,
          f'hybrid training loss {got.item():.6f} != flat + 0.5*hier {want.item():.6f}')
    print(f'  [7] hybrid: +{n_hyb - n_plain} params, P(clean) from flat head, defect split '
          f'from hier, loss = flat + 0.5*hier ({got.item():.5f})')

    # Symmetric clean/defect penalties on top: penalize_false_clean on defect tiles,
    # penalize_false_defect (-log P(clean)) on clean tiles, each a mean over its subset.
    hyb.penalize_false_clean, hyb.penalize_false_defect = 1.0, 1.0
    with torch.no_grad():
        pc = comb.exp()[:, clean]
        is_c = (y == clean)
        want2 = (want - torch.log(1.0 - pc[~is_c] + 1e-8).mean()
                 - torch.log(pc[is_c]).mean())
        got2 = hyb.training_step((x, y), 0)
    check(abs(got2.item() - want2.item()) < 1e-4,
          f'penalised hybrid loss {got2.item():.6f} != base + false_clean + false_defect '
          f'{want2.item():.6f}')
    print(f'  [7] + penalize_false_clean 1.0 + penalize_false_defect 1.0 = {got2.item():.5f}')

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'm.ckpt')
        torch.save({'state_dict': hyb.state_dict(), 'hyper_parameters': dict(hyb.hparams)}, path)
        back = Classifier.load_for_eval(path)
    check(back.hier_hybrid and back.hier_aux_weight == 0.5 and back.model.hybrid,
          'hier_hybrid / hier_aux_weight lost in checkpoint round trip')
    with torch.no_grad():
        check(torch.allclose(back.model(feats), comb, atol=1e-6),
              'reloaded hybrid computes a different output')

    for bad, msg in (({'model': 'convnext_pico', 'hier_hybrid': True}, 'non-hier model'),
                     ({'model': 'hier_convnext_pico', 'hier_hybrid': True,
                       'hier_sev_weight': 0.5}, 'hybrid + sev_weight')):
        cfg = {'model': bad.pop('model'), 'classes': HIER_CLASSES,
               'hparams': {'tile_size': 48, 'dropout_rate': 0.25, 'pretrained': False, **bad}}
        try:
            Classifier.from_config(cfg, num_classes=C, clean_class=clean)
            failures.append(f'hier_hybrid on {msg} was accepted instead of raising')
        except ValueError:
            print(f'  [7] refusal: hier_hybrid on {msg} raises')


# --------------------------------------------------------------- 8. detached branches
def test_hier_detach_branches():
    """hier_detach_branches: the hier (type/severity) loss must put ZERO gradient on the body
    and the pooling, while the flat loss still trains them -- so detection learns exactly as
    in convnext_pico. Control: without detach the hier loss does reach the body. Outputs are
    identical either way (detach only cuts gradients)."""
    torch.manual_seed(0)
    C = len(HIER_CLASSES)
    clean = HIER_CLASSES.index('class_clean')
    x = torch.rand(4, 4, 48, 48) * 255
    y = torch.tensor([0, 2, 3, 4])

    def build(detach):
        cfg = {'model': 'hier_convnext_pico', 'classes': HIER_CLASSES,
               'hparams': {'tile_size': 48, 'dropout_rate': 0.0, 'DoLP': True,
                           'pretrained': False, 'hier_hybrid': True,
                           'hier_detach_branches': detach}}
        return Classifier.from_config(cfg, num_classes=C, clean_class=clean).train()

    def grads(m, which):
        m.zero_grad(set_to_none=True)
        comb, flat, hier = m.model.hybrid_parts(m.build_input_features(x))
        (m.criterion(flat, y) if which == 'flat' else m.criterion(hier, y)).backward()
        body = m.model.stem[0].weight.grad
        pool_ln = m.model.head.pool.norm.weight.grad
        branch = m.model.head.type_out.weight.grad
        nz = lambda g: g is not None and g.abs().sum().item() > 0
        return nz(body), nz(pool_ln), nz(branch)

    det, ref = build(True), build(False)
    ref.load_state_dict(det.state_dict())
    check(grads(det, 'hier') == (False, False, True),
          f'detached: hier loss gradients (body, pool LN, branch) = {grads(det, "hier")}, '
          f'expected (False, False, True)')
    check(grads(det, 'flat')[:2] == (True, True), 'detached: flat loss no longer trains the body')
    check(grads(ref, 'hier')[:2] == (True, True),
          'control: without detach the hier loss should reach the body')
    det.eval(); ref.eval()
    with torch.no_grad():
        check(torch.allclose(det(x), ref(x)), 'detach changed the forward output')
    check(det.hparams['hier_detach_branches'] is True, 'hier_detach_branches not in hparams')
    print('  [8] detached branches: hier loss -> body grad 0, flat loss -> body grad > 0; '
          'control reaches body; outputs identical')
    try:
        Classifier.from_config({'model': 'hier_convnext_pico', 'classes': HIER_CLASSES,
                                'hparams': {'tile_size': 48, 'dropout_rate': 0.0,
                                            'pretrained': False, 'hier_detach_branches': True}},
                               num_classes=C, clean_class=clean)
        failures.append('hier_detach_branches without hier_hybrid was accepted')
    except ValueError:
        print('  [8] refusal: hier_detach_branches without hier_hybrid raises')


# --------------------------------------------------------------- 9. padded downsamples
def test_pad_downsample():
    """timm_pad_downsample: at 48px stage 3 must see ALL 9 stage-2 positions (control: without
    it, (0,2), (2,0), (2,2) have no effect), output 2x2, with identical parameters and state-dict
    keys (pretrained weights still load), surviving a checkpoint round trip; plain timm too."""
    import tempfile
    torch.manual_seed(0)
    C = len(HIER_CLASSES)
    clean = HIER_CLASSES.index('class_clean')

    def build(pad, model='hier_convnext_pico'):
        cfg = {'model': model, 'classes': HIER_CLASSES,
               'hparams': {'tile_size': 48, 'dropout_rate': 0.0, 'DoLP': True, 'pretrained': False,
                           **({'hier_hybrid': True} if model.startswith('hier_') else {}),
                           'timm_pad_downsample': pad}}
        return Classifier.from_config(cfg, num_classes=C, clean_class=clean).eval()

    def reach(m):
        net = m.model
        x = m.build_input_features(torch.rand(1, 4, 48, 48) * 255)
        with torch.no_grad():
            f2 = net.stages[2](net.stages[1](net.stages[0](net.stem(x))))
            base = net.stages[3](f2)
            hits = []
            for r in range(3):
                for c in range(3):
                    g = f2.clone()
                    g[:, :, r, c] += 10 * torch.randn_like(g[:, :, r, c])
                    hits.append(not torch.allclose(net.stages[3](g), base))
        return hits, tuple(base.shape)

    pad, ref = build(True), build(False)
    hp, sp = reach(pad)
    hr, sr = reach(ref)
    check(all(hp), f'padded: stage-2 positions reaching stage 3: {sum(hp)}/9, expected 9')
    check(sum(hr) == 4, f'control: unpadded reaches {sum(hr)}/9, expected 4 (the bug)')
    check(sp == (1, 512, 2, 2) and sr == (1, 512, 1, 1), f'stage-3 shapes {sp} / {sr}')
    check(list(pad.state_dict()) == list(ref.state_dict()), 'padding changed state-dict keys')
    check(sum(p.numel() for p in pad.parameters()) == sum(p.numel() for p in ref.parameters()),
          'padding changed the parameter count')
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'm.ckpt')
        torch.save({'state_dict': pad.state_dict(), 'hyper_parameters': dict(pad.hparams)}, path)
        back = Classifier.load_for_eval(path)
    check(back.timm_pad_downsample and reach(back)[1] == (1, 512, 2, 2),
          'timm_pad_downsample lost in checkpoint round trip')
    flat = build(True, 'convnext_pico')
    y = flat(torch.rand(2, 4, 48, 48) * 255)
    check(tuple(y.shape) == (2, C), f'plain convnext_pico with padding: output {tuple(y.shape)}')
    print(f'  [9] padded downsamples: stage 3 reaches {sum(hp)}/9 stage-2 positions (unpadded '
          f'{sum(hr)}/9), output {sp[2]}x{sp[3]}; same params/keys; round trip and plain timm ok')


# --------------------------------------------------------------- 10. hard-negative ranking loss
def test_rank_loss():
    """rank_loss: matches a hand computation; shift-invariant in the detection score; only the
    hardest rank_hard_fraction of clean tiles (plus every defect) get gradient; raising defect
    scores lowers it; identical on logits and on log-probs (the hybrid's output); bad fraction
    refused."""
    import math
    import torch.nn.functional as F
    torch.manual_seed(0)
    C, clean = 5, 4
    m = Classifier.from_config({'model': 'convnext_pico', 'rank_loss_weight': 1.0,
                                'hparams': {'tile_size': 48, 'dropout_rate': 0.0, 'pretrained': False}},
                               num_classes=C, clean_class=clean)
    y = torch.tensor([4] * 10 + [0, 1, 2, 3, 0, 2])
    logits = torch.randn(len(y), C, requires_grad=True)

    lp = F.log_softmax(logits, 1)
    s = torch.logsumexp(lp[:, :clean], 1) - lp[:, clean]
    sc, sd = s[y == clean], s[y != clean]
    k = math.ceil(0.2 * len(sc))
    hard = torch.sort(sc, descending=True).values[:k]
    want = torch.stack([F.softplus(c - d) for d in sd for c in hard]).mean()
    got = m.rank_loss(logits, y)
    check(abs(got.item() - want.item()) < 1e-6, f'rank_loss {got.item():.6f} != hand {want.item():.6f}')

    shifted = logits.detach().clone()
    shifted[:, :clean] += 3.0                     # every s moves by the same amount
    check(abs(m.rank_loss(shifted, y).item() - got.item()) < 1e-5, 'rank_loss is not shift-invariant')
    check(abs(m.rank_loss(F.log_softmax(logits.detach(), 1), y).item() - got.item()) < 1e-6,
          'rank_loss differs on log-probs vs logits')

    got.backward()
    g = logits.grad.abs().sum(1)
    hard_idx = set(torch.topk(sc.detach(), k).indices.tolist())
    clean_rows = [i for i in range(len(y)) if y[i] == clean]
    nz_clean = {j for j, i in enumerate(clean_rows) if g[i] > 0}
    check(nz_clean == hard_idx, f'clean tiles with gradient {sorted(nz_clean)} != hardest {sorted(hard_idx)}')
    check(all(g[i] > 0 for i in range(len(y)) if y[i] != clean), 'a defect tile got no gradient')

    up = logits.detach().clone()
    up[y != clean, :clean] += 1.0                 # defects look more defective
    check(m.rank_loss(up, y).item() < got.item(), 'raising defect scores did not lower rank_loss')
    print(f'  [10] rank_loss: matches hand ({got.item():.5f}), shift-invariant, gradient only on the '
          f'{k} hardest of {len(sc)} clean + all defects, logits == log-probs')
    try:
        Classifier.from_config({'model': 'convnext_pico', 'rank_hard_fraction': 0.0,
                                'hparams': {'tile_size': 48, 'pretrained': False, 'dropout_rate': 0.0}},
                               num_classes=C, clean_class=clean)
        failures.append('rank_hard_fraction=0 was accepted')
    except ValueError:
        print('  [10] refusal: rank_hard_fraction=0 raises')


# --------------------------------------------------------------- 11. scale jitter augmentation
def test_scale_jitter():
    """scale_jitter: off by default and in eval; shapes/range kept; a uniform tile stays exactly
    uniform (so DoLP is untouched); the SAME geometric map hits all four channels (channel ratios
    preserved pixel-wise); and tiles really are resampled when it is on."""
    import mvc.core.polarization as pol
    torch.manual_seed(0)
    x = torch.randint(0, 256, (64, 4, 48, 48), dtype=torch.uint8)

    off = pol.augment_train_batch(x.clone(), training=True)
    check(torch.equal(off, x), 'all augmentations off but input changed')
    ev = pol.augment_train_batch(x.clone(), training=False, scale_jitter=0.15, scale_jitter_p=1.0)
    check(torch.equal(ev, x), 'scale_jitter active in eval mode')

    y = pol.augment_train_batch(x.clone(), training=True, scale_jitter=0.15, scale_jitter_p=1.0)
    xf = x.float() / 255
    check(y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1, 'shape/range broken')
    changed = (y - xf).abs().flatten(1).max(1).values > 1e-4
    check(changed.float().mean() > 0.9, f'only {changed.float().mean():.0%} of tiles were resampled at p=1')

    flat = torch.ones(8, 4, 48, 48) * torch.tensor([0.2, 0.5, 0.8, 0.3]).view(1, 4, 1, 1)
    yf = pol.augment_train_batch(flat.clone(), training=True, scale_jitter=0.15, scale_jitter_p=1.0)
    check(torch.allclose(yf, flat, atol=1e-6), 'a uniform tile did not stay uniform')
    check(torch.allclose(pol.calculate_DoLP(yf), pol.calculate_DoLP(flat), atol=1e-5),
          'DoLP of a uniform tile changed')

    base = torch.rand(8, 1, 48, 48) * 0.5 + 0.1
    prop = base * torch.tensor([1.0, 1.5, 0.7, 1.2]).view(1, 4, 1, 1)
    torch.manual_seed(1)
    yp = pol.augment_train_batch(prop.clone(), training=True, scale_jitter=0.15, scale_jitter_p=1.0)
    ratio = yp / yp[:, :1]
    check(torch.allclose(ratio, torch.tensor([1.0, 1.5, 0.7, 1.2]).view(1, 4, 1, 1).expand_as(ratio),
                         atol=1e-4), 'channels were not transformed identically')
    print(f'  [11] scale_jitter: off/eval identity, {changed.float().mean():.0%} of tiles resampled at p=1, '
          'uniform tile and DoLP preserved, identical map on all 4 channels')


# --------------------------------------------------------------- 12. per-type severity head
def test_sev_per_type():
    """hier_sev_per_type: with two graded types (NegativeDent, PositiveDent: A/B), the default
    head gives them IDENTICAL P(grade | type) (the flaw, asserted as the control); the per-type
    head gives them different ones; the output stays normalised; the extra cost is exactly
    (T - 1) x (d + 1) x S parameters; a default checkpoint cannot load into the per-type head;
    the knob survives a checkpoint round trip; non-hier models refuse it."""
    import tempfile
    torch.manual_seed(0)
    cls = ['class_NegativeDentClassA', 'class_NegativeDentClassB', 'class_PositiveDentClassA',
           'class_PositiveDentClassB', 'class_WeldingClassA', 'class_clean']
    C, clean = len(cls), cls.index('class_clean')

    def build(per_type):
        cfg = {'model': 'hier_convnext_pico', 'classes': cls,
               'hparams': {'tile_size': 48, 'dropout_rate': 0.0, 'DoLP': True, 'pretrained': False,
                           'hier_hybrid': True, 'hier_sev_per_type': per_type}}
        return Classifier.from_config(cfg, num_classes=C, clean_class=clean).eval()

    base, pt = build(False), build(True)
    x = torch.rand(4, 4, 48, 48) * 255
    neg, pos = [cls.index('class_NegativeDentClassA'), cls.index('class_NegativeDentClassB')], \
               [cls.index('class_PositiveDentClassA'), cls.index('class_PositiveDentClassB')]

    def grades(m):
        with torch.no_grad():
            _, log_sev = m.model.log_parts(m.build_input_features(x))
        return log_sev[:, neg], log_sev[:, pos]

    bn, bp = grades(base)
    check(torch.allclose(bn, bp), 'control: default head should give NegDent and PosDent identical grades')
    pn, pp = grades(pt)
    check(not torch.allclose(pn, pp), 'per-type head still gives NegDent and PosDent identical grades')
    with torch.no_grad():
        f = pt.build_input_features(x)
        s = pt.model(f).exp().sum(1)
    check(torch.allclose(s, torch.ones_like(s), atol=1e-5), 'per-type output is not normalised')
    T, S, d = pt.model.head.num_types, pt.model.head.num_sevs, 128
    extra = sum(p.numel() for p in pt.parameters()) - sum(p.numel() for p in base.parameters())
    check(extra == (T - 1) * (d + 1) * S, f'per-type head adds {extra} params, expected {(T - 1) * (d + 1) * S}')
    try:
        pt.load_state_dict(base.state_dict(), strict=True)
        failures.append('a default checkpoint loaded into the per-type head')
    except RuntimeError:
        pass
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'm.ckpt')
        torch.save({'state_dict': pt.state_dict(), 'hyper_parameters': dict(pt.hparams)}, path)
        back = Classifier.load_for_eval(path)
    check(back.hier_sev_per_type and back.model.head.sev_per_type, 'hier_sev_per_type lost in round trip')
    try:
        Classifier.from_config({'model': 'convnext_pico', 'hparams': {'tile_size': 48, 'dropout_rate': 0.0,
                                'pretrained': False, 'hier_sev_per_type': True}}, num_classes=10, clean_class=9)
        failures.append('hier_sev_per_type on convnext_pico was accepted')
    except ValueError:
        pass
    print(f'  [12] per-type severity: default head gives both dent types identical grades (control), '
          f'per-type head does not; +{extra} params; old checkpoints refused; round trip ok')


# --------------------------------------------------------------- 13. distillation
def test_distill():
    """distill_*: the KD term is 0 when student == teacher and > 0 otherwise; T^2-scaled KL by
    hand; gradient reaches the student only; the teacher is loaded lazily, is frozen, and is NOT
    part of the student's state dict or parameters (so the checkpoint is a plain student one and
    load_for_eval never needs the teacher file); a weight with no teacher is refused; and the
    training loss with distillation equals the loss without it plus weight x KD."""
    import tempfile
    import torch.nn.functional as F
    torch.manual_seed(0)
    C, clean = 5, 4
    hp = {'tile_size': 48, 'dropout_rate': 0.0, 'DoLP': True, 'pretrained': False}
    with tempfile.TemporaryDirectory() as td:
        tpath = os.path.join(td, 'teacher.ckpt')
        teacher = Classifier.from_config({'model': 'convnext_pico', 'hparams': hp},
                                         num_classes=C, clean_class=clean)
        torch.save({'state_dict': teacher.state_dict(), 'hyper_parameters': dict(teacher.hparams)}, tpath)
        cfg = {'model': 'convnext_pico', 'hparams': hp, 'distill_teacher': tpath,
               'distill_weight': 0.5, 'distill_temperature': 3.0}
        m = Classifier.from_config(cfg, num_classes=C, clean_class=clean)
        check(not m._teacher, 'teacher loaded eagerly in __init__')

        s = torch.randn(6, C, requires_grad=True)
        t = torch.randn(6, C)
        check(m.distill_loss(t, t).abs().item() < 1e-6, 'KD of identical logits is not 0')
        want = 9.0 * (F.softmax(t / 3, 1) * (F.log_softmax(t / 3, 1) - F.log_softmax(s / 3, 1))).sum(1).mean()
        got = m.distill_loss(s, t)
        check(abs(got.item() - want.item()) < 1e-5, f'KD {got.item():.6f} != hand {want.item():.6f}')
        tt = t.clone().requires_grad_(True)
        m.distill_loss(s, tt).backward()
        check(s.grad is not None and s.grad.abs().sum() > 0 and tt.grad is None,
              'KD gradient must reach the student only')

        # one training step: loss == loss without KD + 0.5 * KD; teacher never joins the module
        m.train()
        n_params = sum(p.numel() for p in m.parameters())
        keys = set(m.state_dict())
        x = torch.randint(0, 256, (6, 4, 48, 48), dtype=torch.uint8)
        y = torch.tensor([0, 1, 2, 3, 4, 4])
        m.log = lambda *a, **k: None
        torch.manual_seed(1); with_kd = m.training_step((x, y), 0)
        m.distill_weight = 0.0
        torch.manual_seed(1); without = m.training_step((x, y), 0)
        m.distill_weight = 0.5
        with torch.no_grad():
            kd = m.distill_loss(m(x), m._get_teacher()(x))
        check(abs(with_kd.item() - without.item() - 0.5 * kd.item()) < 1e-4,
              f'loss with KD {with_kd.item():.5f} != without {without.item():.5f} + 0.5*KD {kd.item():.5f}')
        check(len(m._teacher) == 1 and not any(p.requires_grad for p in m._teacher[0].parameters()),
              'teacher not loaded once / not frozen')
        check(sum(p.numel() for p in m.parameters()) == n_params and set(m.state_dict()) == keys,
              'teacher leaked into the student parameters / state dict')
        spath = os.path.join(td, 'student.ckpt')
        torch.save({'state_dict': m.state_dict(), 'hyper_parameters': dict(m.hparams)}, spath)
        os.remove(tpath)          # the student checkpoint must load without the teacher file
        back = Classifier.load_for_eval(spath)
        check(not back._teacher and back.distill_weight == 0.5, 'load_for_eval loaded the teacher')
    try:
        Classifier.from_config({'model': 'convnext_pico', 'hparams': hp, 'distill_weight': 1.0},
                               num_classes=C, clean_class=clean)
        failures.append('distill_weight without distill_teacher was accepted')
    except ValueError:
        pass
    print(f'  [13] distillation: KD 0 on identical logits, matches hand ({got.item():.5f}), '
          'student-only gradient, loss = base + w*KD, teacher lazy/frozen/outside the state dict')

    # several teachers: target = mean of the softened distributions; one-element list == single
    t2 = torch.randn(6, C)
    mix = (F.softmax(t / 3, 1) + F.softmax(t2 / 3, 1)) / 2
    want2 = 9.0 * (mix * (mix.log() - F.log_softmax(s.detach() / 3, 1))).sum(1).mean()
    got2 = m.distill_loss(s.detach(), [t, t2])
    check(abs(got2.item() - want2.item()) < 1e-5, f'2-teacher KD {got2.item():.6f} != hand {want2.item():.6f}')
    check(abs(m.distill_loss(s.detach(), [t]).item() - m.distill_loss(s.detach(), t).item()) < 1e-7,
          'a one-teacher list differs from a single teacher')
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(2):
            torch.manual_seed(10 + i)
            ti = Classifier.from_config({'model': 'convnext_pico', 'hparams': hp}, num_classes=C, clean_class=clean)
            paths.append(os.path.join(td, f't{i}.ckpt'))
            torch.save({'state_dict': ti.state_dict(), 'hyper_parameters': dict(ti.hparams)}, paths[-1])
        mm = Classifier.from_config({'model': 'convnext_pico', 'hparams': hp, 'distill_teacher': paths,
                                     'distill_weight': 1.0}, num_classes=C, clean_class=clean)
        mm.log = lambda *a, **k: None
        mm.train()
        keys = set(mm.state_dict())
        loss = mm.training_step((x, y), 0)
        check(len(mm._teacher) == 2 and bool(torch.isfinite(loss)) and set(mm.state_dict()) == keys,
              '2 teachers not loaded / loss not finite / teachers leaked into the state dict')
        mm.eval()
        with torch.no_grad():
            both = mm.distill_loss(mm(x), [tt(x) for tt in mm._teacher])
            first = mm.distill_loss(mm(x), mm._teacher[0](x))
        check(abs(both.item() - first.item()) > 1e-6, 'second teacher had no effect on the target')
    print(f'  [13] 2 teachers: target = mean of softened distributions (matches hand {got2.item():.5f}), '
          'one-element list == single teacher, both loaded lazily and kept out of the state dict')


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
    test_hier_knobs()
    test_hier_sev_weight()
    test_hier_hybrid()
    test_hier_detach_branches()
    test_pad_downsample()
    test_rank_loss()
    test_scale_jitter()
    test_sev_per_type()
    test_distill()
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
