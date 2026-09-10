#!/usr/bin/env python3
"""Study physical/optical correlates of failure tiles (FP/FN) on the coverage set.

Question: when a model misses a defect (FN) or false-alarms on clean (FP), is that
systematically tied to how the tile was CAPTURED -- degree of polarization, sensor
distance/tilt, or which physical light source was on -- rather than being architecture
noise? If so, that is an operating-condition lever (which light to run), not a model lever.

For each coverage tile this computes, alongside the model's own score:
  - DoLP / AoLP, from the tile's own raw 4-channel data via mvc.core.polarization (the
    SAME calculate_stokes/calculate_DoLP/calculate_AoLP the model itself uses internally --
    not a separate approximation).
  - AoLP LOCAL COHERENCE, a polarization-consistency term inspired by Poppy
    (arXiv:2603.27891, "Polarization-based Plug-and-Play Guidance for Enhancing Monocular
    Normal Estimation"): Poppy checks polarization PREDICTED from an estimated surface
    normal, via a differentiable Fresnel render, against the OBSERVED polarization -- a
    real surface produces polarization consistent with its (locally smooth) normal field,
    so a mismatch flags a geometric anomaly. This repo has no normal estimator, so
    `aolp_consistency()` below is a cheaper proxy for the SAME underlying physical
    assumption, computed from the observed field's own internal agreement rather than
    against an independent physics prediction: the mean resultant length of 2*AoLP across
    the tile's pixels (AoLP has period pi, not 2pi -- see calculate_AoLP -- so circular
    statistics need the angle doubled first). 1.0 = every pixel agrees on orientation (a
    locally planar patch); 0.0 = orientations are locally random (consistent with a
    geometric disruption -- a dent edge, weld splatter, deformation). Stated as an
    approximation, not literal Poppy, since there is no independently-predicted normal to
    check against here.
  - DistanceAverage and Distance1-3 (three separate sensor readings) from the frame's own
    capture metadata, plus their spread (max-min across the three) as a proxy for surface
    TILT/angle-of-attack -- there is no literal angle field in the metadata, but three
    distance sensors at different positions disagreeing is what a tilted surface looks like
    geometrically, so this is a derived proxy, not a measured angle. Stated as such in the
    output, not silently presented as "angle".
  - Light1-6 (which of 6 physical light sources were on), lightNumber, lightDirection,
    lightConfidence -- read from the SAME per-tile metadata JSON the frame's Distance
    fields come from (mvc.core.dataset_converter.HDF5Dataset stores one JSON blob per tile;
    RAM-preloading strips it from the DataLoader, so this reads the raw H5 metadata array
    directly, mirroring mvc.core.datasets._dataset_source_frames' unwrap-and-index pattern).

Tiles are classified TP/TN/FP/FN at the model's own FA5-matched threshold (the established
convention every other coverage report in this repo uses), then FN is compared against TP
(both are real defects; what differs about the ones that got missed?) and FP against TN
(both are real clean; what differs about the ones that got flagged?).

OUTPUT: per-light-source-configuration detection and false-alarm rates, so a configuration
that empirically detects more of the same real defects (or false-alarms less on the same
real clean tiles) can be read directly off the table -- this is the "which light should be
on" answer, to the resolution the existing capture metadata supports.

Usage:
    python -m analysis.eval.eval_failure_conditions <config.json> [checkpoint.ckpt]
"""

import glob
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvc.core.config import load_hyperparameters
from mvc.core.datasets import clean_class_index
from mvc.core.lit_classifier import Classifier
from mvc.core.metrics import fa_threshold
from mvc.core.polarization import calculate_AoLP, calculate_DoLP, calculate_stokes
from analysis.eval.eval_coverage import build_coverage_loader

FRAME_FIELDS = ('Distance1', 'Distance2', 'Distance3', 'DistanceAverage',
                'Light1', 'Light2', 'Light3', 'Light4', 'Light5', 'Light6',
                'lightDirection', 'lightNumber', 'lightConfidence')


def metadata_for_positions(dataset, positions):
    """Decoded per-tile metadata dict for each position (0..len(dataset)-1), reading the
    raw H5 'metadata' array directly -- mirrors _dataset_source_frames' unwrap (a
    RAM-preloaded dataset forwards only images/labels, not metadata) and its row-subset
    handling (an inner HDF5Dataset.indices, if the class scheme narrowed rows), so this
    stays correct under exactly the same wrapping every other frame-identity lookup in
    this repo already has to account for."""
    ds = dataset
    if not hasattr(ds, 'file'):
        inner = getattr(ds, '_dataset', None)
        if inner is not None:
            ds = inner
    if not hasattr(ds, 'file'):
        raise ValueError('no raw H5 file behind this dataset -- cannot read metadata')
    prior = ds.indices if getattr(ds, 'indices', None) is not None else None
    raw = ds.file['metadata']
    out = []
    for i, p in enumerate(positions):
        row = int(prior[p]) if prior is not None else int(p)
        m = raw[row]
        m = m.decode() if isinstance(m, bytes) else m
        try:
            out.append(json.loads(m))
        except Exception:
            out.append({})
        if i % 25000 == 0:
            print(f'  metadata {i:,}/{len(positions):,}', end='\r')
    print()
    return out


def to_float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def aolp_consistency(stokes):
    """Poppy-inspired polarization-consistency term -- see the module docstring for the
    full reasoning and the honest caveat (proxy, not literal Poppy: no normal estimator
    here, so this checks the observed AoLP field's internal agreement, not an independent
    physics prediction).

    Per tile: mean resultant length of 2*AoLP over the tile's pixels, in [0, 1].
    2*AoLP = atan2(S2, S1) is the raw, period-2*pi quantity (calculate_AoLP's extra 0.5
    factor is what folds it down to AoLP's period-pi range) -- circular statistics need
    that period-2*pi form, not AoLP itself, or a tile split exactly at the pi/2 wrap would
    read as maximally incoherent when every pixel actually agrees.

    Args:
        stokes: Stokes tensor (batch, 4, H, W) from calculate_stokes.

    Returns:
        (batch,) tensor, 1.0 = every pixel agrees on orientation, 0.0 = locally random.
    """
    S1 = stokes[:, 1, :, :]
    S2 = stokes[:, 2, :, :]
    theta2 = torch.atan2(S2, S1)
    c = torch.cos(theta2).mean(dim=(1, 2))
    s = torch.sin(theta2).mean(dim=(1, 2))
    return torch.sqrt(c ** 2 + s ** 2)


def light_signature(meta):
    """A tuple like ('L1','L3') for the lights ON in this tile's frame -- the ROS-level
    knob a light-selection recommendation would actually toggle."""
    on = tuple(f'L{i}' for i in range(1, 7) if str(meta.get(f'Light{i}')) == '1')
    return on if on else ('none',)


def pick_best_checkpoint(cfg):
    cks = sorted(glob.glob(os.path.join(cfg['checkpoint_dir'], '*.ckpt')))
    if not cks:
        sys.exit(f'no checkpoints in {cfg["checkpoint_dir"]}')
    mon = cfg.get('checkpoint_monitor', 'val_detect_auroc')
    mode = 'max' if cfg.get('checkpoint_mode', 'max') == 'max' else 'min'
    scored = [(float(m.group(1)), c) for c in cks
              if (m := re.search(rf'{re.escape(mon)}=([0-9]+\.[0-9]+)', c))]
    return (max if mode == 'max' else min)(scored)[1] if scored else cks[-1]


def summarize(name, values, mask_a, label_a, mask_b, label_b):
    a, b = values[mask_a], values[mask_b]
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        print(f'  {name}: insufficient data ({label_a} n={len(a)}, {label_b} n={len(b)})')
        return
    print(f'  {name}:')
    print(f'    {label_a:4s} (n={len(a):6,d})  mean={a.mean():7.4f}  median={np.median(a):7.4f}  std={a.std():7.4f}')
    print(f'    {label_b:4s} (n={len(b):6,d})  mean={b.mean():7.4f}  median={np.median(b):7.4f}  std={b.std():7.4f}')
    print(f'    Δ mean ({label_a} - {label_b}) = {a.mean() - b.mean():+.4f}')


def main():
    cfg_path = sys.argv[1]
    cfg = load_hyperparameters(cfg_path)
    ds, subset, payload = build_coverage_loader(cfg)
    class_names = list(ds.classes)
    clean_id = clean_class_index(class_names)

    ckpt = sys.argv[2] if len(sys.argv) > 2 else pick_best_checkpoint(cfg)
    print(f'[failure-conditions] {cfg["name"]}_{cfg["model"]}, checkpoint '
          f'{os.path.basename(ckpt)}')

    model = Classifier.from_config(cfg, num_classes=len(class_names), clean_class=clean_id)
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()

    loader = DataLoader(subset, batch_size=cfg['hparams']['batch_size'], shuffle=False,
                         num_workers=8)
    mass, truth, dolp_mean, aolp_mean, aolp_consist = [], [], [], [], []
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            xf = x.to(dev).float() * (1.0 / 255.0) if x.dtype == torch.uint8 else x.to(dev).float()
            pol = xf[:, 0:4, :, :]
            stokes = calculate_stokes(pol)
            dolp_mean.append(calculate_DoLP(stokes).mean(dim=(1, 2)).cpu().numpy())
            aolp_mean.append(calculate_AoLP(stokes).mean(dim=(1, 2)).cpu().numpy())
            aolp_consist.append(aolp_consistency(stokes).cpu().numpy())
            logits = model(x.to(dev))
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            mass.append(1.0 - prob[:, clean_id])
            truth.append(y.numpy())
            if bi % 50 == 0:
                print(f'  scoring batch {bi}/{len(loader)}', end='\r')
    print()
    mass = np.concatenate(mass)
    truth = np.concatenate(truth)
    dolp_mean = np.concatenate(dolp_mean)
    aolp_mean = np.concatenate(aolp_mean)
    aolp_consist = np.concatenate(aolp_consist)

    print('[failure-conditions] reading capture metadata for the same tiles...')
    meta = metadata_for_positions(ds, subset.indices)
    distance_avg = np.array([to_float(m.get('DistanceAverage')) for m in meta])
    d1 = np.array([to_float(m.get('Distance1')) for m in meta])
    d2 = np.array([to_float(m.get('Distance2')) for m in meta])
    d3 = np.array([to_float(m.get('Distance3')) for m in meta])
    with np.errstate(invalid='ignore'):
        d_spread = np.nanmax(np.stack([d1, d2, d3]), axis=0) - np.nanmin(np.stack([d1, d2, d3]), axis=0)
    light_conf = np.array([to_float(m.get('lightConfidence')) for m in meta])
    signatures = [light_signature(m) for m in meta]

    is_clean = truth == clean_id
    thr = fa_threshold(mass, is_clean, 0.05)
    pred_defect = mass >= thr
    tp = pred_defect & ~is_clean
    fn = ~pred_defect & ~is_clean
    fp = pred_defect & is_clean
    tn = ~pred_defect & is_clean
    print(f'\n[failure-conditions] FA5-matched threshold={thr:.3f} -- '
          f'TP={tp.sum():,} FN={fn.sum():,} FP={fp.sum():,} TN={tn.sum():,}\n')

    print('=== Degree/Angle of Linear Polarization ===')
    summarize('DoLP (mean over tile)', dolp_mean, fn, 'FN', tp, 'TP')
    summarize('DoLP (mean over tile)', dolp_mean, fp, 'FP', tn, 'TN')
    summarize('AoLP (mean over tile, NOT circular -- see caveat)', aolp_mean, fn, 'FN', tp, 'TP')
    summarize('AoLP (mean over tile, NOT circular -- see caveat)', aolp_mean, fp, 'FP', tn, 'TN')
    print('  caveat: AoLP wraps at +-pi/2 (it is an axis, not a vector); a plain arithmetic')
    print('  mean is misleading near the wrap boundary. Treat the AoLP rows as a rough')
    print('  signal, not a rigorous circular-statistics result.')

    print('\n=== Polarization consistency (Poppy-inspired, arXiv:2603.27891) ===')
    summarize('AoLP local coherence (1=coherent, 0=random; see caveat)',
              aolp_consist, fn, 'FN', tp, 'TP')
    summarize('AoLP local coherence (1=coherent, 0=random; see caveat)',
              aolp_consist, fp, 'FP', tn, 'TN')
    print("  caveat: this is a PROXY, not literal Poppy. Poppy checks OBSERVED polarization")
    print('  against polarization PREDICTED from an independently estimated surface normal')
    print('  via a differentiable Fresnel render -- a real physics-based consistency check.')
    print('  This repo has no normal estimator, so this instead measures the tile\'s own')
    print('  AoLP field agreeing with ITSELF (mean resultant length of 2*AoLP, correctly')
    print('  circular -- see aolp_consistency()\'s docstring), i.e. local orientation')
    print('  coherence, not a verified-against-geometry consistency. Low coherence is')
    print('  consistent with a geometric disruption (dent edge, weld splatter, deformation)')
    print('  but is not proof of one -- e.g. a flat but low-signal (low DoLP) tile can also')
    print('  read as locally incoherent from noise alone; read alongside the DoLP row above.')

    print('\n=== Sensor distance / tilt proxy ===')
    summarize('DistanceAverage', distance_avg, fn, 'FN', tp, 'TP')
    summarize('DistanceAverage', distance_avg, fp, 'FP', tn, 'TN')
    summarize('Distance1/2/3 spread (tilt proxy, NOT a measured angle -- see caveat)',
              d_spread, fn, 'FN', tp, 'TP')
    summarize('Distance1/2/3 spread (tilt proxy, NOT a measured angle -- see caveat)',
              d_spread, fp, 'FP', tn, 'TN')
    print('  caveat: no angle-of-attack field exists in the capture metadata. This spread')
    print('  is a geometric proxy (three sensors at different positions read more')
    print('  differently when the surface is tilted relative to the camera than when it is')
    print('  flat and perpendicular) -- indicative, not a calibrated angle measurement.')

    print('\n=== Light configuration vs. detection / false-alarm rate ===')
    print(f'{"light signature":24s} {"defect n":>9s} {"detect%":>8s} {"clean n":>9s} '
          f'{"FA%":>6s} {"lightConf":>10s}')
    sig_set = sorted(set(signatures), key=lambda s: -sum(1 for x in signatures if x == s))
    for sig in sig_set:
        m = np.array([s == sig for s in signatures])
        defect_n = int((m & ~is_clean).sum())
        clean_n = int((m & is_clean).sum())
        det_pct = 100.0 * (m & tp).sum() / defect_n if defect_n else float('nan')
        fa_pct = 100.0 * (m & fp).sum() / clean_n if clean_n else float('nan')
        conf = np.nanmean(light_conf[m]) if m.any() else float('nan')
        print(f'{"+".join(sig):24s} {defect_n:9,d} {det_pct:8.2f} {clean_n:9,d} '
              f'{fa_pct:6.2f} {conf:10.3f}')
    print('\ndetect% = TP / (TP+FN) among real defect tiles under this light signature -- the')
    print('rate that matters for "does this light configuration let the model actually see')
    print('the defect". FA% = FP / (FP+TN) among real clean tiles -- lower is better.')
    print('Read sample sizes before trusting a row: signatures with few frames in this')
    print('coverage set are noisy, not necessarily bad lighting.')


if __name__ == '__main__':
    main()
