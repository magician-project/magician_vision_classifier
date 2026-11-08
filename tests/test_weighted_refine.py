#!/usr/bin/env python3
"""Synthetic checks for analysis/uncertainty_analysis.py's covariance estimator and
weighted PnP refinement -- the RaCo-inspired pieces, verified without the corpus.

  [1] HETEROSCEDASTIC WIN. With half the features observed 8x more precisely than the
      other half, weighted_refine must beat the unweighted solvePnPRefineLM on tvec RMSE
      (>= 15% across 100 trials) -- that is RaCo's covariance weighting transplanted to
      PnP: downweighting noisy observations in the fit.
  [2] HOMOSCEDASTIC PARITY. With equal noise the two refinements solve the same problem
      and must agree (~1%).
  [3] CALIBRATION. With the true covariances, the whitened cost / N sits near 1 (its
      chi-squared expectation) and the first-order pos_std_mm covers the empirical tvec
      error at the 3-sigma rate.
  [4] ANISOTROPY. On a synthetic step edge, feature_covariances is tighter across the
      edge than along it -- the mechanism RaCo's learned covariances capture.
  [5] SCIOPTIMUM. When scipy is present, the Gauss-Newton cost lands within 1e-6 of
      scipy.optimize.least_squares on the same whitened problem.

Run:  python tests/test_weighted_refine.py     (needs the classifier venv for cv2)
"""

import os
import sys
import types

import numpy as np

# Runnable both as `python -m tests.test_weighted_refine` and directly; see test_metrics.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402  (analysis imports pull it in anyway; venv-only)

from analysis.uncertainty_analysis import (feature_covariances,  # noqa: E402
                                           mahalanobis2, pose_std, weighted_refine)

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


# The real sensor's intrinsics; distortion left at zero so the synthetic scene is exact.
K = np.array([[2332.1, 0.0, 561.5], [0.0, 2335.4, 496.6], [0.0, 0.0, 1.0]], np.float64)
DIST = np.zeros(5, np.float64)
RNG = np.random.default_rng(0)


def synth_scene(n=30):
    """n points at 0.32-0.45 m depth, spread over the frame; camera pose is identity."""
    depth = RNG.uniform(0.32, 0.45, n)
    uv = np.column_stack([RNG.uniform(60, 1164, n), RNG.uniform(60, 964, n)])
    xyz = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * depth,
                           (uv[:, 1] - K[1, 2]) / K[1, 1] * depth, depth])
    projected, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, DIST)
    return xyz, projected.reshape(-1, 2)


def noisy_image_pts(projected, stds):
    return projected + RNG.standard_normal(projected.shape) * stds[:, None]


def tvec_error(tvec):
    return float(np.linalg.norm(np.asarray(tvec).ravel()))


def refine_trial(xyz, observed, sigmas):
    """One trial: EPNP init -> unweighted LM and weighted GN from the same start."""
    ok, rvec0, tvec0 = cv2.solvePnP(xyz, observed, K, DIST, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    _, tvec_lm = cv2.solvePnPRefineLM(xyz, observed, K, DIST, rvec0, tvec0)
    _, tvec_gn, J, cost, _ = weighted_refine(xyz, observed, K, DIST, rvec0, tvec0,
                                             sigmas, rho=0.0)
    return tvec_lm, tvec_gn, cost, pose_std(J)


def run_trials(xyz, projected, stds, sigmas, trials=100):
    lm_errors, gn_errors, costs, pos_stds = [], [], [], []
    covered = 0
    for _ in range(trials):
        result = refine_trial(xyz, noisy_image_pts(projected, stds), sigmas)
        if result is None:
            continue
        tvec_lm, tvec_gn, cost, (pos_std_mm, _) = result
        lm_errors.append(tvec_error(tvec_lm))
        gn_errors.append(tvec_error(tvec_gn))
        costs.append(cost)
        pos_stds.append(pos_std_mm)
        covered += tvec_error(tvec_gn) < 3.0 * pos_std_mm
    return (np.array(lm_errors), np.array(gn_errors), np.array(costs),
            np.array(pos_stds), covered)


xyz, projected = synth_scene(30)
sigs = np.array([0.3 ** 2] * 15 + [2.5 ** 2] * 15, np.float64)
stds = np.sqrt(sigs)
hetero = np.stack([np.eye(2) * s for s in sigs])
lm, gn, costs, pos_stds, covered = run_trials(xyz, projected, stds, hetero)
print(f"[1] heteroscedastic: mean tvec RMSE  lm={lm.mean():.4f} m  gn={gn.mean():.4f} m")
check(gn.mean() <= 0.85 * lm.mean(),
      f"[1] weighted refine not 15% better than LM under heteroscedastic noise "
      f"({gn.mean():.4f} vs {lm.mean():.4f})")

sigs_eq = np.full(30, 0.5 ** 2, np.float64)
stds_eq = np.sqrt(sigs_eq)
homo = np.stack([np.eye(2) * s for s in sigs_eq])
lm_h, gn_h, _, _, _ = run_trials(xyz, projected, stds_eq, homo)
print(f"[2] homoscedastic:   mean tvec RMSE  lm={lm_h.mean():.4f} m  gn={gn_h.mean():.4f} m")
check(abs(gn_h.mean() - lm_h.mean()) <= 0.01 * lm_h.mean(),
      f"[2] refinements diverge under homoscedastic noise "
      f"({gn_h.mean():.4f} vs {lm_h.mean():.4f})")

cost_over_n = costs / 30.0                       # E[cost/N] = 1 for the true covariances
print(f"[3] calibration: mean cost/N={cost_over_n.mean():.3f}, "
      f"3-sigma coverage of pos_std_mm={covered}/{len(costs)}")
check(0.7 <= cost_over_n.mean() <= 1.3,
      f"[3] whitened cost/N={cost_over_n.mean():.3f} not near its chi-squared expectation")
check(covered / len(costs) >= 0.95,
      f"[3] pos_std_mm covers only {covered}/{len(costs)} trials at 3 sigma")

edge = np.zeros((512, 512), np.uint8)
edge[:, :256] = 80
edge[:, 256:] = 180
noise = RNG.normal(0, 3.0, edge.shape)             # sensor-like Gaussian noise
edge = np.clip(edge.astype(np.float32) + noise, 0, 255).astype(np.uint8)
edge = cv2.GaussianBlur(edge, (5, 5), 0)     # a real edge has a gradient ramp, not a step
kp = types.SimpleNamespace(pt=(256.0, 256.0), size=32.0)
sigma_edge = feature_covariances(edge, [kp])[0]
print(f"[4] step edge: Sigma=[[{sigma_edge[0, 0]:.6f}, {sigma_edge[0, 1]:.6f}], "
      f"[{sigma_edge[1, 0]:.6f}, {sigma_edge[1, 1]:.6f}]] px^2")
check(sigma_edge[1, 1] > 5.0 * sigma_edge[0, 0],
      f"[4] covariance not tighter across the edge than along it ({sigma_edge})")

try:
    from scipy.optimize import least_squares  # noqa: F401
    from analysis.uncertainty_analysis import whiten_chol

    observed = noisy_image_pts(projected, stds)
    ws = whiten_chol(hetero, 0.0)
    _, rvec0, tvec0 = cv2.solvePnP(xyz, observed, K, DIST, flags=cv2.SOLVEPNP_EPNP)

    def fun(theta):
        rvec = theta[:3].reshape(3, 1)
        tvec = theta[3:].reshape(3, 1)
        proj, _ = cv2.projectPoints(xyz, rvec, tvec, K, DIST)
        r = observed - proj.reshape(-1, 2)
        return np.einsum("nij,nj->ni", ws, r).ravel()

    start = np.concatenate([rvec0.ravel(), tvec0.ravel()]).astype(np.float64)
    ls = least_squares(fun, start, method="trf")
    ls_cost = 0.5 * float(fun(ls.x) @ fun(ls.x))
    _, _, _, gn_cost, _ = weighted_refine(xyz, observed, K, DIST, rvec0, tvec0, hetero,
                                          rho=0.0)
    print(f"[5] scipy cross-check: gn cost={gn_cost:.6f}  least_squares={ls_cost:.6f}")
    check(gn_cost >= ls_cost - 1e-6,
          f"[5] Gauss-Newton cost {gn_cost:.6f} below scipy optimum {ls_cost:.6f}")
except ImportError:
    print("[5] scipy not available, skipping cross-check")

if failures:
    print(f"\n{len(failures)} FAILURE(S):")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("\nall checks passed")
