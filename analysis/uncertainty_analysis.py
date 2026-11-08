#!/usr/bin/env python3
"""Uncertainty-aware pattern localisation: does per-feature covariance help?

RaCo (github.com/cvg/RaCo) trains a network to predict a 2x2 pixel-space covariance per
keypoint, in metric scale, and uses it to weight reprojection residuals in geometric
refinement -- "end-to-end uncertainty propagation from keypoints to pose". This script
transplants the idea to the classical pipeline in extrinsics_from_pattern.py WITHOUT a
learned front-end: the per-feature covariance comes from the local gradient
autocorrelation (Harris) matrix at the SIFT keypoint (the KLT/CRLB localisation
uncertainty), and the pose is refined by minimising the same whitened reprojection cost
RaCo's NLL reduces to:

    cost = 1/2 * sum_i r_i^T (Sigma_i + rho^2 I)^-1 r_i,   r_i = x_i - project(X_i)

with rho^2 I standing in for the map side of the error. Three insertion points, each
swept separately and combined:

  prefilter     drop the least-certain matched features (top quantile of sqrt(trace Sigma))
                BEFORE RANSAC -- RaCo's ranking/top-k idea.
  mahalanobis   re-classify the RANSAC inliers by Mahalanobis distance < k^2 instead of
                the fixed --max-reproj-px circle RANSAC used.
  weighted      refine the pose with damped Gauss-Newton on the whitened cost above
                (the --refine gn option), replacing the unweighted solvePnPRefineLM.

The whitened Jacobian at the optimum gives the first-order pose covariance for free, so
every frame also gets pos_std_mm / rot_std_deg, and a --mc run cross-checks the empirical
spread against it -- the same calibration check RaCo reports (their beta ~= 0.94).

This deliberately does NOT touch extrinsics_from_pattern.py: same policy as
locate_single_frame ("changing a working, measured tool was judged more risk than the
duplication is worth"). If the sweep below shows a win, the winning pieces move into that
module, the CSV and the ROS path; if it does not, this file is the record of why.

MEASURED ON THIS CORPUS -- AND WHY NONE OF THIS SHIPS
-----------------------------------------------------
Map from AltinayUniquePattern750 (367 pairs, 6981 points, stride 3, 4000 features,
--min-baseline 0.05 m), held-out AltinayUniquePattern750_2, stride 10 = 160 frames.
s00 reproduces the production locate() cell-for-cell (A/B validity check passed).

  config                 solved    vs-marker   rot med/90th   pos med/90th   inlMed  wrms
  baseline (locate)      76/160    on 33       2.27 / 4.68     15.2 / 34.5     21     -
  s00 (this script)      76/160    on 33       2.27 / 4.68     15.2 / 34.5     21    1.73
  --refine gn            76/160    on 33       2.62 / 5.17     16.7 / 36.2     21    1.69
  --mahalanobis-k 3      70/160 *  on 32       2.13 / 4.49     15.2 / 34.5     23    1.66
  --prefilter-q 0.25     64/160    on 32       2.04 / 5.03     12.9 / 36.6     22    1.74
  gn + mahal             70/160 *  on 32       2.33 / 4.59     16.3 / 34.3     23    1.60
  gn + pref              64/160    on 32       1.95 / 4.36     12.5 / 35.9     22    1.70
  gn + mahal + pref      60/160 *  on 30       1.95 / 4.51     12.1 / 38.3     24    1.60
  cv2lm + mahal + pref   60/160 *  on 30       1.95 / 4.51     12.1 / 38.3     24    1.60

  * frames lost: 6 at the Mahalanobis re-classification (s02/s04), 12 at the prefilter
    plus 4 more at re-classification (s06/s09).

Uncertainty readouts (median over solved frames, --refine gn runs only):

  config                 pos_std mm   rot_std deg   MC/first-order ratio
  --refine gn            0.75         0.55
  gn + mahal             0.73         0.56
  gn + pref              0.70         0.58
  gn + mahal + pref      0.72         0.58          pos 0.08, rot 0.02  (--mc 200)

Paired on the frames the baseline ALSO solved, every config moves the median by at most
+0.9 mm and 0.07 deg -- the apparent wins are survivorship (pruning drops the frames it
would have gotten wrong), not better poses. Three conclusions, all of them the same one
the docstring of extrinsics_from_pattern.py already reached about --refine:

  1. Weighted vs unweighted refinement is indistinguishable (gn == cv2lm on identical
     kept sets). The pose fit is not limited by feature noise, so RaCo's covariance
     weighting -- which downweights noisy OBSERVATIONS -- has nothing to act on.
  2. The covariance pruners are weaker quality gates than the existing --min-inliers
     sweep, and cost coverage: 60 frames at pos90 38.3 mm against 31 frames at pos90
     18.3 mm for min-inliers 30.
  3. The uncertainty readouts are not calibrated: first-order pos_std median 0.72 mm
     against 12-15 mm actual error (~17x too optimistic), and the Monte-Carlo spread of
     resampling features (ratio 0.08 pos / 0.02 rot) is an order of magnitude smaller
     still. The error budget is MAP BIAS -- 3D points triangulated from noisy marker
     scaffolding -- a systematic error no per-feature covariance can express.

So the gate fails and none of this moves into extrinsics_from_pattern.py or the ROS
path. The covariance machinery is correct (see tests/test_weighted_refine.py: it beats
the unweighted fit 3x where feature noise really is the limiter). It would earn its
keep only behind a better map -- bundle adjustment of the triangulation (the real
consumer of RaCo's "uncertainty propagation from keypoints to pose"), or learned
features -- both noted as the actual limits in extrinsics_from_pattern.py's docstring.

Units follow extrinsics_from_pattern.py: --marker-length 0.025 m by default.

Usage:
  python analysis/uncertainty_analysis.py locate --folder AltinayUniquePattern750_2 \
      --intrinsics intrinsics.json --map map.npz --out s00.csv --compare-markers \
      [--refine gn] [--mahalanobis-k 3] [--prefilter-q 0.25] [--mc 200]

  python analysis/uncertainty_analysis.py --summary baseline.csv s01.csv s02.csv ...
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extrinsics_from_markers import ARUCO_DICT, DATA_ROOT, png_mono, resolve  # noqa: E402
from extrinsics_from_pattern import (CLAHE, detect_and_mask, draw_overlay,  # noqa: E402
                                     load_intrinsics, marker_object_points, match_to_map)

SIGMA_N_FLOOR = 0.05       # grey-levels; guards a textureless whole image
MAX_STD_PX = 10.0          # cap on sqrt(eigenvalues of Sigma) so flat patches stay finite
JAC_ROT_STEP = 1e-5        # rad; central-difference step for the rvec components
JAC_TRANS_STEP = 1e-5      # m; ~0.06 px at 0.4 m depth with f~2332 -- safe for float64


def feature_covariances(gray, keypoints, sigma_n=None, eps_frac=1e-2):
    """Per-keypoint 2x2 localisation covariance in raw (distorted) pixel coords.

    The KLT/CRLB result: a feature's localisation error is the image noise filtered
    through the gradient autocorrelation (Harris) matrix at the keypoint scale,
    Sigma_i = sigma_n^2 * M_i^-1, with M_i the weighted sum of g g^T over a Gaussian
    window. `gray` must be the image the features were DETECTED in -- the CLAHE-equalised
    mono -- because the covariance describes localisation in that image.

    Eigenvalues of M are floored at eps_frac of the largest so flat patches yield a
    large-but-finite covariance; sigma_n is auto-estimated from the flat-region gradient
    energy: the ksize-3 Sobel kernels have squared-sum 12 per axis, so for white noise of
    variance s^2, E[gx^2 + gy^2] = 24 s^2.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad2 = gx * gx + gy * gy
    if sigma_n is None:
        flat = grad2 <= np.quantile(grad2, 0.25)
        sigma_n = float(np.sqrt(max(np.median(grad2[flat]) / 24.0, SIGMA_N_FLOOR ** 2)))

    height, width = gray.shape
    floor = sigma_n ** 2 / (MAX_STD_PX ** 2)
    sigmas = np.empty((len(keypoints), 2, 2), np.float64)
    for i, kp in enumerate(keypoints):
        x, y = kp.pt
        radius = max(kp.size, 1.0)          # SIFT size is the region diameter
        win_sigma = kp.size / 6.0
        x0 = max(min(int(np.floor(x - radius)), width - 1), 0)
        x1 = min(int(np.ceil(x + radius)) + 1, width)
        y0 = max(min(int(np.floor(y - radius)), height - 1), 0)
        y1 = min(int(np.ceil(y + radius)) + 1, height)
        uu, vv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        w = np.exp(-(((uu - x) ** 2 + (vv - y) ** 2) / (2 * win_sigma ** 2)))
        patch_gx, patch_gy = gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
        m00 = float((w * patch_gx ** 2).sum())
        m01 = float((w * patch_gx * patch_gy).sum())
        m11 = float((w * patch_gy ** 2).sum())
        vals, vecs = np.linalg.eigh([[m00, m01], [m01, m11]])
        vals = np.maximum(vals, eps_frac * vals.max())
        vals = np.maximum(vals, floor)      # cap Sigma at (MAX_STD_PX px)^2 per axis
        sigmas[i] = sigma_n ** 2 * (vecs / vals) @ vecs.T
    return sigmas


def whiten_chol(sigmas, rho2):
    """Whitening matrices W_i with W_i (Sigma_i + rho2 I) W_i^T = I (pixel units).

    rho2 stands in for the map side of the reprojection error, which per-point map
    covariances would need to model exactly; a scalar is the stage-1 approximation.
    """
    out = np.empty((len(sigmas), 2, 2), np.float64)
    for i, s in enumerate(sigmas):
        vals, vecs = np.linalg.eigh(s + rho2 * np.eye(2))
        out[i] = (vecs / np.sqrt(vals)) @ vecs.T
    return out


def mahalanobis2(residual, sigmas, rho2):
    """Per-point squared Mahalanobis distances r^T (Sigma + rho2 I)^-1 r."""
    ws = whiten_chol(sigmas, rho2)
    wr = np.einsum("nij,nj->ni", ws, residual)
    return np.einsum("ni,ni->n", wr, wr)


def weighted_refine(object_pts, image_pts, K, dist, rvec0, tvec0, sigmas, rho=1.0,
                    max_iter=20, lambda0=1e-3, tol=1e-9):
    """Damped Gauss-Newton pose refinement on RaCo's whitened reprojection cost.

    Minimises 0.5 * sum_i r_i^T (Sigma_i + rho^2 I)^-1 r_i over rvec(3)+tvec(3) only
    (K and dist stay fixed), via Cholesky whitening and a numerical central-difference
    Jacobian. numpy-only so the same code can move into the ROS-side module unchanged.

    Returns (rvec, tvec, J, cost, converged) with J the WHITENED residual Jacobian at
    the solution: C_pose = inv(J^T J) is the first-order pose covariance, which is what
    pose_std() reads. cost is 0.5 * ||whitened residuals||^2.
    """
    n = len(image_pts)
    ws = whiten_chol(sigmas, rho * rho)

    def project(rvec, tvec):
        projected, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
        return projected.reshape(-1, 2)

    def residual(theta):
        rvec = theta[:3].reshape(3, 1)
        tvec = theta[3:].reshape(3, 1)
        r = image_pts - project(rvec, tvec)               # (n, 2)
        return np.einsum("nij,nj->ni", ws, r).ravel()     # whitened 2n-vector

    def jacobian(theta):
        steps = np.full(6, JAC_ROT_STEP)
        steps[3:] = JAC_TRANS_STEP
        J = np.empty((2 * n, 6), np.float64)
        for j in range(6):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += steps[j]
            minus[j] -= steps[j]
            J[:, j] = (residual(plus) - residual(minus)) / (2 * steps[j])
        return J

    theta = np.concatenate([rvec0.ravel().astype(np.float64),
                            tvec0.ravel().astype(np.float64)])
    rw = residual(theta)
    cost = 0.5 * float(rw @ rw)
    lam = lambda0
    converged = False
    for _ in range(max_iter):
        J = jacobian(theta)
        A = J.T @ J
        g = J.T @ rw
        accepted = False
        for _ in range(50):                              # damping retries
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
            except np.linalg.LinAlgError:
                step = None
            if step is not None:
                trial = theta + step
                rw_trial = residual(trial)
                cost_trial = 0.5 * float(rw_trial @ rw_trial)
                if cost_trial < cost:
                    improvement = cost - cost_trial
                    theta, rw, cost = trial, rw_trial, cost_trial
                    lam = max(lam / 3.0, 1e-12)
                    accepted = True
                    break
            lam *= 10
        if not accepted:
            break
        if improvement <= tol:
            converged = True
            break
    J = jacobian(theta)
    return (theta[:3].reshape(3, 1), theta[3:].reshape(3, 1), J, cost, converged)


def pose_std(J):
    """First-order pose standard deviations from the whitened Jacobian.

    Returns (pos_std_mm, rot_std_deg); the mm assumes tvec is in metres, i.e. the
    default --marker-length 0.025. Small-angle rotation approximation throughout.
    """
    cpose = np.linalg.inv(J.T @ J + 1e-12 * np.eye(6))
    pos_std_mm = 1000.0 * float(np.sqrt(np.trace(cpose[3:, 3:])))
    rot_std_deg = float(np.degrees(np.sqrt(np.trace(cpose[:3, :3]))))
    return pos_std_mm, rot_std_deg


def monte_carlo_std(object_in, image_in, K, dist, sigmas, trials):
    """Empirical pose spread: sample each feature from N(x_i, Sigma_i), re-solve EPNP.

    Returns (pos_std_mm, rot_std_deg) or (None, None) if too many trials fail. This is
    the calibration cross-check for the first-order pose_std(): the ratio of the two is
    RaCo's beta by another name.
    """
    rng = np.random.default_rng(0)
    chol = np.linalg.cholesky(sigmas)
    noise = rng.standard_normal((trials, len(image_in), 2))
    samples = image_in[None, :, :] + np.einsum("kij,tkj->tki", chol, noise)
    centres = []
    rvec_angles = []
    for trial in samples:
        ok, rvec, tvec = cv2.solvePnP(object_in, trial, K, dist,
                                      flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        centres.append((-R.T @ tvec).ravel())
        rvec_angles.append(rvec.ravel())
    if len(centres) < 10:
        return None, None
    centres = np.array(centres)
    rvec_angles = np.array(rvec_angles)
    pos_std_mm = 1000.0 * float(np.sqrt(np.trace(np.cov(centres.T))))
    rot_std_deg = float(np.degrees(np.sqrt(rvec_angles.var(axis=0).sum())))
    return pos_std_mm, rot_std_deg


def locate(args):
    K, dist = load_intrinsics(args.intrinsics)
    obj = marker_object_points(args.marker_length)
    data = np.load(args.map, allow_pickle=False)
    map_points = data["points"].astype(np.float64)
    map_desc = data["descriptors"].astype(np.float32)
    reference_marker = int(data["reference_marker"])

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                       cv2.aruco.DetectorParameters())
    sift = cv2.SIFT_create(nfeatures=args.max_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    folder = resolve(args.folder, args.data_root)
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))[::args.stride]
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    rows = []
    for path in paths:
        mono = png_mono(path)
        if mono is None:
            continue
        marker_pose, keypoints, descriptors = detect_and_mask(
            mono, detector, sift, obj, K, dist, reference_marker, not args.no_clahe)
        # Covariances describe localisation in the image the features were detected in
        # (the CLAHE-equalised mono), never the raw mono. CLAHE is deterministic, so
        # re-applying it here gives the exact image detect_and_mask detected on.
        cov_gray = mono if args.no_clahe else CLAHE.apply(mono)

        separation = args.min_separation if args.min_separation else 0.25 * args.marker_length
        idx = match_to_map(descriptors, map_desc, map_points, matcher, separation)
        row = {"frame": os.path.basename(path), "light": "", "pooled": 0,
               "matches": len(idx), "inliers": 0, "inlier_ratio": "", "reproj_rms_px": "",
               "spread_px": "", "rvec_x": "", "rvec_y": "", "rvec_z": "",
               "cam_x": "", "cam_y": "", "cam_z": "", "rot_err_deg": "", "pos_err": "",
               "prefilter_dropped": "", "mahal_kept": "", "weighted_rms_px": "",
               "cov_med_px": "", "pos_std_mm": "", "rot_std_deg": "",
               "pos_std_mm_mc": "", "rot_std_deg_mc": "", "pos_ratio": "", "rot_ratio": ""}

        pattern_pose, inlier_pts = None, None
        if len(idx) >= 6:
            image_pts = np.array([keypoints[k].pt for k in idx[:, 0]], np.float64)
            sigmas = feature_covariances(cov_gray, [keypoints[k] for k in idx[:, 0]],
                                         sigma_n=args.sigma_noise)
            if args.cov_model == "isotropic":
                # Trace-preserving isotropic version of the same covariances: the sweep
                # isolates the value of anisotropy, not of the estimator's scale.
                scale = 0.5 * (sigmas[:, 0, 0] + sigmas[:, 1, 1])
                sigmas = np.zeros_like(sigmas)
                sigmas[:, 0, 0] = sigmas[:, 1, 1] = scale

            keep_pairs = np.arange(len(idx))
            if args.prefilter_q > 0:
                confidence = np.sqrt(sigmas[:, 0, 0] + sigmas[:, 1, 1])
                drop = confidence > np.quantile(confidence, 1.0 - args.prefilter_q)
                row["prefilter_dropped"] = int(drop.sum())
                keep_pairs = keep_pairs[~drop]

            ok, rvec, tvec, inliers = False, None, None, None
            if len(keep_pairs) >= 6:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    map_points[idx[keep_pairs, 1]], image_pts[keep_pairs], K, dist,
                    reprojectionError=args.max_reproj_px, iterationsCount=500,
                    flags=cv2.SOLVEPNP_EPNP)
            if ok and inliers is not None and len(inliers) >= args.min_inliers:
                keep = inliers.ravel()
                object_in = map_points[idx[keep_pairs[keep], 1]]
                image_in = image_pts[keep_pairs[keep]]
                sigma_in = sigmas[keep_pairs[keep]]

                if args.mahalanobis_k > 0:
                    projected, _ = cv2.projectPoints(object_in, rvec, tvec, K, dist)
                    residual = image_in - projected.reshape(-1, 2)
                    rho2 = args.rho_px * args.rho_px
                    near = mahalanobis2(residual, sigma_in, rho2) < args.mahalanobis_k ** 2
                    row["mahal_kept"] = int(near.sum())
                    object_in, image_in, sigma_in = (object_in[near], image_in[near],
                                                     sigma_in[near])

                if len(object_in) >= args.min_inliers:
                    if args.refine == "cv2lm":
                        rvec, tvec = cv2.solvePnPRefineLM(object_in, image_in, K, dist,
                                                          rvec, tvec)
                    elif args.refine == "gn":
                        rvec, tvec, J, cost, converged = weighted_refine(
                            object_in, image_in, K, dist, rvec, tvec, sigma_in,
                            rho=args.rho_px)
                        row["pos_std_mm"], row["rot_std_deg"] = pose_std(J)
                        if args.mc > 0:
                            pos_mc, rot_mc = monte_carlo_std(object_in, image_in, K, dist,
                                                             sigma_in, args.mc)
                            if pos_mc is not None:
                                row.update({"pos_std_mm_mc": pos_mc, "rot_std_deg_mc": rot_mc,
                                            "pos_ratio": pos_mc / row["pos_std_mm"],
                                            "rot_ratio": rot_mc / row["rot_std_deg"]})

                    R, _ = cv2.Rodrigues(rvec)
                    centre = (-R.T @ tvec).ravel()
                    pattern_pose = (rvec, tvec)
                    inlier_pts = image_in

                    projected, _ = cv2.projectPoints(object_in, rvec, tvec, K, dist)
                    residual = image_in - projected.reshape(-1, 2)
                    reproj = np.linalg.norm(residual, axis=1)
                    spread = float(np.sqrt(image_in[:, 0].var() + image_in[:, 1].var()))
                    rho2 = args.rho_px * args.rho_px
                    row.update({"inliers": len(object_in),
                                "inlier_ratio": len(object_in) / len(idx),
                                "reproj_rms_px": float(np.sqrt((reproj ** 2).mean())),
                                "spread_px": spread,
                                "rvec_x": rvec[0, 0], "rvec_y": rvec[1, 0],
                                "rvec_z": rvec[2, 0],
                                "cam_x": centre[0], "cam_y": centre[1],
                                "cam_z": centre[2],
                                "weighted_rms_px": float(np.sqrt(
                                    mahalanobis2(residual, sigma_in, rho2).mean())),
                                "cov_med_px": float(np.median(np.sqrt(
                                    sigma_in[:, 0, 0] + sigma_in[:, 1, 1])))})
                    if args.compare_markers and marker_pose is not None:
                        R_ref, t_ref = marker_pose
                        delta = R_ref.T @ R
                        angle = np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2,
                                                             -1, 1)))
                        row["rot_err_deg"] = angle
                        row["pos_err"] = float(np.linalg.norm(
                            centre - (-R_ref.T @ t_ref).ravel()))

        if args.debug_dir:
            draw_overlay(os.path.join(args.debug_dir, os.path.basename(path)), mono,
                         K, dist, pattern_pose, marker_pose, inlier_pts, row,
                         args.axis_length or args.marker_length)
        rows.append(row)

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    solved = [r for r in rows if r["inliers"]]
    print(f"{len(solved)}/{len(rows)} frames localised")
    if solved:
        inl = np.array([r["inliers"] for r in solved])
        print(f"inliers: median {np.median(inl):.0f}, min {inl.min()}, max {inl.max()}")
    compared = [r for r in solved if r["rot_err_deg"] != ""]
    if compared:
        rot = np.array([r["rot_err_deg"] for r in compared])
        pos = np.array([r["pos_err"] for r in compared])
        unit = "m" if args.marker_length != 1.0 else "marker sides"
        print(f"vs ArUco on {len(compared)} frames: rotation median {np.median(rot):.2f} deg "
              f"(90th {np.percentile(rot, 90):.2f}), position median {np.median(pos):.3f} "
              f"{unit} (90th {np.percentile(pos, 90):.3f})")
    with_std = [r for r in solved if r["pos_std_mm"] != ""]
    if with_std:
        print(f"pose std on {len(with_std)} frames: position median "
              f"{np.median([r['pos_std_mm'] for r in with_std]):.2f} mm, rotation median "
              f"{np.median([r['rot_std_deg'] for r in with_std]):.2f} deg")
    with_mc = [r for r in with_std if r["pos_ratio"] != ""]
    if with_mc:
        print(f"MC/first-order calibration on {len(with_mc)} frames: position median ratio "
              f"{np.median([r['pos_ratio'] for r in with_mc]):.2f}, rotation "
              f"{np.median([r['rot_ratio'] for r in with_mc]):.2f}")
    if args.mahalanobis_k > 0:
        lost = [r for r in rows if r["mahal_kept"] != "" and not r["inliers"]]
        print(f"{len(lost)} frames lost at the Mahalanobis re-classification")
    print(f"wrote {args.out}")


def load_csv(path):
    with open(path) as handle:
        data = list(csv.DictReader(handle))
    solved = [r for r in data if r["inliers"] not in ("", "0")]
    compared = [r for r in solved if r["rot_err_deg"] != ""]
    return data, solved, compared


def summary(args):
    configs = [(os.path.basename(path),) + load_csv(path) for path in args.csvs]
    print(f"{'config':32s} {'solved':>6s} {'cmp':>3s} {'rotMed':>6s} {'rot90':>6s} "
          f"{'posMed':>6s} {'pos90':>6s} {'inlMed':>6s} {'rmsMed':>6s} {'wrmsMed':>6s}")
    for name, data, solved, compared in configs:
        inl = np.array([r["inliers"] for r in solved], int)
        rms = np.array([r["reproj_rms_px"] for r in solved], float)
        wrms = np.array([r.get("weighted_rms_px") for r in solved
                         if r.get("weighted_rms_px")], float)
        rot = np.array([r["rot_err_deg"] for r in compared], float)
        pos = 1000.0 * np.array([r["pos_err"] for r in compared], float)
        print(f"{name:32s} {len(solved):6d}/{len(data):<4d} {len(compared):3d} "
              f"{np.median(rot):6.2f} {np.percentile(rot, 90):6.2f} "
              f"{np.median(pos):6.1f} {np.percentile(pos, 90):6.1f} "
              f"{np.median(inl):6.0f} {np.median(rms):6.2f} "
              f"{np.median(wrms) if len(wrms) else float('nan'):6.2f}")
    # Paired rows on the frames every config solved: the honest comparison when the
    # solved sets differ between configs.
    base_name, base_data, base_solved, base_compared = configs[0]
    base_by_frame = {r["frame"]: r for r in base_compared}
    print(f"\npaired vs {base_name} (frames with a marker solved by BOTH):")
    print(f"{'config':32s} {'frames':>6s} {'rotMed':>6s} {'posMed':>6s}")
    for name, data, solved, compared in configs[1:]:
        by_frame = {r["frame"]: r for r in compared}
        common = sorted(base_by_frame.keys() & by_frame.keys())
        if not common:
            continue
        rot_delta = np.array([float(by_frame[f]["rot_err_deg"])
                              - float(base_by_frame[f]["rot_err_deg"]) for f in common])
        pos_delta = np.array([1000.0 * (float(by_frame[f]["pos_err"])
                                        - float(base_by_frame[f]["pos_err"]))
                              for f in common])
        print(f"{name:32s} {len(common):6d} {np.median(rot_delta):+6.2f} "
              f"{np.median(pos_delta):+6.1f} mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DATA_ROOT)
    sub = parser.add_subparsers(dest="mode", required=True)

    l = sub.add_parser("locate")
    l.add_argument("--folder", required=True)
    l.add_argument("--intrinsics", required=True)
    l.add_argument("--map", required=True)
    l.add_argument("--out", default="uncertainty_poses.csv")
    l.add_argument("--marker-length", type=float, default=0.025)
    l.add_argument("--stride", type=int, default=5)
    l.add_argument("--max-features", type=int, default=2000)
    l.add_argument("--max-reproj-px", type=float, default=3.0)
    l.add_argument("--min-inliers", type=int, default=12)
    l.add_argument("--min-separation", type=float)
    l.add_argument("--compare-markers", action="store_true")
    l.add_argument("--no-clahe", action="store_true")
    l.add_argument("--debug-dir")
    l.add_argument("--axis-length", type=float)
    l.add_argument("--cov-model", choices=("gradient", "isotropic"), default="gradient",
                   help="isotropic = trace-preserving isotropic version of the gradient "
                        "covariances (isolates the value of anisotropy)")
    l.add_argument("--sigma-noise", type=float,
                   help="override the flat-region noise estimate (grey levels)")
    l.add_argument("--prefilter-q", type=float, default=0.0,
                   help="drop this top quantile of least-certain matches BEFORE RANSAC")
    l.add_argument("--mahalanobis-k", type=float, default=0.0,
                   help="re-classify RANSAC inliers by Mahalanobis distance < k^2 "
                        "(0 disables; 3 is the Gaussian tail)")
    l.add_argument("--rho-px", type=float, default=1.0,
                   help="scalar stand-in for map-side uncertainty, pixels")
    l.add_argument("--refine", choices=("none", "cv2lm", "gn"), default="none",
                   help="cv2lm = the incumbent solvePnPRefineLM, gn = covariance-weighted "
                        "Gauss-Newton")
    l.add_argument("--mc", type=int, default=0,
                   help="Monte-Carlo trials per frame to cross-check the first-order pose "
                        "std (only with --refine gn)")
    l.set_defaults(func=locate)

    s = sub.add_parser("summary")
    s.add_argument("csvs", nargs="+", help="CSVs to compare; the first is the baseline")
    s.set_defaults(func=summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
