#!/usr/bin/env python3
"""Camera extrinsics from the car's own pattern, with or without ArUco markers present.

`extrinsics_from_markers.py` needs a marker in view. Nobody will glue markers to customer
vehicles, so this tool learns the mirror-mount region itself and localises against that.
The markers are used only as scaffolding, during `build`, and are masked out of every
feature so the map contains car pattern and nothing else -- otherwise "marker-free"
localisation would just be reading the marker again.

  build    a marker-bearing recording -> map.npz. Frames where the reference marker is
           visible get a pose from it; SIFT features are extracted with every detected
           marker quad masked out, matched between frame pairs with enough baseline, and
           triangulated into 3D points in the REFERENCE MARKER's frame. A point is kept
           only if it lies in front of both cameras, reprojects to under --max-reproj-px
           in both, and was seen with at least --min-parallax-deg between the two rays.
           That last test is what keeps the map honest: a pair with a short baseline can
           place a point at near-infinite depth and still reproject perfectly, so without
           it the map fills with far-field garbage.

  locate   any folder -> per-frame pose in that same frame, by matching features to the
           map and running solvePnPRansac. Markers, if present, are masked out here too.
           With --compare-markers the ArUco pose is computed from the unmasked image and
           reported alongside, which is the honest way to measure this: same frame, same
           intrinsics, one pose from the marker and one from the pattern alone.

The reference marker (default id 10, the most-seen of 10/11/12) defines the coordinate
frame, so every pose is relative to that point on the car. Poses from frames seeing only
markers 11 or 12 are in DIFFERENT frames and are not mixed in; unifying them needs the
marker-to-marker transforms, which nothing in the corpus records.

MEASURED ON THIS CORPUS
-----------------------
All features come from the unpolarized mono image (png_mono: the mean of the 4 DoFP
channels), never a luma grayscale, so matching never depends on which polarization
channels the BGR weights happen to favour.

Map from AltinayUniquePattern750 (stride 3, 4000 features, --min-baseline 2.0): 367 pairs,
6977 points, extent 4.7 x 4.1 x 1.2 marker sides.

  Altinayuniquepatternmirror  18/20 frames localised, median 46 inliers  -- bare panel
  AltinayUniquePattern750_2   77/160 frames localised                   -- held-out
  vs ArUco on the 33 of those that also show a marker:
      rotation  median 1.97 deg   (90th 4.68)
      position  median 13.8 mm    (90th 34.3 mm)

On AltinayUniquePattern750 itself, the recording the map was built from, the same
comparison gives 0.93 deg and 5.6 mm. Treat the held-out figures as the real ones: the map
fits its own session considerably better than it generalises to the next.

SCALE. The markers are 25 mm and measure a median 144.8 px, so this corpus was recorded at
0.173 mm/px, an optical distance of ~403 mm. That is NOT a different camera or a different
focus -- the frames are as sharp as the defect recordings (median Tenengrad 332 and 189
here, against 180 for AltinayWeldings and 268 for AltinayCarDown650_2). It is the same
tool, and the stand-off datum is simply not the lens: D5.3 quotes a 153 x 129 mm footprint
at 60 mm hover, which puts the lens centre ~232 mm behind the "bottom of the sensor" the
hover is measured from. So these scans sit at roughly 171 mm hover against the nominal
45-60 mm band, and the scale difference that actually matters is 0.173 vs 0.125 mm/px --
a 38% coarser view, not the order of magnitude the raw optical distances suggest. SIFT
absorbs 1.4x scale without complaint, so the corpus is usable as it stands; a map built
here is simply 38% coarser than one built at the inspection hover.

BASELINE IS THE LEVER, AND SHORT PAIRS ARE POISON
--------------------------------------------------
--min-baseline swept over 0.15 / 0.5 / 1.0 / 2.0 / 4.0 marker sides, paired on the 20
held-out frames every setting solved:

     50 mm (0.15 sides)   2.38 deg   15.5 mm   (25847 map points)   <- old default
     12 mm (0.5)          2.46       16.5      (25579)
     25 mm (1.0)          1.89       13.5      (19557)
     50 mm (2.0)          1.80       12.1       (6977)   <- default
    100 mm (4.0)          2.99       22.7       (1168)

2.0 wins on accuracy AND coverage (77 of 160 held-out frames against 69 at 0.15) with a
map a quarter the size, so it matches faster too. Bigger maps were not better maps: the
extra points at a short baseline are badly conditioned triangulations that survive the
reprojection gate and then drag the pose. Past 2.0 there are too few pairs left and it
collapses. Nothing here is close to the parallax gate's job -- that removes points whose
rays are near-parallel, while this removes whole PAIRS before matching.

WHAT THE ILLUMINATOR LABEL IS AND IS NOT WORTH
----------------------------------------------
Each frame is captured with one of six illuminators active and records which (frame_light).
Two ways to use that were tried, and only one is on by default.

Pairing within a light when building (default, --any-light disables): 1134 pairs yield
39741 points against 14715 from the same pair budget across lights, because two frames
under one light actually match. Localisation is a wash -- 73 vs 74 of 160 frames, rotation
3.47 vs 3.31 deg paired on the 37 both solved -- but inlier support is better (median 29
vs 23), so the denser map is kept. --any-light gives a map a third the size, and a
correspondingly faster match, for the same accuracy.

Restricting the RANSAC pool to the query frame's own light (--pool-by-light, OFF): this
LOSES, on the same map. 55 of 160 frames against 73, and 12 of 20 against 18 on the bare
panel; paired on the 27 both solved it is worse on every axis -- rotation 3.74 vs 3.47 deg,
position 0.998 vs 0.962 marker sides, inliers 22 vs 29. The premise that cross-light
descriptors compete as wrong answers does not hold: RANSAC rejects them geometrically
already, and cutting the map to one sixth discards real coverage of the region. Kept as a
flag because it is the obvious thing to try and the measurement is worth not repeating.

DISCARDING OUTLIERS: GATE ON INLIER COUNT, DO NOT BOTHER REFINING
------------------------------------------------------------------
Of the four confidence signals recorded per frame, only two predict the error at all
(Spearman rho against position error, 33 held-out frames): inlier count -0.50 (p 0.003)
and reprojection RMS +0.40 (p 0.022). inlier_ratio is worthless (+0.01, p 0.96) and
spread_px is weak (-0.34, p 0.054). So --min-inliers is the gate worth turning:

    min-inliers   frames kept   pos median   pos 90th   rot median
         12          76/160       15.1 mm     34.3 mm     1.94 deg    <- default
         16          57/160       13.8         32.9       1.93
         20          41/160       11.9         27.6       1.78
         25          34/160       11.9         18.3       1.78
         30          31/160       11.2         18.3       1.74
         40          19/160        9.3         17.8       1.41

The knee is 25-30: the tail halves (34.3 -> 18.3 mm at the 90th percentile) for a cut to
roughly 40% of the frames. That trade is usually worth taking, because aligning a car does
not need a pose from every frame of a scan, it needs one pose that can be trusted. The
default is left at 12 so nothing is silently discarded; raise it deliberately.

What is NOT available here is the bigger win. Every frame gives an independent estimate of
the same car-to-robot transform, so a robust aggregate over a scan should beat any single
frame comfortably -- but combining them needs the robot TCP pose per frame to relate the
moving camera to a fixed frame, and no recording in this corpus logs one. That is the same
gap that stops the grabber's calibrateHandEye half from running.

WHAT LIMITS THIS, MEASURED RATHER THAN GUESSED
-----------------------------------------------
The surface is dark, specular and low-texture, so plain SIFT finds a median of 282
keypoints per frame; CLAHE lifts that to 652 and is worth roughly 3x the localisation rate.
64% of triangulated points are lost at the reprojection gate, which is the marker poses
themselves being noisy -- the map is only ever as good as the scaffolding that built it.
And marker 10 is detected in 90 of the Top-lit frames but only 13 Bottom-lit ones, so even
the scaffolding is unevenly distributed over the lights. All three argue for learned
features (SuperPoint/LoFTR) and a multi-marker board, not for tuning this pipeline.

Units follow extrinsics_from_markers.py: the markers measure 25 mm on a side, so
--marker-length defaults to 0.025 and the map, the baselines and every translation are in
metres. Pass --marker-length 1.0 to work in marker-side units instead.

Usage:
  python analysis/extrinsics_from_pattern.py build --folder AltinayUniquePattern750 \
      --intrinsics intrinsics.json --out map.npz [--stride 5] [--reference-marker 10]

  python analysis/extrinsics_from_pattern.py locate --folder Altinayuniquepatternmirror \
      --intrinsics intrinsics.json --map map.npz --out pattern_poses.csv \
      [--compare-markers]
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extrinsics_from_markers import ARUCO_DICT, DATA_ROOT, png_mono, resolve  # noqa: E402

RATIO = 0.75          # Lowe ratio for descriptor matching
MASK_DILATE_PX = 12   # grow the marker mask to swallow its white border

# The mirror-mount region is dark, specular and low-texture: plain SIFT finds a median of
# 282 keypoints per frame on it, CLAHE-equalised it finds 652. Applied in both build and
# locate, so map and query descriptors always come from the same preprocessing.
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def load_intrinsics(path):
    with open(path) as handle:
        intr = json.load(handle)
    return (np.array(intr["camera_matrix"], dtype=np.float64),
            np.array(intr["dist_coeffs"], dtype=np.float64))


def frame_light(path, min_confidence=0.0):
    """Active illuminator for a frame, from its sibling annotation json.

    lightDirection is not the controller's command: mga/core/light_decoder.py fuses the
    commanded CSV cycle with the observed per-channel image signature, because the emitted
    light lags the command with variable latency (7% of Altinay frames are stalls where
    the CSV advanced but the image did not). The result is a canonical, wiring-invariant
    physical DIRECTION, which is why this keys on the direction string rather than
    lightNumber -- and it is just as well, since the bare mirror recording carries
    directions with no numbers at all.

    lightConfidence flags the labels to trust least: mid-transition frames sit near the
    decoder's change threshold and score low, and their appearance is a blend of two
    lights, so pooling them under either one is wrong. Frames below min_confidence are
    reported as unlabelled. A MISSING confidence counts as trusted -- the mirror recording
    has none -- so the gate never silently excludes a whole recording.

    Returns None for unlabelled, 'No Light', 'Unknown' and low-confidence frames.
    """
    for candidate in (path[:-4] + ".json", path[:-4] + ".pnm.json"):
        if os.path.isfile(candidate):
            try:
                with open(candidate) as handle:
                    meta = json.load(handle)
            except (ValueError, OSError):
                return None
            light = meta.get("lightDirection")
            if light in (None, "No Light", "Unknown"):
                return None
            confidence = meta.get("lightConfidence")
            if confidence is not None and confidence < min_confidence:
                return None
            return light
    return None


def marker_object_points(marker_length):
    half = marker_length / 2.0
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], dtype=np.float32)


def detect_and_mask(mono, detector, sift, obj, K, dist, reference_marker, equalise=True):
    """SIFT features outside every marker, plus the reference marker's pose if visible.

    Takes the unpolarized mono image (png_mono), not a luma grayscale: matching a specular
    panel across frames lit from different directions must not depend on which
    polarization channels the luma weights happen to favour.

    `detector=None` skips ArUco entirely (no masking, pose always None) -- the marker-free
    LOCATE path (locate_single_frame, used by the live ROS service) has no need for one:
    markers are scaffolding for `build` only, not something a deployed car is expected to
    carry (see the module docstring). `build`/`locate`'s own CLI callers always pass a
    real detector, unchanged from before this was made optional.

    Returns (pose, keypoints, descriptors) where pose is (R, t) or None.
    """
    gray = mono
    if detector is None:
        corners, ids = None, None
    else:
        corners, ids, _ = detector.detectMarkers(gray)

    mask = np.full(gray.shape, 255, np.uint8)
    pose = None
    if ids is not None:
        for corner, marker_id in zip(corners, ids.ravel()):
            quad = corner[0].astype(np.int32)
            cv2.fillConvexPoly(mask, quad, 0)
            if int(marker_id) == reference_marker:
                ok, rvec, tvec = cv2.solvePnP(obj, corner[0].astype(np.float32), K, dist,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    pose = (R, tvec)
        kernel = np.ones((MASK_DILATE_PX, MASK_DILATE_PX), np.uint8)
        mask = cv2.erode(mask, kernel)   # erode the keep-region = grow the masked quads

    if equalise:
        gray = CLAHE.apply(gray)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    return pose, keypoints, descriptors


def match(desc_a, desc_b, matcher):
    """Ratio-tested mutual matches as index pairs."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0, 2), int)
    pairs = []
    for m, n in matcher.knnMatch(desc_a, desc_b, k=2):
        if m.distance < RATIO * n.distance:
            pairs.append((m.queryIdx, m.trainIdx))
    return np.array(pairs, int).reshape(-1, 2)


def match_to_map(desc, map_desc, map_points, matcher, min_separation):
    """Ratio test against the nearest map point that is a DIFFERENT physical point.

    The same surface feature is triangulated by many frame pairs, so the map holds near
    duplicate descriptors of it -- measured at 32% of this map, a median of 0.21 marker
    sides apart. A plain ratio test compares a match against its own duplicate, finds them
    equally close, and throws the match away, which is why naive matching localises almost
    nothing. So the second neighbour is taken to be the nearest candidate lying more than
    --min-separation away in 3D.
    """
    if desc is None or len(desc) < 2:
        return np.empty((0, 2), int)
    pairs = []
    for candidates in matcher.knnMatch(desc, map_desc, k=8):
        if not candidates:
            continue
        best = candidates[0]
        origin = map_points[best.trainIdx]
        rival = next((c for c in candidates[1:]
                      if np.linalg.norm(map_points[c.trainIdx] - origin) > min_separation),
                     None)
        if rival is None or best.distance < RATIO * rival.distance:
            pairs.append((best.queryIdx, best.trainIdx))
    return np.array(pairs, int).reshape(-1, 2)


def locate_single_frame(mono, K, dist, map_points, map_desc, sift, matcher, obj,
                        reference_marker, *, detector=None, min_inliers=12, max_reproj_px=3.0,
                        min_separation=None, marker_length=0.025, refine=False,
                        equalise=True):
    """Localise ONE in-memory mono frame against an already-loaded map -- NO marker needs
    to be visible in `mono` for this to work; that is the entire point of this tool over
    extrinsics_from_markers.py (see the module docstring). `detector` defaults to None
    (ArUco skipped entirely, see detect_and_mask) for exactly that reason -- pass a real
    one only if you also want an incidentally-visible marker masked out of the SIFT
    features, or its pose back for comparison (result['marker_pose']).

    Same detect_and_mask -> match_to_map -> solvePnPRansac steps `locate()` runs per frame
    in its folder loop, factored out here for a caller that has a single live frame instead
    of a folder to batch (mvc/inference/live_torch_ros.py). Deliberately NOT wired into
    `locate()` itself -- that CLI path is validated against real recordings (see the module
    docstring's measured numbers) and additionally supports --pool-by-light/--compare-markers/
    --debug-dir, which this single-frame path does not reproduce; changing a working,
    measured tool to share this code was judged more risk than the duplication is worth.

    Returns a dict (rvec, tvec, inliers, inlier_ratio, reproj_rms_px, marker_pose) or None
    if the frame does not localise (too few matches or RANSAC inliers) -- callers should
    treat None as "no pose available now", not an error.
    """
    marker_pose, keypoints, descriptors = detect_and_mask(
        mono, detector, sift, obj, K, dist, reference_marker, equalise)
    separation = min_separation if min_separation else 0.25 * marker_length
    idx = match_to_map(descriptors, map_desc, map_points, matcher, separation)
    if len(idx) < 6:
        return None
    image_pts = np.array([keypoints[k].pt for k in idx[:, 0]], np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        map_points[idx[:, 1]], image_pts, K, dist,
        reprojectionError=max_reproj_px, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None
    keep = inliers.ravel()
    object_in = map_points[idx[keep, 1]]
    image_in = image_pts[keep]
    if refine and len(keep) >= 4:
        rvec, tvec = cv2.solvePnPRefineLM(object_in, image_in, K, dist, rvec, tvec)
    projected, _ = cv2.projectPoints(object_in, rvec, tvec, K, dist)
    residual = np.linalg.norm(projected.reshape(-1, 2) - image_in, axis=1)
    return {
        'rvec': rvec, 'tvec': tvec, 'inliers': int(len(keep)),
        'inlier_ratio': float(len(keep) / len(idx)),
        'reproj_rms_px': float(np.sqrt((residual ** 2).mean())),
        'marker_pose': marker_pose,
    }


def draw_overlay(out_path, mono, K, dist, pattern_pose, marker_pose, inlier_pts, row,
                 axis_length):
    """Write the frame with the recovered axes drawn on it.

    The pattern axes are the answer being checked, so they are drawn full length; the
    marker axes, when a marker happens to be in view, are drawn shorter and from the same
    origin, so agreement shows up as two overlapping triads and disagreement as a visible
    fork. Both use the OpenCV convention, X red, Y green, Z blue. The dots are the RANSAC
    inliers: they show WHICH surface features carried the pose, which is what you want to
    see when a frame fails -- a pose supported by a tight cluster in one corner is one to
    distrust even when it looks plausible.
    """
    canvas = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    if inlier_pts is not None:
        for point in inlier_pts.astype(int):
            cv2.circle(canvas, tuple(point), 2, (0, 255, 255), -1)

    if marker_pose is not None:
        R_ref, t_ref = marker_pose
        rvec_ref, _ = cv2.Rodrigues(R_ref)
        cv2.drawFrameAxes(canvas, K, dist, rvec_ref, t_ref, axis_length * 0.6, 2)

    if pattern_pose is not None:
        cv2.drawFrameAxes(canvas, K, dist, pattern_pose[0], pattern_pose[1], axis_length, 4)

    lines = [f"{row['frame']}  light={row['light'] or '?'}",
             f"matches {row['matches']}  inliers {row['inliers']}"
             + ("  POSE" if pattern_pose is not None else "  NO POSE")]
    if row["rot_err_deg"] != "":
        lines.append(f"vs marker: {row['rot_err_deg']:.2f} deg, {row['pos_err']:.3f} sides")
    elif marker_pose is not None:
        lines.append("marker visible (thin axes)")
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def build(args):
    K, dist = load_intrinsics(args.intrinsics)
    obj = marker_object_points(args.marker_length)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                       cv2.aruco.DetectorParameters())
    sift = cv2.SIFT_create(nfeatures=args.max_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    folder = resolve(args.folder, args.data_root)
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))[::args.stride]

    views = []
    for path in paths:
        mono = png_mono(path)
        if mono is None:
            continue
        pose, keypoints, descriptors = detect_and_mask(mono, detector, sift, obj, K, dist,
                                                       args.reference_marker,
                                                       not args.no_clahe)
        if pose is None or descriptors is None:
            continue
        R, t = pose
        views.append({
            "path": path,
            "light": frame_light(path, args.min_light_confidence),
            "P": K @ np.hstack([R, t]),
            "centre": (-R.T @ t).ravel(),
            "pts": np.array([kp.pt for kp in keypoints], np.float32),
            "desc": descriptors,
        })
    print(f"{len(views)}/{len(paths)} frames pose-anchored on marker {args.reference_marker}")
    print("views per illuminator: "
          + str({v: sum(1 for w in views if (w["light"] or "") == v)
                 for v in sorted({(w["light"] or "") for w in views})}))
    if len(views) < 2:
        sys.exit("not enough anchored views to triangulate")

    # Pair within an illuminator, not across. The capture cycles the six lights frame by
    # frame, so a view's same-light neighbours are about six frames away and would fall
    # outside any small span over the raw sequence -- grouping first is what makes
    # --pair-span mean "the next few views under this light".
    if args.any_light:
        groups = [views]
    else:
        by_light = {}
        for view in views:
            by_light.setdefault(view["light"], []).append(view)
        groups = list(by_light.values())

    candidates = []
    for group in groups:
        for i, view in enumerate(group):
            for j in range(i + 1, min(i + 1 + args.pair_span, len(group))):
                candidates.append((view, group[j]))

    points, descs, lights = [], [], []
    pairs_used = 0
    for view, other in candidates:
        baseline = np.linalg.norm(view["centre"] - other["centre"])
        if baseline < (args.min_baseline or 2.0 * args.marker_length):
            continue
        idx = match(view["desc"], other["desc"], matcher)
        if len(idx) < 8:
            continue
        pairs_used += 1
        pts_a = view["pts"][idx[:, 0]].T
        pts_b = other["pts"][idx[:, 1]].T
        homogeneous = cv2.triangulatePoints(view["P"], other["P"], pts_a, pts_b)
        xyz = (homogeneous[:3] / homogeneous[3]).T

        ray_a = xyz - view["centre"]
        ray_b = xyz - other["centre"]
        cosine = np.sum(ray_a * ray_b, axis=1) / (
            np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1) + 1e-12)
        parallax = np.degrees(np.arccos(np.clip(cosine, -1, 1)))

        keep = parallax >= args.min_parallax_deg
        for source, pts_2d in ((view, pts_a.T), (other, pts_b.T)):
            P = source["P"]
            projected = (P @ np.hstack([xyz, np.ones((len(xyz), 1))]).T)
            depth = projected[2]
            uv = (projected[:2] / np.where(depth == 0, 1e-9, depth)).T
            error = np.linalg.norm(uv - pts_2d, axis=1)
            keep &= (depth > 0) & (error < args.max_reproj_px)

        points.append(xyz[keep])
        descs.append(view["desc"][idx[keep, 0]])
        lights.extend([view["light"] or ""] * int(keep.sum()))

    if not points:
        sys.exit("no points survived triangulation")
    points = np.vstack(points)
    descs = np.vstack(descs)
    lights = np.array(lights, dtype="<U16")

    radius = np.linalg.norm(points, axis=1)
    limit = args.max_radius if args.max_radius else 10.0 * args.marker_length
    culled = int(np.sum(radius > limit))
    keep_radius = radius <= limit
    points, descs, lights = points[keep_radius], descs[keep_radius], lights[keep_radius]

    if len(points) > args.max_map_points:
        pick = np.random.default_rng(0).choice(len(points), args.max_map_points, replace=False)
        points, descs, lights = points[pick], descs[pick], lights[pick]

    np.savez(args.out, points=points, descriptors=descs, lights=lights,
             reference_marker=args.reference_marker, marker_length=args.marker_length,
             source_folder=os.path.basename(folder.rstrip("/")))
    # 5th-95th percentile, not min-max: a couple of stray points make min-max meaningless.
    low, high = np.percentile(points, [5, 95], axis=0)
    extent = high - low
    unit = "m" if args.marker_length != 1.0 else "marker sides"
    print(f"{pairs_used} frame pairs triangulated, {len(points)} map points kept "
          f"({culled} beyond {limit:.1f} {unit} of the marker, dropped)")
    print(f"map extent (5-95 pct) {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} {unit}")
    per_light = {light: int(np.sum(lights == light)) for light in sorted(set(lights))}
    print(f"points per illuminator: {per_light}")
    print(f"wrote {args.out}")


def locate(args):
    K, dist = load_intrinsics(args.intrinsics)
    obj = marker_object_points(args.marker_length)
    data = np.load(args.map, allow_pickle=False)
    map_points = data["points"].astype(np.float64)
    map_desc = data["descriptors"].astype(np.float32)
    map_lights = data["lights"] if "lights" in data else np.array([""] * len(map_points))
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

        # Optionally match within the query frame's own illuminator. Measured on this
        # corpus it LOSES: see the docstring. RANSAC already rejects cross-light matches
        # geometrically, and restricting the map to one light throws away real coverage.
        light = (frame_light(path, args.min_light_confidence)
                 if args.pool_by_light else None)
        pool = np.flatnonzero(map_lights == light) if light else np.empty(0, int)
        if len(pool) < args.min_pool:
            pool = np.arange(len(map_points))       # fall back to the whole map
            pooled = False
        else:
            pooled = True
        separation = args.min_separation if args.min_separation else 0.25 * args.marker_length
        idx = match_to_map(descriptors, map_desc[pool], map_points[pool], matcher, separation)
        if len(idx):
            idx[:, 1] = pool[idx[:, 1]]
        row = {"frame": os.path.basename(path), "light": light or "",
               "pooled": int(pooled), "matches": len(idx), "inliers": 0,
               "inlier_ratio": "", "reproj_rms_px": "", "spread_px": "",
               "rvec_x": "", "rvec_y": "", "rvec_z": "",
               "cam_x": "", "cam_y": "", "cam_z": "",
               "rot_err_deg": "", "pos_err": ""}

        pattern_pose, inlier_pts = None, None
        if len(idx) >= 6:
            image_pts = np.array([keypoints[k].pt for k in idx[:, 0]], np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                map_points[idx[:, 1]], image_pts, K, dist,
                reprojectionError=args.max_reproj_px, iterationsCount=500,
                flags=cv2.SOLVEPNP_EPNP)
            if ok and inliers is not None and len(inliers) >= args.min_inliers:
                keep = inliers.ravel()
                object_in = map_points[idx[keep, 1]]
                image_in = image_pts[keep]

                # Refitting the pose to all RANSAC inliers (--refine) ought to help and
                # measurably does not: it improved 17 of 33 held-out frames and worsened
                # 16, a net +0.18 mm. The limiting error is the MAP -- 3D points
                # triangulated from noisy marker scaffolding -- not the fit to it, and
                # polishing a fit against biased points buys nothing. Left as a flag.
                if args.refine and len(keep) >= 4:
                    rvec, tvec = cv2.solvePnPRefineLM(object_in, image_in, K, dist,
                                                      rvec, tvec)

                R, _ = cv2.Rodrigues(rvec)
                centre = (-R.T @ tvec).ravel()
                pattern_pose = (rvec, tvec)
                inlier_pts = image_in

                # Two cheap confidence signals, recorded so a caller can gate on them:
                # how well the kept correspondences fit, and how far across the frame
                # they are spread (a pose carried by one tight cluster is ill-conditioned
                # however many inliers back it).
                projected, _ = cv2.projectPoints(object_in, rvec, tvec, K, dist)
                residual = np.linalg.norm(projected.reshape(-1, 2) - image_in, axis=1)
                spread = float(np.sqrt(image_in[:, 0].var() + image_in[:, 1].var()))

                row.update({"inliers": len(keep),
                            "inlier_ratio": len(keep) / len(idx),
                            "reproj_rms_px": float(np.sqrt((residual ** 2).mean())),
                            "spread_px": spread,
                            "rvec_x": rvec[0, 0], "rvec_y": rvec[1, 0], "rvec_z": rvec[2, 0],
                            "cam_x": centre[0], "cam_y": centre[1], "cam_z": centre[2]})
                if args.compare_markers and marker_pose is not None:
                    R_ref, t_ref = marker_pose
                    delta = R_ref.T @ R
                    angle = np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
                    row["rot_err_deg"] = angle
                    row["pos_err"] = float(np.linalg.norm(centre - (-R_ref.T @ t_ref).ravel()))

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
    pooled_frames = sum(r["pooled"] for r in rows)
    print(f"{len(solved)}/{len(rows)} frames localised against the pattern "
          f"({pooled_frames} matched within their own illuminator, "
          f"{len(rows) - pooled_frames} fell back to the whole map)")
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
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DATA_ROOT)
    sub = parser.add_subparsers(dest="mode", required=True)

    for name, func in (("build", build), ("locate", locate)):
        p = sub.add_parser(name)
        p.add_argument("--folder", required=True)
        p.add_argument("--intrinsics", required=True)
        p.add_argument("--marker-length", type=float, default=0.025,
                       help="printed marker side in metres (measured: 25 mm); pass 1.0 to "
                            "work in marker-side units instead")
        p.add_argument("--stride", type=int, default=5, help="use every Nth frame")
        p.add_argument("--max-features", type=int, default=2000, help="SIFT keypoint cap")
        p.add_argument("--max-reproj-px", type=float, default=3.0)
        p.add_argument("--min-light-confidence", type=float, default=0.5,
                       help="treat frames whose decoded light scores below this as "
                            "unlabelled (mid-transition frames blend two lights); frames "
                            "with no recorded confidence are trusted")
        p.add_argument("--refine", action="store_true",
                       help="refit the pose to the RANSAC inliers with Levenberg-Marquardt "
                            "(measured a wash on this corpus; see the docstring)")
        p.add_argument("--no-clahe", action="store_true",
                       help="skip contrast equalisation before feature detection")
        p.set_defaults(func=func)

    b = sub.choices["build"]
    b.add_argument("--out", default="map.npz")
    b.add_argument("--reference-marker", type=int, default=10,
                   help="marker id whose frame the map is expressed in")
    b.add_argument("--pair-span", type=int, default=4,
                   help="how many later views each view is matched against")
    b.add_argument("--min-baseline", type=float,
                   help="minimum camera separation for a pair, in the same units as "
                        "--marker-length (default: 2 marker sides, i.e. 50 mm, which a "
                        "sweep found optimal -- see the docstring)")
    b.add_argument("--any-light", action="store_true",
                   help="also pair frames lit by different illuminators (off by default: "
                        "the same surface under two lights is two different appearances)")
    b.add_argument("--min-parallax-deg", type=float, default=2.0,
                   help="minimum angle between the two rays to a triangulated point")
    b.add_argument("--max-radius", type=float,
                   help="drop points further than this from the marker "
                        "(default 10 marker sides)")
    b.add_argument("--max-map-points", type=int, default=60000)

    l = sub.choices["locate"]
    l.add_argument("--map", required=True)
    l.add_argument("--out", default="pattern_poses.csv")
    l.add_argument("--min-inliers", type=int, default=12)
    l.add_argument("--pool-by-light", action="store_true",
                   help="restrict matching to map points from the query frame's own "
                        "illuminator (measured worse on this corpus; off by default)")
    l.add_argument("--min-pool", type=int, default=200,
                   help="with --pool-by-light, fall back to the whole map below this many "
                        "points for the frame's illuminator")
    l.add_argument("--min-separation", type=float,
                   help="how far apart two map points must be to count as rivals in the "
                        "ratio test (default 0.25 marker sides)")
    l.add_argument("--debug-dir",
                   help="write each frame with the recovered axes, the marker axes where "
                        "one is visible, and the RANSAC inliers drawn on it")
    l.add_argument("--axis-length", type=float,
                   help="length of the drawn axes (default: one marker side)")
    l.add_argument("--compare-markers", action="store_true",
                   help="also report error against the ArUco pose, where a marker is visible")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
