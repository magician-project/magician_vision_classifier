#!/usr/bin/env python3
"""Camera extrinsics from the ArUco markers in the UniquePattern corpus.

The `Altinay*UniquePattern*` recordings scan the sheet-metal region around the door-mirror
mount with printed ArUco markers (DICT_6X6_250, ids 10/11/12 observed) stuck to the panel.
This tool turns those frames into a per-frame camera pose, which is the ground truth any
later marker-free method -- learned feature matching, dense correspondence regression --
has to be trained against and scored on.

Two modes:

  calibrate   a board capture -> intrinsics.json. The board is 9x6 inner corners of 11.5 mm
              squares, per BOARD_W/BOARD_H/BOARD_DIM in
              magician_grabber/PolarShadowVisionSensorCalibrationFromDatasets.py, which is
              the canonical calibrator for this sensor: it debayers raw polar PNMs and also
              solves hand-eye against logged Doosan poses. This is the cut-down path --
              same board, same intrinsics, no robot side -- and it reads either a flat
              folder of PNGs or a parent holding `calib_frame*` pose dirs of PNMs.

  extrinsics  a UniquePattern folder + intrinsics.json -> one CSV row per detected marker,
              giving the marker pose in the camera frame (rvec/tvec, the solvePnP
              convention) and the camera position in the marker frame (-R'*t), which is
              usually the quantity wanted when the marker is glued to the car.

WHICH CAPTURE TO CALIBRATE FROM -- this corpus has three, and two are traps
--------------------------------------------------------------------------
Focal length trades off against board distance unless the board is seen at a range of
tilts, so a capture can fit its own corners beautifully and still not pin fx or the
principal point. Measured on this corpus:

  calib_frame1..15 (PNM)        15 poses, tilt std 8.4 deg   fx 2268-2414 over subsets  USE
  AltinayemulateCalibration     194 views, tilt std 1.8 deg  fx 3420-3682 over subsets  no
  Altinaychessboardbase         one static pose repeated     degenerate                 no

`Altinaychessboardbase` looks like 557 frames of calibration data and is in fact a single
viewpoint held still: corner centroids vary by 0.0 px across the whole recording. It
cannot calibrate anything. `AltinayemulateCalibration` does move the board but keeps it
nearly fronto-parallel, so its fx swings by 8% depending on which half you fit. Only the
`calib_frame*` pose dirs have real tilt diversity, which is what they were captured for.
This mode prints the tilt spread it actually saw and warns when it is too small to trust.

UNITS -- read this before believing a translation
-------------------------------------------------
Scaling the object points scales the translations and leaves the rotations, the camera
matrix and the distortion coefficients untouched.

The chessboard square is known (11.5 mm), so `calibrate` is metric. The printed ArUco
markers have NOT been measured -- they never share a frame with the board, so nothing in
the corpus pins their size -- and so --marker-length defaults to 1.0, which emits
translations in units of ONE MARKER SIDE. That is deliberate: a plausible-looking default
in metres would produce millimetres that are silently wrong. Measure a marker, pass
--marker-length 0.0138 (or whatever it is), and every translation becomes metric.
Rotations are correct either way.

Usage:
  python analysis/extrinsics_from_markers.py calibrate --folder . --out intrinsics.json

  python analysis/extrinsics_from_markers.py extrinsics --folder AltinayUniquePattern750 \
      --intrinsics intrinsics.json --out poses.csv [--marker-length 0.0138] \
      [--stride 1] [--debug-dir overlays/]

Folders resolve against --data-root (default /media/ammar/games2/Datasets/Magician) unless
given as an absolute path.
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

DATA_ROOT = "/media/ammar/games2/Datasets/Magician"
ARUCO_DICT = cv2.aruco.DICT_6X6_250   # matches aruco_create.py in the D-PoSE checkout
MIN_TILT_STD_DEG = 5.0                # below this, fx and the board distance are confounded


def resolve(folder, data_root):
    path = folder if os.path.isabs(folder) else os.path.join(data_root, folder)
    if not os.path.isdir(path):
        sys.exit(f"no such folder: {path}")
    return path


def png_mono(path):
    """Unpolarized intensity from a corpus PNG: the mean of the 4 DoFP channels.

    These PNGs are 4-channel [I90, I45, I0, I135] (see mvc/core/polarization.py). A plain
    cv2.imread() drops I135 and luma-weights what is left, so its "grayscale" is a
    polarization- and illuminator-dependent mix -- the one thing you do not want when
    matching a surface across frames lit from different directions. Averaging all four
    gives S0/2, the mono image the sensor would produce without the polarizer, and matches
    what the grabber's calibrator does with raw PNMs.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 2:
        return raw
    return raw.astype(np.float32).mean(axis=2).astype(np.uint8)


def pnm_gray(path):
    """Debayer a raw polar PNM to the mean of its 4 polarization channels.

    The 2x2 mosaic layout follows debayerPolarImage() in the grabber's calibrator, and
    the result is half the raw resolution -- which is exactly the size of the corpus
    PNGs, so intrinsics from PNM views apply unchanged to the PNG recordings.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        return None
    channels = [raw[1::2, 1::2], raw[0::2, 1::2], raw[0::2, 0::2], raw[1::2, 0::2]]
    mean = np.mean(channels, axis=0)
    return cv2.normalize(mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def calibration_views(folder, stride, pose_glob):
    """(group, path, grayscale) triples: flat PNGs, or PNMs under `calib_frame*` dirs.

    The group is what a single board pose is: a PNG folder is a stream so every frame is
    its own group, while a pose dir holds repeats of one pose and contributes one view,
    as the grabber's calibrator does.
    """
    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    if pngs:
        for path in pngs[::stride]:
            yield path, path, png_mono(path)
        return

    pose_dirs = sorted(glob.glob(os.path.join(folder, pose_glob)))
    if not pose_dirs:
        sys.exit(f"no *.png frames and no {pose_glob} pose dirs in {folder}")
    for pose_dir in pose_dirs:
        for path in sorted(glob.glob(os.path.join(pose_dir, "colorFrame_0_*.pnm"))):
            yield pose_dir, path, pnm_gray(path)


def board_tilt_degrees(rvecs):
    """Angle between each board normal and the optical axis."""
    tilts = []
    for rvec in rvecs:
        R, _ = cv2.Rodrigues(rvec)
        tilts.append(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1.0, 1.0))))
    return np.array(tilts)


def calibrate(args):
    """Chessboard intrinsics. --square-size only scales the per-view translations."""
    cols, rows = (int(v) for v in args.grid.lower().split("x"))
    folder = resolve(args.folder, args.data_root)

    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points, img_points, size = [], [], None
    seen, groups_used, unreadable = 0, set(), 0
    for group, path, gray in calibration_views(folder, args.stride, args.pose_dirs):
        if gray is None:                   # one frame in the corpus is a truncated PNG
            unreadable += 1
            continue
        seen += 1
        if group in groups_used:           # one view per board pose
            continue
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not ok:
            continue
        obj_points.append(grid)
        img_points.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
        groups_used.add(group)
        size = gray.shape[::-1]

    print(f"{len(obj_points)}/{seen} frames with a complete {cols}x{rows} board"
          + (f" ({unreadable} unreadable)" if unreadable else ""))
    if len(obj_points) < 5:
        sys.exit("too few usable views to calibrate")

    rms, K, dist, rvecs, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    tilts = board_tilt_degrees(rvecs)

    out = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "image_size": list(size),
        "rms_reproj_px": float(rms),
        "views_used": len(obj_points),
        "tilt_deg_std": float(tilts.std()),
        "tilt_deg_max": float(tilts.max()),
        "source_folder": os.path.basename(folder.rstrip("/")),
        "square_size": args.square_size,
    }
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)

    print(f"RMS reprojection error {rms:.3f} px")
    print(f"board tilt: std {tilts.std():.1f} deg, max {tilts.max():.1f} deg")
    print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    if tilts.std() < MIN_TILT_STD_DEG:
        print(f"WARNING: tilt spread below {MIN_TILT_STD_DEG} deg -- the board is close to "
              f"fronto-parallel throughout, so fx trades off against board distance and "
              f"this K is not trustworthy. Calibrate from the calib_frame* pose dirs.")
    print(f"wrote {args.out}")


def extrinsics(args):
    """Per-frame marker poses. --marker-length only scales the translations."""
    with open(args.intrinsics) as handle:
        intr = json.load(handle)
    K = np.array(intr["camera_matrix"], dtype=np.float64)
    dist = np.array(intr["dist_coeffs"], dtype=np.float64)

    half = args.marker_length / 2.0
    # solvePnP object points, counter-clockwise from top-left, matching detectMarkers.
    obj = np.array([[-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)

    folder = resolve(args.folder, args.data_root)
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))[::args.stride]
    if not paths:
        sys.exit(f"no .png frames in {folder}")
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    rows, frames_with_marker, unreadable = [], 0, 0
    for path in paths:
        mono = png_mono(path)
        if mono is None:               # one frame in the corpus is a truncated PNG
            unreadable += 1
            continue
        corners, ids, _ = detector.detectMarkers(mono)
        if ids is None:
            continue
        frames_with_marker += 1
        canvas = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR) if args.debug_dir else None
        for corner, marker_id in zip(corners, ids.ravel()):
            img_pts = corner[0].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            reproj = float(np.linalg.norm(projected.reshape(-1, 2) - img_pts, axis=1).mean())
            R, _ = cv2.Rodrigues(rvec)
            cam_in_marker = (-R.T @ tvec).ravel()
            rows.append({
                "frame": os.path.basename(path),
                "marker_id": int(marker_id),
                "rvec_x": rvec[0, 0], "rvec_y": rvec[1, 0], "rvec_z": rvec[2, 0],
                "tvec_x": tvec[0, 0], "tvec_y": tvec[1, 0], "tvec_z": tvec[2, 0],
                "cam_x": cam_in_marker[0], "cam_y": cam_in_marker[1], "cam_z": cam_in_marker[2],
                "reproj_px": reproj,
            })
            if args.debug_dir:
                cv2.drawFrameAxes(canvas, K, dist, rvec, tvec, args.marker_length * 0.5)
        if args.debug_dir:
            cv2.aruco.drawDetectedMarkers(canvas, corners, ids)
            cv2.imwrite(os.path.join(args.debug_dir, os.path.basename(path)), canvas)

    if not rows:
        sys.exit("no markers detected in any frame")

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unit = "m" if args.marker_length != 1.0 else "marker sides"
    errors = np.array([r["reproj_px"] for r in rows])
    print(f"{frames_with_marker}/{len(paths)} frames with a marker, {len(rows)} poses"
          + (f" ({unreadable} unreadable)" if unreadable else ""))
    print(f"marker ids: {sorted({r['marker_id'] for r in rows})}")
    print(f"reprojection error: mean {errors.mean():.3f} px, max {errors.max():.3f} px")
    print(f"translations are in {unit}")
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DATA_ROOT)
    sub = parser.add_subparsers(dest="mode", required=True)

    cal = sub.add_parser("calibrate", help="board capture -> intrinsics.json")
    cal.add_argument("--folder", default=".",
                     help="flat folder of PNGs, or parent of the calib_frame* pose dirs "
                          "(default: the data root, which is where they live)")
    cal.add_argument("--out", default="intrinsics.json")
    cal.add_argument("--grid", default="9x6", help="inner corners, COLSxROWS")
    cal.add_argument("--square-size", type=float, default=0.0115,
                     help="square side in metres (default 11.5 mm, the measured board)")
    cal.add_argument("--pose-dirs", default="calib_frame*",
                     help="glob for pose dirs holding PNMs, used when there are no PNGs")
    cal.add_argument("--stride", type=int, default=10,
                     help="use every Nth frame (PNG folders only)")
    cal.set_defaults(func=calibrate)

    ext = sub.add_parser("extrinsics", help="marker folder -> per-frame poses CSV")
    ext.add_argument("--folder", required=True)
    ext.add_argument("--intrinsics", required=True)
    ext.add_argument("--out", default="poses.csv")
    ext.add_argument("--marker-length", type=float, default=1.0,
                     help="printed marker side; 1.0 (default) leaves translations "
                          "in marker sides")
    ext.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ext.add_argument("--debug-dir", help="write frames with detections and axes drawn")
    ext.set_defaults(func=extrinsics)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
