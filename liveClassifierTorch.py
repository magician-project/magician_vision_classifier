#!/usr/bin/python3

"""
Author : "Nikos Vasilikopoulos, Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Live/CLI streaming entry point. The classifier core (ClassifierPnm, tiling,
heatmaps, model_scan, ...) lives in classifierPnm.py; everything is re-exported
here so existing `from liveClassifierTorch import ...` imports keep working.

This module ALSO owns every piece of the runtime that is not ROS-specific --
the deployment presets, the laser/marker geometry, the detection bookkeeping and
the frame loop -- so that liveClassifierTorchROS.py is a thin ROS shell over the
same code instead of a second implementation that drifts away from this one.
Run it standalone with:

    python3 liveClassifierTorch.py [--config NAME] [--model NAME] ...

Feature parity with the ROS node: same presets, same auto-download, same gate /
step / voting / erosion / frame-limiter / FPS knobs, same two-stage ensemble,
same snapshot + sidecar JSON files, same ArUco marker scan, same depth fusion
maths. What the ROS node exposes as services is reachable here through command
line flags and single-key commands (press 'h' for the list); what it publishes
as topics is printed and, with --detections-jsonl, appended to a JSONL file.
The one thing standalone mode cannot reproduce is the laser SUBSCRIPTIONS --
there is no ROS to receive them -- so depths come from --laser-depths instead.
"""

from classifierPnm import *   # noqa: F401,F403 -- re-export the classifier core

import os
import sys
import json
import time
import math
import select
import argparse
import threading
from datetime import datetime

import cv2
import numpy as np
import torch


# ========================================================
# Deployment presets
# ========================================================
# Which model to run and at what operating point comes from recommended_configuration.json
# next to this script. That file is COMMITTED TO GIT, so a deployment site picks up new
# models and thresholds with a plain `git pull` -- deliberately NOT environment variables,
# which are awkward to change on-site.
#
# The FIRST entry is the startup default; pass `--config NAME` to select another.
# The model's .pth/.json are auto-fetched on first run (ModelDownload.ensure_model).
RECOMMENDED_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "recommended_configuration.json")

# Fallback used only if recommended_configuration.json is missing or unreadable, so the
# runtime still starts on a sane model rather than dying at the first import.
FALLBACK_PRESET = {
    "name": "fallback",
    "model": "mix_convnext_tiny",
    "gate": {"mode": "defect_mass", "threshold": 0.90, "assign_best_defect_class": True},
    "runtime": {"step": 18, "target_fps": 23.0, "erosion_kernel": 1, "min_votes": 2,
                "majority_voting": True, "frame_limiter": True, "two_stage": False},
}


def load_recommended_configuration(name=None, path=RECOMMENDED_CONFIG_FILE):
    """Return one preset dict from recommended_configuration.json.

    name=None -> the first entry (the committed default). Otherwise the entry whose
    "name" matches. Falls back to FALLBACK_PRESET, never raises, because failing to
    read a preferences file must not stop the runtime from running.
    """
    try:
        with open(path, "r") as f:
            doc = json.load(f)
        presets = doc.get("configurations") or []
        if not presets:
            raise ValueError("no 'configurations' entries")
        if name is None:
            chosen = presets[0]
        else:
            matches = [p for p in presets if p.get("name") == name]
            if not matches:
                available = [p.get("name") for p in presets]
                raise ValueError(f"preset {name!r} not found; available: {available}")
            chosen = matches[0]
        # Merge over the fallback so a partial preset cannot leave a key undefined.
        merged = dict(FALLBACK_PRESET)
        merged.update(chosen)
        merged["gate"] = {**FALLBACK_PRESET["gate"], **(chosen.get("gate") or {})}
        merged["runtime"] = {**FALLBACK_PRESET["runtime"], **(chosen.get("runtime") or {})}
        return merged
    except Exception as e:
        print(f"[config] could not use {path} ({e!r}) — falling back to "
              f"{FALLBACK_PRESET['model']}")
        return dict(FALLBACK_PRESET)


def list_recommended_configurations(path=RECOMMENDED_CONFIG_FILE):
    """Return the raw preset list (empty on any failure), for `--list-configs`."""
    try:
        with open(path, "r") as f:
            return json.load(f).get("configurations") or []
    except Exception as e:
        print(f"[config] could not read {path} ({e!r})")
        return []


# Two-stage ensemble members. This path is OPTIONAL: if any member cannot be resolved the
# ensemble is skipped and the runtime still starts on the single classifier (previously a
# missing member called sys.exit(1) inside ClassifierPnm and killed the process before it
# ever reported a detection).
ENSEMBLE_STAGE1 = "binary_small_cnn"
ENSEMBLE_MEMBERS = [
    "allclass_verysmall_cnn",
    "allclass_resnet18",
    "allclass_resnext50",
    "allclass_convnext_tiny",
]


# ========================================================
# Laser fusion geometry (project-specific / fixed hardware)
# ========================================================
# The ROS node subscribes to the three laser topics; standalone mode has no such
# source and takes fixed values from --laser-depths. The geometry and the
# interpolation itself are shared so both paths report the same depth.

# Laser locations in the classifier's 2D image plane (pixels)
# (x0,y0), (x1,y1), (x2,y2)
LASER_XY_PIXELS = [
    (120.0, 200.0),
    (320.0, 200.0),
    (520.0, 200.0),
]

LASER_IDW_POWER = 2.0  # IDW interpolation power


def idw_depth(x: float, y: float, xy_list, d_list, p: float = 2.0) -> float:
    """
    Inverse Distance Weighting (IDW) interpolation at point (x, y) from 3 known samples.

    Each sample is a (pixel_x, pixel_y) position with a measured depth value.
    Returns NaN if no finite samples are available. Uses power parameter *p*
    (default 2.0, matching LASER_IDW_POWER).
    """
    # Exact hit
    for (sx, sy), d in zip(xy_list, d_list):
        if sx == x and sy == y:
            return float(d)

    wsum = 0.0
    acc = 0.0
    for (sx, sy), d in zip(xy_list, d_list):
        r = math.hypot(x - sx, y - sy)
        r = max(r, 1e-6)
        w = 1.0 / (r ** p)
        wsum += w
        acc += w * float(d)

    if wsum <= 0.0:
        return float("nan")
    return float(acc / wsum)


# ========================================================
# Marker scanning globals
# ========================================================
MARKER_SCAN_DURATION_S   = 3.0          # seconds each marker scan stays active
ARUCO_DICT_NAME          = "DICT_6X6_250"
DEFAULT_MARKER_LENGTH_M  = 0.05         # 5 cm default ArUco marker side length
CHESSBOARD_W             = 9            # inner corner columns
CHESSBOARD_H             = 6            # inner corner rows
CHESSBOARD_SQUARE_M      = 0.024        # 24 mm square size


def estimatePoseSingleMarkers(corners_list, marker_length, K, dist):
    """
    Estimate camera pose from ArUco marker corners (OpenCV 4.7+ compatible).

    Wraps cv2.aruco.estimatePoseSingleMarkers when available; falls back to
    cv2.solvePnP with IPPE_SQUARE for newer OpenCV versions where the function
    was removed. Returns (rvecs, tvecs) as lists of (3,) arrays.
    """
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners_list, marker_length, K, dist
        )
        return [r.reshape(3) for r in rvecs], [t.reshape(3) for t in tvecs]

    half = marker_length / 2.0
    marker_objp = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    rvecs, tvecs = [], []
    for corners in corners_list:
        img_pts = corners.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            marker_objp, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(rvec.reshape(3) if ok else np.zeros(3))
        tvecs.append(tvec.reshape(3) if ok else np.zeros(3))
    return rvecs, tvecs


def make_approx_camera_matrix(width, height):
    """
    Return an approximate pinhole camera matrix (K) and zero distortion coefficients (dist).

    Uses fx = fy = 0.9 * max(width, height) with principal point at image center.
    This is a rough estimate suitable for ArUco pose estimation when a calibrated
    camera matrix is not available.
    """
    fx = fy = 0.9 * max(width, height)
    cx = width  / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    return K, dist


def rvec_to_quaternion(rvec):
    """
    Convert an OpenCV Rodrigues rotation vector to a (qx, qy, qz, qw) quaternion.

    Returns identity (0, 0, 0, 1) for zero-norm rotation vectors.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-10:
        return 0.0, 0.0, 0.0, 1.0  # identity
    axis = rvec / angle
    s = math.sin(angle / 2.0)
    return (float(axis[0] * s),
            float(axis[1] * s),
            float(axis[2] * s),
            float(math.cos(angle / 2.0)))


# ========================================================
# Helpers
# ========================================================
# Default box the visualization window is fitted into. --window-scale multiplies it.
VISUALIZATION_MAX_W = 1280
VISUALIZATION_MAX_H = 720
WINDOW_SCALE_MIN    = 0.25
WINDOW_SCALE_MAX    = 8.0


def resize_to_fit_screen(img, max_w=VISUALIZATION_MAX_W, max_h=VISUALIZATION_MAX_H, only_shrink=True):
    """
    Resize an image to fit within (max_w, max_h) while preserving aspect ratio.

    When only_shrink=True and the image is smaller than the limits, it is returned
    unchanged. Uses INTER_AREA for downscaling and INTER_LINEAR for upscaling.
    Returns (resized_img, scale_factor).
    """
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h))

    if only_shrink and scale >= 1.0:
        return img, 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp), scale


def scale_window_image(img, scale):
    """
    Multiply an already-fitted visualization image by *scale* for display.

    scale=1.0 returns the image untouched (the historical behaviour). Nearest
    neighbour is used when magnifying so the tile grid of a heatmap stays crisp
    instead of being smeared by interpolation.
    """
    if scale == 1.0:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def filter_type(det_type: str):
    """
    Strip the 'class_' prefix and trailing severity suffix from a detection label.

    Example: 'class_NegativeDentClassB' -> ('NegativeDent', 'ClassB')

    Args:
        det_type: Raw detection type string (e.g. 'class_NegativeDentClassB').

    Returns:
        (clean_type, det_class) where det_class is one of 'ClassA/B/C' or 'Unknown'.
    """
    det_class  = "Unknown"
    clean_type = det_type

    if clean_type.startswith("class_"):
        clean_type = clean_type[len("class_"):]

    for cls in ("ClassA", "ClassB", "ClassC"):
        if clean_type.endswith(cls):
            det_class  = cls
            clean_type = clean_type[:-len(cls)]
            break

    return clean_type, det_class


# DetectionM.msg severity convention: ClassA=1, ClassB=2, ClassC=3, unknown=0
_SEVERITY_MAP: dict[str, int] = {"ClassA": 1, "ClassB": 2, "ClassC": 3}

def class_to_severity(det_class: str) -> int:
    """Map a severity class label (ClassA/B/C) to the DetectionM integer severity field."""
    return _SEVERITY_MAP.get(det_class, 0)


class ConsoleLogger:
    """Stand-in for rclpy's node logger so the standalone runtime logs like the node."""

    def __init__(self, name="magician_vision_classifier", debug=False):
        self.name   = name
        self._debug = debug

    def _emit(self, level, text):
        print(f"[{level}] [{self.name}]: {text}", flush=True)

    def info(self, text):     self._emit("INFO", text)
    def warning(self, text):  self._emit("WARN", text)
    def error(self, text):    self._emit("ERROR", text)
    def debug(self, text):
        if self._debug:
            self._emit("DEBUG", text)


class KeyboardControl:
    """
    Non-blocking single-key reader, the standalone stand-in for the ROS services.

    Keys arrive either from the OpenCV window (when visualization is on) or from
    the terminal, which is put in cbreak mode so a keypress needs no Enter. With
    no TTY (a log-redirected deployment run) it simply returns nothing and the
    loop runs on its command line defaults.
    """

    def __init__(self, enabled=True):
        self._fd      = None
        self._saved   = None
        self._termios = None
        if not enabled:
            return
        try:
            if not sys.stdin.isatty():
                return
            import termios, tty
            self._termios = termios
            self._fd      = sys.stdin.fileno()
            self._saved   = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            # Any terminal we cannot put in cbreak mode simply gets no key control.
            self._fd    = None
            self._saved = None

    @property
    def active(self):
        return self._fd is not None

    def read(self, window_open=False):
        """Return the next pending key as a 1-char string, or None."""
        if window_open:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                return chr(key)
        if self._fd is not None:
            try:
                ready, _, _ = select.select([self._fd], [], [], 0)
                if ready:
                    ch = os.read(self._fd, 1)
                    if ch:
                        return ch.decode("utf-8", "ignore")
            except Exception:
                return None
        return None

    def restore(self):
        if self._fd is not None and self._saved is not None:
            try:
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._saved)
            except Exception:
                pass
        self._fd = None


# ========================================================
# Standalone runtime (the non-ROS twin of DefectPublisher)
# ========================================================
class LiveClassifier:
    """
    Runtime state + detection sink for standalone (non-ROS) inference.

    Holds exactly the state DefectPublisher holds and exposes the same
    thread-safe getters, so the frame loop below is line-for-line the node's
    loop. What the node does with services this class does with set_* methods
    driven by keystrokes, and what the node publishes on topics this class
    prints and optionally appends to a JSONL file.
    """

    def __init__(self, detections_jsonl=None, quiet=False, verbose_detections=False,
                 laser_depths=None, debug=False):
        self.logger = ConsoleLogger(debug=debug)

        # "Publishers": a JSONL sink instead of ROS topics
        self._jsonl_path  = detections_jsonl
        self._jsonl_file  = open(detections_jsonl, "a") if detections_jsonl else None
        self._quiet       = quiet
        self._verbose     = verbose_detections
        self._frame_detections = []   # accumulated per frame, flushed by flush_frame()
        self._frame_markers    = []

        # Last received frame (for saving)
        self._last_frame = None

        # Where to store images
        self._output_path = "./data"
        os.makedirs(self._output_path, exist_ok=True)
        self._snapshot_path = "./snapshots"
        os.makedirs(self._snapshot_path, exist_ok=True)

        # Internal execution state. Visualization defaults ON here (the whole point of
        # running standalone is watching the heatmap); the node defaults it off.
        self._visualization_enabled = True
        # On-screen size multiplier for the visualization window. 1.0 = the heatmap
        # fitted inside VISUALIZATION_MAX_W x VISUALIZATION_MAX_H (the old behaviour);
        # 2.0 shows that same view twice as big, for inspecting tiles on a large
        # monitor. Purely a display setting -- inference is untouched.
        self._window_scale = 1.0
        self._inference_paused = False
        self._two_stage_enabled = False
        self._autosave_defect_snapshots = False
        self._frame_limiter = True

        # Runtime tunables (dynamic via keystrokes)
        self._target_fps = 23.0
        self._step_size = 18
        # Gate score threshold. NOTE the semantics depend on the model's gate MODE
        # (classifierPnm.gate_tiles): under the default "defect_mass" this thresholds
        # 1 - P(clean), NOT the max softmax probability, so it is NOT comparable to a
        # max_prob threshold of the same numeric value.
        #
        # 0.90 is a deliberate FALSE-ALARM-SUPPRESSING choice and is intentionally
        # stricter than the model's own KPI-optimal gate: a frame holds thousands of
        # tiles, so a per-tile FA rate that looks small becomes many crosses per
        # frame. On allclass_forthalt_custom the trainer's sweep gives
        #   0.675 (model KPI gate) -> detect 88.9%  FA 15.92%
        #   0.900 (this default)   -> detect 73.6%  FA  1.97%
        # The cost in missed defects is real, so it is logged at startup and on every
        # change rather than left implicit -- see _log_threshold_tradeoff.
        #
        # Set to None (or call set_threshold with a negative value) to FOLLOW THE
        # MODEL's own calibrated gate instead of pinning a value here.
        self._threshold = 0.90
        self._erosion_kernel = 1   # neighborhood radius for tile voting: (2k+1)^2 tiles
        self._min_votes = 2        # activated tiles (incl. itself) required in the neighborhood to accept a tile; 0/1 = voting off
        self._majority_voting = True

        self._lock = threading.Lock()

        # Model hot-swap state
        self._single_classifier = None
        self._model_dir = "."
        self._model_lock = threading.Lock()

        # Last inference results (for saving alongside frames)
        self._last_responses = None
        self._last_tile_size = 0
        self._last_frame_timestamp = 0

        # Marker scanning state
        self._marker_scan_until = 0.0   # monotonic time until which scanning is active
        _aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, ARUCO_DICT_NAME)
        )
        self._aruco_detector = cv2.aruco.ArucoDetector(
            _aruco_dict, cv2.aruco.DetectorParameters()
        )
        self._cb_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
        self._cam_matrix_cache = {}   # (h, w) -> (K, dist)

        # Laser state. Standalone has no laser subscriptions, so these stay at the
        # fixed values given on the command line (NaN = "no depth", which is what an
        # un-fused detection reports).
        self._laser_depths = [float("nan"), float("nan"), float("nan")]
        self._use_lasers = False
        if laser_depths is not None:
            if len(laser_depths) != 3:
                self.logger.error("--laser-depths needs exactly 3 values. Disabling lasers.")
            elif len(LASER_XY_PIXELS) != 3:
                self.logger.error("LASER_XY_PIXELS must have exactly 3 (x,y) pairs. Disabling lasers.")
            else:
                self._laser_depths = [float(d) for d in laser_depths]
                self._use_lasers = True

        if self._use_lasers:
            self.logger.info(f"Laser fusion ENABLED (fixed depths {self._laser_depths}) "
                             f"xy={LASER_XY_PIXELS} p={LASER_IDW_POWER}")
        else:
            self.logger.info("Laser fusion DISABLED (no laser source outside ROS; see --laser-depths)")

    # -------------------------
    # Presets
    # -------------------------
    def apply_preset(self, preset):
        """Adopt a recommended_configuration.json preset as the startup state.

        These are only DEFAULTS -- every one stays overridable at runtime through the
        keyboard commands, so an operator can still retune live without editing the file.
        """
        rt = preset.get("runtime") or {}
        gate = preset.get("gate") or {}
        with self._lock:
            self._step_size          = int(rt.get("step", self._step_size))
            self._target_fps         = float(rt.get("target_fps", self._target_fps))
            self._erosion_kernel     = int(rt.get("erosion_kernel", self._erosion_kernel))
            self._min_votes          = int(rt.get("min_votes", self._min_votes))
            self._majority_voting    = bool(rt.get("majority_voting", self._majority_voting))
            self._frame_limiter      = bool(rt.get("frame_limiter", self._frame_limiter))
            self._two_stage_enabled  = bool(rt.get("two_stage", self._two_stage_enabled))
            if gate.get("threshold") is not None:
                self._threshold = float(gate["threshold"])
        m = preset.get("measured") or {}
        self.logger.info(
            f"Preset '{preset.get('name','?')}': model={preset.get('model')} "
            f"gate={gate.get('mode')}@{gate.get('threshold')} step={self._step_size} "
            f"fps={self._target_fps} erosion_kernel={self._erosion_kernel} "
            f"min_votes={self._min_votes} majority_voting={self._majority_voting}")
        if preset.get("description"):
            self.logger.info(f"  {preset['description']}")
        if "detected" in m and "false_alarm" in m:
            self.logger.info(
                f"  expected (from {m.get('source','curve')}): detects {m['detected']:.1%} "
                f"of defect tiles, false-alarms on {m['false_alarm']:.2%} of clean tiles")

    def apply_cli_overrides(self, args):
        """Command line flags win over the preset, the same way a service call would."""
        with self._lock:
            if args.step is not None:            self._step_size = max(1, int(args.step))
            if args.fps is not None:             self._target_fps = max(0.0, float(args.fps))
            if args.erosion_kernel is not None:  self._erosion_kernel = max(0, min(5, int(args.erosion_kernel)))
            if args.min_votes is not None:       self._min_votes = max(0, int(args.min_votes))
            if args.majority_voting is not None: self._majority_voting = bool(args.majority_voting)
            if args.frame_limiter is not None:   self._frame_limiter = bool(args.frame_limiter)
            if args.two_stage is not None:       self._two_stage_enabled = bool(args.two_stage)
            if args.visualization is not None:   self._visualization_enabled = bool(args.visualization)
            if args.window_scale is not None:
                self._window_scale = max(WINDOW_SCALE_MIN, min(WINDOW_SCALE_MAX, float(args.window_scale)))
            if args.autosave_defects:            self._autosave_defect_snapshots = True
            if args.threshold is not None:
                self._threshold = None if args.threshold < 0.0 else max(0.0, min(1.0, float(args.threshold)))
        if args.output_path:
            self._output_path = args.output_path
            os.makedirs(self._output_path, exist_ok=True)
        if args.snapshot_path:
            self._snapshot_path = args.snapshot_path
            os.makedirs(self._snapshot_path, exist_ok=True)

    # -------------------------
    # Setters (the standalone twin of the service callbacks)
    # -------------------------
    def set_visualization(self, enabled):
        """Toggle visualization on/off."""
        with self._lock:
            self._visualization_enabled = bool(enabled)
        if not enabled:
            cv2.destroyAllWindows()
        self.logger.info("Visualization ENABLED" if enabled else "Visualization DISABLED")

    def set_window_scale(self, scale):
        """Scale the visualization window up/down; 1.0 is the default fit-to-screen size."""
        scale = max(WINDOW_SCALE_MIN, min(WINDOW_SCALE_MAX, float(scale)))
        with self._lock:
            self._window_scale = scale
        # The window itself needs no resizing: it is a WINDOW_AUTOSIZE one, so the
        # next imshow of a bigger image grows it.
        self.logger.info(f"Visualization window scale set to {scale:.2f}x")

    def pause_inference(self, paused):
        """Pause/resume inference."""
        with self._lock:
            self._inference_paused = bool(paused)
        self.logger.info("Inference PAUSED" if paused else "Inference RESUMED")

    def set_two_stage(self, enabled):
        """Toggle two-stage ensemble mode on/off."""
        with self._lock:
            self._two_stage_enabled = bool(enabled)
        self.logger.info("Two-stage execution ENABLED" if enabled else "Two-stage execution DISABLED")

    def set_fps(self, fps):
        """Set target FPS (0 = no limiting)."""
        with self._lock:
            self._target_fps = max(0.0, float(fps))  # 0 => no limiting
        self.logger.info(f"Target FPS set to {self._target_fps}")

    def set_step(self, step):
        """Set tile step size (minimum 1)."""
        with self._lock:
            self._step_size = max(1, int(step))
        self.logger.info(f"Step size set to {self._step_size}")

    def set_threshold(self, raw):
        """Set the gate score threshold. Semantics depend on the model's gate mode
        (under the default "defect_mass" this thresholds 1 - P(clean), not the max
        softmax probability). A NEGATIVE value clears the override and follows the
        model's own calibrated gate. The expected detection / false-alarm trade-off
        at the new setting is looked up from the model's threshold curve and
        logged."""
        raw = float(raw)
        thr = None if raw < 0.0 else max(0.0, min(1.0, raw))
        with self._lock:
            self._threshold = thr
        if thr is None:
            self.logger.info("Threshold override CLEARED — following the model's calibrated gate")
        else:
            self.logger.info(f"Gate threshold set to {thr:.3f}\n" + self._threshold_tradeoff_text(thr))

    def _threshold_tradeoff_text(self, threshold):
        """Expected trade-off at `threshold`, from the active model's curve."""
        clf = self._single_classifier
        if clf is None or not hasattr(clf, "format_threshold_tradeoff"):
            return "  (no classifier loaded yet — trade-off unavailable)"
        try:
            return clf.format_threshold_tradeoff(threshold)
        except Exception as e:
            return f"  (threshold curve lookup failed: {e})"

    def _log_threshold_tradeoff(self, threshold, context=""):
        """Log what the current gate setting buys and costs. Called at startup and
        whenever the value actually changes — never per frame."""
        self.logger.info(f"{context}{self._threshold_tradeoff_text(threshold)}")

    def set_erosion_kernel(self, k):
        """Set the voting neighborhood radius k; votes are counted over the (2k+1)^2 tiles around each activation."""
        k = max(0, min(5, int(k)))
        with self._lock:
            self._erosion_kernel = k
        self.logger.info(f"Erosion kernel set to {self._erosion_kernel} "
                         f"(neighborhood {(2*self._erosion_kernel+1)**2} tiles)")

    def set_min_votes(self, v):
        """Require N activated tiles (including the tile itself) in the voting neighborhood for an activation to be accepted. 0/1 disables voting."""
        with self._lock:
            self._min_votes = max(0, int(v))
        self.logger.info(f"Minimum votes set to {self._min_votes}")

    def set_autosave_defect_snapshots(self, enabled):
        """Enable/disable automatic saving of frames when a defect is detected."""
        with self._lock:
            self._autosave_defect_snapshots = bool(enabled)
        self.logger.info("Autosave defect snapshots ENABLED" if enabled
                         else "Autosave defect snapshots DISABLED")

    def set_frame_limiter(self, enabled):
        """Enable/disable the duplicate-frame limiter (False = unlimited framerate)."""
        with self._lock:
            self._frame_limiter = bool(enabled)
        self.logger.info("Frame limiter ENABLED" if enabled
                         else "Frame limiter DISABLED (unlimited framerate)")

    def set_majority_voting(self, enabled):
        """Enable/disable majority voting across inference tiles."""
        with self._lock:
            self._majority_voting = bool(enabled)
        self.logger.info("Majority voting ENABLED" if enabled else "Majority voting DISABLED")

    def set_model(self, name):
        """Hot-swap the single classifier model at runtime. Returns (ok, message)."""
        name = str(name).strip()

        # Support both a bare stem ("allclass_resnet18") and an absolute path stem
        if os.sep in name or "/" in name:
            directory = os.path.dirname(os.path.abspath(name))
            stem = os.path.basename(name)
        else:
            directory = os.path.abspath(self._model_dir)
            stem = name

        model_path = os.path.join(directory, f"{stem}.pth")
        cfg_path   = os.path.join(directory, f"{stem}.json")

        if not os.path.isfile(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False, f"Model file not found: {model_path}"

        if not os.path.isfile(cfg_path):
            self.logger.error(f"Config file not found: {cfg_path}")
            return False, f"Config file not found: {cfg_path}"

        if self._single_classifier is None:
            self.logger.error("Single classifier not yet initialized")
            return False, "Single classifier not yet initialized"

        self.logger.info(f"Hot-swapping model to '{stem}' from {directory} ...")
        with self._model_lock:
            ok = self._single_classifier.reload_model(directory, stem)

        if ok:
            with self._lock:
                self._model_dir = directory

        message = f"Model reloaded: {stem}" if ok else f"Failed to reload model: {stem}"
        self.logger.info(message)
        return ok, message

    def scan_markers(self):
        """Activate marker scanning for MARKER_SCAN_DURATION_S seconds."""
        with self._lock:
            self._marker_scan_until = time.monotonic() + MARKER_SCAN_DURATION_S
        self.logger.info(f"Marker scanning active for {MARKER_SCAN_DURATION_S:.0f} s")

    def remember_defect(self):
        """Save the current frame tagged as a defect."""
        success, msg = self._save_current_frame("defect")
        self.logger.info(msg)
        return success, msg

    def remember_clean(self):
        """Save the current frame tagged as clean."""
        success, msg = self._save_current_frame("clean")
        self.logger.info(msg)
        return success, msg

    def snapshot(self):
        """Save the current frame on demand to the snapshots directory."""
        with self._lock:
            frame = self._last_frame

        if frame is None:
            self.logger.warning("No frame available to save.")
            return False, "No frame available to save."

        full_path = os.path.join(
            self._snapshot_path,
            self._make_timestamped_basename("snapshot") + ".png",
        )
        try:
            cv2.imwrite(full_path, frame)
            message = f"Saved: {full_path}"
            ok = True
        except Exception as e:
            message = str(e)
            ok = False
        self.logger.info(message)
        return ok, message

    # -------------------------
    # Frame saving
    # -------------------------
    @staticmethod
    def _make_timestamped_basename(prefix: str) -> str:
        """Return a filename stem like '<prefix>_YYYY_MM_DD_HH_MM_SS_mmm' using the current wall time."""
        now = datetime.now()
        return (
            f"{prefix}_"
            f"{now.year:04d}_{now.month:02d}_{now.day:02d}_"
            f"{now.hour:02d}_{now.minute:02d}_{now.second:02d}_"
            f"{int(now.microsecond / 1000):03d}"
        )

    def _save_current_frame(self, prefix: str):
        """
        Save the last received frame as a PNG with a timestamped filename, plus a
        JSON sidecar with the same basename containing the current detections.

        The JSON structure mirrors what publish_detection emits:
          { "tile_size": int,
            "background_avg_prob": float,
            "detections": [ {"x", "y", "w", "h", "type", "class_name", "probability"}, ... ] }
        where x,y are the tile CENTRE in demosaiced (half-res) pixels, same
        contract as msg/Detection.msg.

        Thread-safe: acquires the lock to read shared pointers.
        Returns (success: bool, message: str).
        """
        with self._lock:
            frame           = self._last_frame
            responses       = self._last_responses
            tile_size       = self._last_tile_size
            frame_timestamp = self._last_frame_timestamp

        if frame is None:
            return False, "No frame available to save."

        basename  = self._make_timestamped_basename(prefix)
        png_path  = os.path.join(self._output_path, f"{basename}.png")
        json_path = os.path.join(self._output_path, f"{basename}.json")

        try:
            cv2.imwrite(png_path, frame)
        except Exception as e:
            return False, str(e)

        detections = []
        if responses is not None:
            points      = responses.get("points",      [])
            classes     = responses.get("classes",     [])
            confidences = responses.get("confidences", [])
            for (x, y), description, confidence in zip(points, classes, confidences):
                det_type, det_class = filter_type(description)
                detections.append({
                    # x,y are the tile CENTRE in demosaiced (half-res) pixels, matching
                    # Detection.msg -- NOT a top-left corner. See publish_detection.
                    "x":           int(x),
                    "y":           int(y),
                    "w":           int(tile_size),
                    "h":           int(tile_size),
                    "type":        det_type,
                    "class_name":  det_class,
                    "probability": float(confidence),
                })

        payload = {
            "timestamp_ns":         int(frame_timestamp),
            "tile_size":            int(tile_size),
            "background_avg_prob":  float(responses.get("background_avg_prob", 0.0)) if responses else 0.0,
            "detections":           detections,
        }

        try:
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            return True, f"Saved PNG: {png_path} (JSON failed: {e})"

        return True, f"Saved: {png_path} + {json_path}"

    # -------------------------
    # Thread-safe getters
    # -------------------------
    def get_step_size(self):
        """Thread-safe getter for the tile step size."""
        with self._lock:
            return self._step_size

    def get_max_probability_threshold(self):
        """Thread-safe getter for the max probability threshold."""
        with self._lock:
            return self._threshold

    def get_erosion_kernel(self):
        """Thread-safe getter for the voting neighborhood radius."""
        with self._lock:
            return self._erosion_kernel

    def get_min_votes(self):
        """Thread-safe getter for the votes required to accept a tile."""
        with self._lock:
            return self._min_votes

    def get_target_fps(self):
        """Thread-safe getter for the target FPS limit."""
        with self._lock:
            return self._target_fps

    def visualization_enabled(self):
        """Thread-safe getter for the visualization toggle."""
        with self._lock:
            return self._visualization_enabled

    def get_window_scale(self):
        """Thread-safe getter for the visualization window scale factor."""
        with self._lock:
            return self._window_scale

    def inference_paused(self):
        """Thread-safe getter for the inference pause state."""
        with self._lock:
            return self._inference_paused

    def two_stage_enabled(self):
        """Thread-safe getter for the two-stage ensemble mode."""
        with self._lock:
            return self._two_stage_enabled

    def autosave_defect_snapshots_enabled(self):
        """Thread-safe getter for the autosave defect snapshots toggle."""
        with self._lock:
            return self._autosave_defect_snapshots

    def frame_limiter_enabled(self):
        """Thread-safe getter for the frame limiter toggle."""
        with self._lock:
            return self._frame_limiter

    def majority_voting_enabled(self):
        """Thread-safe getter for the majority voting toggle."""
        with self._lock:
            return self._majority_voting

    def lasers_enabled(self):
        """Whether depth fusion has a source (standalone: --laser-depths was given)."""
        with self._lock:
            return self._use_lasers

    def get_laser_depths(self):
        """Thread-safe getter for the latest laser depth readings."""
        with self._lock:
            return list(self._laser_depths)

    def is_marker_scanning(self):
        """Check whether marker scanning is currently active."""
        with self._lock:
            return time.monotonic() < self._marker_scan_until

    # -------------------------
    # Markers
    # -------------------------
    def _get_camera_matrix(self, frame):
        """
        Return a cached approximate camera matrix for the given frame resolution.

        Caches (K, dist) per (height, width) to avoid recomputing on every frame.
        """
        h, w = frame.shape[:2]
        key = (h, w)
        if key not in self._cam_matrix_cache:
            self._cam_matrix_cache[key] = make_approx_camera_matrix(w, h)
        return self._cam_matrix_cache[key]

    def publish_marker(self, marker_id: str, tvec, rvec):
        """
        Report a detected marker with 3D position and orientation.

        The ROS node publishes this on the "markers" topic; here it is printed and
        recorded for the JSONL sink, with the same quaternion conversion.
        """
        qx, qy, qz, qw = rvec_to_quaternion(rvec)
        record = {
            "id": str(marker_id),
            "position":    {"x": float(tvec[0]), "y": float(tvec[1]), "z": float(tvec[2])},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        }
        self._frame_markers.append(record)

    def scan_and_publish_markers(self, frame):
        """
        Detect ArUco markers in the current frame and report Marker records.

        Uses the cached camera matrix for pose estimation. Chessboard detection
        is present but disabled (too slow for real-time).
        """
        self.logger.debug("Scanning frame for markers...")

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            gray = frame if len(frame.shape) == 2 else frame[:, :, 0]
        else:
            gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        K, dist = self._get_camera_matrix(frame)

        # --- ArUco ---
        self.logger.debug("Running ArUco detection...")
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is not None:
            self.logger.info(f"Found {len(ids)} ArUco marker(s): {ids.flatten().tolist()}")
            rvecs, tvecs = estimatePoseSingleMarkers(corners, DEFAULT_MARKER_LENGTH_M, K, dist)
            for marker_id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs):
                tvec_flat = tvec.flatten()
                self.logger.info(
                    f"  id={marker_id}  tvec=[{tvec_flat[0]:.3f}, {tvec_flat[1]:.3f}, {tvec_flat[2]:.3f}] m"
                )
                self.publish_marker(str(int(marker_id)), tvec, rvec)
        else:
            self.logger.info("No ArUco markers found.")

        # --- Chessboard (disabled: too slow) ---
        # See liveClassifierTorchROS.py for the original commented-out chessboard path;
        # CHESSBOARD_W/H/SQUARE_M above are kept for whoever re-enables it.
        self.logger.debug("Marker scan complete.")

    # -------------------------
    # "Publishers" (console + optional JSONL)
    # -------------------------
    def publish_detection(self, x, y, w, h, det_type, det_class, probability, depth_z=0.0, ts=0):
        """Record a Detection with 2D box, type, class, and optional depth.

        COORDINATE CONTRACT (see msg/Detection.msg): x,y are the tile CENTRE, NOT
        the top-left corner, in DEMOSAICED (half-resolution) pixels. w,h are the
        tile size, so the covered box is [x-w/2, y-h/2]..[x+w/2, y+h/2]. This
        matches responses["points"], which generate_heatmap and
        process_predictions_erode already emit as centres -- do NOT add half a
        tile again at the call site.
        """
        record = {
            "timestamp_ns": int(ts),
            "x":            int(x),
            "y":            int(y),
            "w":            int(w),
            "h":            int(h),
            "depth":        float(depth_z),
            "type":         det_type,
            "class_name":   det_class,
            "probability":  float(probability),
        }
        self._frame_detections.append(record)
        if self._verbose:
            self.logger.info(
                f"  detection {det_type}/{det_class} at ({record['x']},{record['y']}) "
                f"p={record['probability']:.3f} depth={record['depth']:.3f}")

    def publish_detection_m(self, cx, cy, severity, depth_z, ts):
        """Attach the DetectionM fields (severity + interpolated depth) to the last detection.

        cx, cy are the tile CENTRE in demosaiced (half-res) pixels -- the same
        coordinate carried by Detection.x/y. See publish_detection. In the ROS node
        this is a second message on "detections_m"; standalone there is one record
        per detection carrying both, since nothing subscribes separately.
        """
        if not self.lasers_enabled():
            return
        self._pending_m = {
            "severity": int(severity),
            "location": {"x": float(cx), "y": float(cy), "z": float(depth_z)},
            "timestamp_ns": int(ts),
        }

    def publish_background_activations(self, avg_prob, ts):
        """Record the average softmax probability of clean (non-activated) tiles."""
        self._background_probability = 1.0 - float(avg_prob)
        self._background_timestamp   = int(ts)

    def flush_frame(self, ts, tile_size, hz=0.0):
        """Emit one frame's worth of detections: a console summary + optional JSONL line."""
        detections = self._frame_detections
        markers    = self._frame_markers
        bg         = getattr(self, "_background_probability", 0.0)

        if not self._quiet:
            per_type = {}
            for d in detections:
                key = f"{d['type']}/{d['class_name']}"
                per_type[key] = per_type.get(key, 0) + 1
            breakdown = " ".join(f"{k}:{v}" for k, v in sorted(per_type.items())) or "-"
            self.logger.info(
                f"frame ts={ts} tile={tile_size} detections={len(detections)} [{breakdown}] "
                f"background_probability={bg:.4f} inference={hz:.1f} Hz")

        if self._jsonl_file is not None:
            payload = {
                "timestamp_ns":            int(ts),
                "tile_size":               int(tile_size),
                "background_probability":  float(bg),
                "detections":              detections,
            }
            if markers:
                payload["markers"] = markers
            try:
                self._jsonl_file.write(json.dumps(payload) + "\n")
                self._jsonl_file.flush()
            except Exception as e:
                self.logger.error(f"Failed to write {self._jsonl_path}: {e}")

        self._frame_detections = []
        self._frame_markers    = []

    def close(self):
        if self._jsonl_file is not None:
            try:
                self._jsonl_file.close()
            except Exception:
                pass
            self._jsonl_file = None


# ========================================================
# Keyboard command table (the standalone twin of the ROS services)
# ========================================================
KEY_HELP = """
Keys (the standalone equivalent of the ROS services):
  h / ?  this help                       q      quit
  v      toggle visualization            p      toggle pause  (set_visualization / pause)
  2      toggle two-stage ensemble       m      toggle majority voting
  f      toggle frame limiter            a      toggle autosave of defect frames
  d      remember current frame as DEFECT  (remember_defect)
  c      remember current frame as CLEAN   (remember_clean)
  s      snapshot current frame            (snapshot)
  k      scan for ArUco markers            (scan_markers)
  t / T  gate threshold -/+ 0.01         0      clear threshold override (follow model gate)
  [ / ]  step size -/+ 1                 e / E  erosion kernel -/+ 1
  n / N  min votes -/+ 1                 , / .  target FPS -/+ 1  (0 = unlimited)
  - / +  visualization window scale -/+ 0.25    =      reset window scale to 1.0
  r      hot-swap to the next model found next to this script  (set_model)
"""


def handle_key(key, runtime, model_names):
    """Apply one keystroke to the runtime. Returns False when the user asked to quit."""
    if key in ("q", "\x03", "\x1b"):
        return False

    if key in ("h", "?"):
        print(KEY_HELP, flush=True)
    elif key == "v":
        runtime.set_visualization(not runtime.visualization_enabled())
    elif key in ("p", " "):
        runtime.pause_inference(not runtime.inference_paused())
    elif key == "2":
        runtime.set_two_stage(not runtime.two_stage_enabled())
    elif key == "m":
        runtime.set_majority_voting(not runtime.majority_voting_enabled())
    elif key == "f":
        runtime.set_frame_limiter(not runtime.frame_limiter_enabled())
    elif key == "a":
        runtime.set_autosave_defect_snapshots(not runtime.autosave_defect_snapshots_enabled())
    elif key == "d":
        runtime.remember_defect()
    elif key == "c":
        runtime.remember_clean()
    elif key == "s":
        runtime.snapshot()
    elif key == "k":
        runtime.scan_markers()
    elif key in ("t", "T"):
        current = runtime.get_max_probability_threshold()
        current = 0.0 if current is None else current
        runtime.set_threshold(current + (0.01 if key == "T" else -0.01))
    elif key == "0":
        runtime.set_threshold(-1.0)
    elif key in ("[", "]"):
        runtime.set_step(runtime.get_step_size() + (1 if key == "]" else -1))
    elif key in ("e", "E"):
        runtime.set_erosion_kernel(runtime.get_erosion_kernel() + (1 if key == "E" else -1))
    elif key in ("n", "N"):
        runtime.set_min_votes(runtime.get_min_votes() + (1 if key == "N" else -1))
    elif key in (",", "."):
        runtime.set_fps(runtime.get_target_fps() + (1.0 if key == "." else -1.0))
    elif key in ("-", "_", "+"):
        runtime.set_window_scale(runtime.get_window_scale() + (0.25 if key == "+" else -0.25))
    elif key == "=":
        runtime.set_window_scale(1.0)
    elif key == "r":
        if not model_names:
            runtime.logger.warning("No other models found next to this script")
        else:
            clf  = runtime._single_classifier
            here = os.path.basename(clf.model_path)[:-4] if clf is not None else None
            i    = model_names.index(here) + 1 if here in model_names else 0
            runtime.set_model(model_names[i % len(model_names)])
    return True


def bootstrap_shared_memory_library():
    """Build/link libSharedMemoryVideoBuffers.so on first run, as this script always has."""
    if checkIfFileExists("libSharedMemoryVideoBuffers.so"):
        print("Found a shared memory video buffer library..!")
        return
    print("Bootstrapping a new shared memory video buffer library")
    os.system("git clone https://github.com/AmmarkoV/SharedMemoryVideoBuffers")
    os.system("cd SharedMemoryVideoBuffers && make && cd ..")
    os.system("ln -s SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so")
    os.system("SharedMemoryVideoBuffers/server --nokb&")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone live defect classifier (non-ROS twin of liveClassifierTorchROS.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=KEY_HELP)
    parser.add_argument("--config", default=None,
                        help="preset name from recommended_configuration.json (default: first entry)")
    parser.add_argument("--list-configs", action="store_true", help="list the presets and exit")
    parser.add_argument("--model", default=None,
                        help="override the preset's model name (auto-downloaded if missing)")
    parser.add_argument("--stream", default="stream1", help="shared memory stream name")
    parser.add_argument("--descriptor", default="video_frames.shm", help="shared memory descriptor file")

    parser.add_argument("--step", type=int, default=None, help="tile step in pixels")
    parser.add_argument("--fps", type=float, default=None, help="target FPS (0 = unlimited)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="gate threshold; negative follows the model's own calibrated gate")
    parser.add_argument("--erosion-kernel", type=int, default=None, help="voting neighborhood radius (0..5)")
    parser.add_argument("--min-votes", type=int, default=None, help="votes required to accept a tile (0/1 = off)")

    parser.add_argument("--majority-voting", dest="majority_voting", action="store_true", default=None)
    parser.add_argument("--no-majority-voting", dest="majority_voting", action="store_false")
    parser.add_argument("--frame-limiter", dest="frame_limiter", action="store_true", default=None,
                        help="skip frames whose shared memory timestamp did not change")
    parser.add_argument("--no-frame-limiter", dest="frame_limiter", action="store_false")
    parser.add_argument("--two-stage", dest="two_stage", action="store_true", default=None,
                        help="start in two-stage ensemble mode")
    parser.add_argument("--no-two-stage", dest="two_stage", action="store_false")
    parser.add_argument("--visualization", dest="visualization", action="store_true", default=None,
                        help="show the live heatmap window (default on)")
    parser.add_argument("--no-visualization", dest="visualization", action="store_false")
    parser.add_argument("--window-scale", type=float, default=None,
                        help=f"visualization window size multiplier ({WINDOW_SCALE_MIN}..{WINDOW_SCALE_MAX}); "
                             f"1.0 (default) fits the heatmap in {VISUALIZATION_MAX_W}x{VISUALIZATION_MAX_H}, "
                             f"2.0 shows that twice as big. Adjustable live with the -/+ keys")

    parser.add_argument("--autosave-defects", action="store_true",
                        help="save frame + sidecar JSON whenever a defect is detected")
    parser.add_argument("--output-path", default=None, help="directory for remembered frames (default ./data)")
    parser.add_argument("--snapshot-path", default=None, help="directory for snapshots (default ./snapshots)")
    parser.add_argument("--detections-jsonl", default=None,
                        help="append one JSON object per frame with its detections to this file")
    parser.add_argument("--laser-depths", default=None,
                        help="three fixed laser depths 'd1,d2,d3' standing in for the ROS laser topics")
    parser.add_argument("--quiet", action="store_true", help="do not print the per-frame summary")
    parser.add_argument("--verbose-detections", action="store_true", help="print every single detection")
    parser.add_argument("--debug", action="store_true", help="print debug-level messages")
    parser.add_argument("--no-keyboard", action="store_true", help="do not put the terminal in cbreak mode")
    return parser.parse_args(argv)


# ========================================================
# Main
# ========================================================
def main(argv=None):
    args = parse_arguments(argv)

    if args.list_configs:
        for p in list_recommended_configurations():
            gate = p.get("gate") or {}
            print(f"{p.get('name'):<20} model={p.get('model'):<26} "
                  f"gate={gate.get('mode')}@{gate.get('threshold')}")
            if p.get("description"):
                print(f"{'':<20} {p['description']}")
        return 0

    # Configure PyTorch global settings before any model is loaded.
    # TF32 on Ampere+ GPUs gives ~3× matmul throughput with negligible accuracy loss.
    torch.set_float32_matmul_precision("high")
    # cuDNN auto-tunes the fastest convolution algorithm for each fixed input shape.
    # Since tile_size is constant per model, this pays off after the first forward pass.
    torch.backends.cudnn.benchmark = True

    laser_depths = None
    if args.laser_depths:
        try:
            laser_depths = [float(v) for v in args.laser_depths.split(",")]
        except ValueError:
            print(f"[config] could not parse --laser-depths {args.laser_depths!r}; ignoring")

    runtime = LiveClassifier(detections_jsonl=args.detections_jsonl,
                             quiet=args.quiet,
                             verbose_detections=args.verbose_detections,
                             laser_depths=laser_depths,
                             debug=args.debug)

    PATH = os.path.dirname(os.path.abspath(__file__))

    # --config NAME selects a preset from recommended_configuration.json; default = first.
    preset = load_recommended_configuration(args.config)
    model_name = args.model or preset["model"]
    runtime.apply_preset(preset)
    runtime.apply_cli_overrides(args)

    # Fetch the model if it is not already here, then load it. ensure_model is a no-op
    # when model_scan() already sees a valid {name}.pth + {name}.json pair.
    from ModelDownload import ensure_model
    if not ensure_model(model_name, PATH):
        runtime.logger.error(
            f"Could not obtain model '{model_name}'. Check network access to the model "
            f"server, or place {model_name}.pth + {model_name}.json in {PATH}. "
            f"Edit recommended_configuration.json to use a different preset.")
        runtime.close()
        return 1

    single_classifier = ClassifierPnm(
        model_path=os.path.join(PATH, f"{model_name}.pth"),
        cfg_path=os.path.join(PATH, f"{model_name}.json"),
    )
    # The preset's gate wins over the model json's own calibration, since the preset is
    # the deployment decision. Mode too -- a threshold means nothing without its mode.
    # An explicit --model is a deliberate departure from the preset, so that model keeps
    # its own calibrated mode instead of inheriting another model's.
    if args.model is None:
        single_classifier.gateMode = preset["gate"].get("mode", single_classifier.gateMode)
        single_classifier.assignBestDefectClass = bool(
            preset["gate"].get("assign_best_defect_class", single_classifier.assignBestDefectClass))

    # Expose the classifier to the runtime for hot-swap via the 'r' key
    runtime._model_dir = PATH
    runtime._single_classifier = single_classifier

    # State the gate's expected trade-off once, up front. The runtime default is
    # deliberately stricter than the model's KPI-optimal gate to suppress false
    # alarms; this makes the cost in missed defects explicit instead of implicit.
    runtime._log_threshold_tradeoff(
        runtime.get_max_probability_threshold(),
        "Startup gate setting — ")

    # Two-stage ensemble is OPTIONAL. Build it only if every member resolves; a
    # missing member used to reach ClassifierPnm's sys.exit(1) and kill the process
    # before it reported anything, even though two-stage mode is off by default.
    ensemble_classifier = None
    _needed = [ENSEMBLE_STAGE1] + ENSEMBLE_MEMBERS
    _missing = [m for m in _needed
                if not (os.path.isfile(os.path.join(PATH, m + ".pth"))
                        and os.path.isfile(os.path.join(PATH, m + ".json")))]
    if _missing:
        runtime.logger.warning(
            f"Two-stage ensemble DISABLED — missing models: {_missing}. "
            f"Fetch them with: python3 ModelDownload.py {' '.join(_missing)}")
    else:
        try:
            from EnsembleClassifier import EnsembleClassifierPnm
            ensemble_classifier = EnsembleClassifierPnm(
                initial_model_cfg=(os.path.join(PATH, ENSEMBLE_STAGE1 + ".pth"),
                                   os.path.join(PATH, ENSEMBLE_STAGE1 + ".json")),
                model_cfg_list=[(os.path.join(PATH, m + ".pth"),
                                 os.path.join(PATH, m + ".json")) for m in ENSEMBLE_MEMBERS],
            )
        except Exception as e:
            runtime.logger.warning(f"Two-stage ensemble DISABLED — build failed: {e!r}")
            ensemble_classifier = None

    tile_size = 0  # updated inside the inference block each iteration

    # Shared memory frame source
    bootstrap_shared_memory_library()
    smm = SharedMemoryManager(
        "./libSharedMemoryVideoBuffers.so",
        descriptor=args.descriptor,
        frameName=args.stream,
        connect=True,
    )

    keyboard = KeyboardControl(enabled=not args.no_keyboard)
    model_names = sorted(ClassifierPnm.model_scan(PATH))
    if keyboard.active:
        runtime.logger.info("Keyboard control ready — press 'h' for the command list, 'q' to quit")
    else:
        runtime.logger.info("No interactive terminal — running with the command line settings")

    last_processed_timestamp = None
    last_pushed_threshold    = None   # only log a gate change, never a per-frame no-op
    _warned_no_ensemble      = False  # log the two-stage fallback once, not every frame
    _warned_channels         = False

    try:
        while True:
            loop_start = time.perf_counter()

            key = keyboard.read(window_open=runtime.visualization_enabled())
            if key is not None and not handle_key(key, runtime, model_names):
                break

            if runtime.frame_limiter_enabled():
                ts = smm.get_timestamp()
                if ts is not None and ts == last_processed_timestamp:
                    time.sleep(0.001)
                    continue

            frame          = smm.read_from_shared_memory()
            frameTimestamp = smm.unix_timestamp

            # Get image to work on
            if frame is None or smm.frame_size == 0:
                runtime.logger.warning("Couldn't read frame from Shared Memory")
                time.sleep(0.1)
                continue

            if runtime.frame_limiter_enabled() and frameTimestamp == last_processed_timestamp:
                time.sleep(0.001)
                continue
            last_processed_timestamp = frameTimestamp

            if frame.ndim != 3 or frame.shape[2] != 4:
                if not _warned_channels:
                    runtime.logger.warning(
                        f"Frame is {frame.shape}, the classifier expects HxWx4 RGBA tiles — "
                        f"check the grabber's stream format")
                    _warned_channels = True

            runtime._last_frame = frame.copy()
            runtime._last_frame_timestamp = frameTimestamp

            # Marker scanning (runs regardless of inference pause state)
            if runtime.is_marker_scanning():
                runtime.scan_and_publish_markers(frame)

            # Pause inference
            if runtime.inference_paused():
                time.sleep(0.01)
                continue

            # Run the neural network
            majority_voting = runtime.majority_voting_enabled()
            with torch.inference_mode():
                if runtime.two_stage_enabled() and ensemble_classifier is None:
                    if not _warned_no_ensemble:
                        runtime.logger.warning(
                            "two_stage requested but the ensemble is unavailable — "
                            "falling back to the single classifier")
                        _warned_no_ensemble = True

                if runtime.two_stage_enabled() and ensemble_classifier is not None:
                    with runtime._model_lock:
                        ensemble_classifier.step = runtime.get_step_size()
                        thr = runtime.get_max_probability_threshold()
                        if thr is not None and thr != ensemble_classifier.maxProbabilityThreshold:
                            ensemble_classifier.maxProbabilityThreshold = thr
                        tile_size = ensemble_classifier.tile_size

                        heatmap, occupancy, responses = ensemble_classifier.forward(
                            frame,
                            majorityVote=majority_voting,
                            parallel=True,
                            multimodel=True,
                        )
                        inference_hz = getattr(ensemble_classifier, "hz", 0.0)
                else:
                    with runtime._model_lock:
                        single_classifier.step = runtime.get_step_size()
                        # Push the override only when it CHANGED. Assigning every frame
                        # meant a model hot-swap silently lost the new model's calibrated
                        # gate (reload_model re-reads it, then the next frame overwrote
                        # it again). A None override leaves the model's own gate alone.
                        thr = runtime.get_max_probability_threshold()
                        if thr is not None and thr != single_classifier.maxProbabilityThreshold:
                            single_classifier.maxProbabilityThreshold = thr
                            if thr != last_pushed_threshold:
                                runtime._log_threshold_tradeoff(thr, "Gate threshold changed: ")
                                last_pushed_threshold = thr
                        tile_size = single_classifier.tile_size

                        heatmap, occupancy, responses = single_classifier.forward(
                            frame,
                            majorityVote=majority_voting,
                            erosion_kernel=runtime.get_erosion_kernel(),
                            erosion_threshold=runtime.get_min_votes(),
                        )
                        inference_hz = single_classifier.hz

            # Snapshot responses for _save_current_frame sidecar JSON
            runtime._last_responses = responses
            runtime._last_tile_size = tile_size

            # Publish detections
            points      = responses.get("points",      [])
            classes     = responses.get("classes",     [])
            confidences = responses.get("confidences", [])

            for (x, y), description, confidence in zip(points, classes, confidences):
                det_type, det_class = filter_type(description)

                z = 0.0
                # DetectionM with interpolated depth
                if runtime.lasers_enabled():
                    # responses["points"] already holds the tile CENTRE --
                    # generate_heatmap / process_predictions_erode append
                    # (x + tile_size//2, y + tile_size//2). Adding half a tile again
                    # here shifted every DetectionM location and every laser-depth
                    # lookup by tile_size/2 (24 px at tile_size 48).
                    cx = float(x)
                    cy = float(y)

                    depths = runtime.get_laser_depths()
                    if all(np.isfinite(d) for d in depths):
                        z = idw_depth(cx, cy, LASER_XY_PIXELS, depths, p=LASER_IDW_POWER)
                    else:
                        z = float("nan")

                    severity = class_to_severity(det_class)
                    #Detection_M accepts severities 1,2,3
                    runtime.publish_detection_m(cx, cy, severity, z, frameTimestamp)

                # Existing 2D detection
                runtime.publish_detection(
                    x=x,
                    y=y,
                    w=tile_size,
                    h=tile_size,
                    det_type=det_type,
                    det_class=det_class,
                    probability=confidence,
                    depth_z = z,
                    ts = frameTimestamp
                )

            # Autosave one snapshot per frame whenever a defect is detected
            if runtime.autosave_defect_snapshots_enabled() and points:
                runtime._save_current_frame("autosaved_defect")

            # Publish average background (clean-tile) softmax probability
            runtime.publish_background_activations(
                responses.get("background_avg_prob", 0.0),
                frameTimestamp,
            )

            runtime.flush_frame(frameTimestamp, tile_size, inference_hz)

            # Visualization
            if runtime.visualization_enabled():
                heatmapForAWindow,scale = resize_to_fit_screen(heatmap)
                # --window-scale (keys -/+) multiplies the window we would normally
                # show, so 2.0 is exactly twice the default size whatever the heatmap
                # resolution is -- deliberately AFTER the fit, so a 4K heatmap is
                # still shrunk to the fit box first instead of being blown past it.
                heatmapForAWindow = scale_window_image(heatmapForAWindow,
                                                       runtime.get_window_scale())
                cv2.imshow("Classifier Output",heatmapForAWindow)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            # Target FPS limiting
            target_fps = runtime.get_target_fps()
            if target_fps > 0.0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, (1.0 / target_fps) - elapsed)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        runtime.logger.info("Interrupted by user.")

    finally:
        keyboard.restore()
        runtime.close()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
