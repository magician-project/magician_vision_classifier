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

from mvc.inference.classifier_pnm import *   # noqa: F401,F403 -- re-export the classifier core
from mvc.core.shared_memory import SharedMemoryManager
from mvc.paths import repo_root

import os
import sys
import json
import time
import math
import shutil
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
# The model's .pth/.json are auto-fetched on first run (model_download.ensure_model).
#
# RECOMMENDED_CONFIG_FILE, FALLBACK_PRESET and load_recommended_configuration come from
# classifierPnm through the star import above, so the ROS node, wxAnnotator and this
# runner share ONE definition. Only `--list-configs` needs the raw preset list, which no
# other consumer wants, so that helper stays here.


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
        # Ordered N-stage screen-then-recheck chain (CascadeClassifierPnm) -- a DIFFERENT
        # mechanism from the two-stage ensemble above (which reuses stage 1's tiles verbatim
        # and majority-votes with no per-member threshold; see mvc/inference/
        # ensemble_classifier.py). Populated from the preset's "cascade" block, never from
        # per-frame keystrokes: a cascade's value is a jointly-tuned set of per-stage
        # step/threshold pairs, not a knob an operator free-runs one stage at a time --
        # retuning means switching presets (or restarting with a different one), same as a
        # model hot-swap.
        self._cascade_enabled = False
        self._cascade_stages = []
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
        # What the terminal view ('V') redraws: the heatmap's resolution, which is the
        # space detection x,y live in, and the same stats dict the status line prints.
        self._last_view_shape  = None
        self._last_infer_stats = None

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
        cascade = preset.get("cascade") or {}
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
            self._cascade_enabled = bool(cascade.get("enabled", False))
            self._cascade_stages  = list(cascade.get("stages") or [])
        m = preset.get("measured") or {}
        self.logger.info(
            f"Preset '{preset.get('name','?')}': model={preset.get('model')} "
            f"gate={gate.get('mode')}@{gate.get('threshold')} step={self._step_size} "
            f"fps={self._target_fps} erosion_kernel={self._erosion_kernel} "
            f"min_votes={self._min_votes} majority_voting={self._majority_voting}")
        if self._cascade_enabled:
            chain = " -> ".join(f"{s.get('model')}(step={s.get('step')}, "
                                f"thr={(s.get('gate') or {}).get('threshold')})"
                                for s in self._cascade_stages)
            self.logger.info(f"  cascade ENABLED: {chain}")
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
            if args.cascade is not None:          self._cascade_enabled = bool(args.cascade)
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

    def set_cascade(self, enabled):
        """Toggle cascade mode on/off. Only takes effect if the active preset defined a
        stage list at startup (this does not build a cascade out of thin air) -- see
        CascadeClassifierPnm."""
        with self._lock:
            self._cascade_enabled = bool(enabled)
        self.logger.info("Cascade execution ENABLED" if enabled else "Cascade execution DISABLED")

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

    def cascade_enabled(self):
        """Thread-safe getter for the cascade mode."""
        with self._lock:
            return self._cascade_enabled

    def get_cascade_stages(self):
        """Thread-safe getter for the active preset's ordered stage list (list of dicts,
        each at least {'model', 'step', 'gate': {'mode', 'threshold', ...}})."""
        with self._lock:
            return list(self._cascade_stages)

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

    @staticmethod
    def _compact(n):
        """4588 -> '4.5K'. Keeps the status line inside one terminal row."""
        n = float(n)
        if n >= 1000.0:
            return f"{n / 1000.0:.1f}K"
        return f"{n:.0f}"

    def flush_frame(self, ts, tile_size, hz=0.0, infer_stats=None):
        """Emit one frame's worth of detections: a console summary + optional JSONL line."""
        detections = self._frame_detections
        markers    = self._frame_markers
        bg         = getattr(self, "_background_probability", 0.0)

        if not self._quiet:
            per_type = {}
            for d in detections:
                # "PositiveDent/ClassA" -> "Pos/A". Generic (prefix + declassed suffix)
                # rather than a lookup table, so a newly trained class shortens too.
                key = f"{d['type'][:3]}/{d['class_name'].replace('Class', '')}"
                per_type[key] = per_type.get(key, 0) + 1
            # Busiest classes first, and only the top few: a frame can activate all seven
            # at once, and the full list alone ran past 150 characters and wrapped, which
            # cost the colour its whole at-a-glance value.
            ranked    = sorted(per_type.items(), key=lambda kv: -kv[1])
            breakdown = " ".join(f"{k}:{v}" for k, v in ranked[:3])
            if len(ranked) > 3:
                breakdown += f" +{len(ranked) - 3}"
            if breakdown:
                breakdown = " " + breakdown
            # Colour the whole line by verdict. At ~18 frames/sec this scrolls faster than
            # anyone can read a count, so the operator needs a signal they can catch
            # peripherally: solid green means this frame activated nothing, red means it did.
            # Same raw-ANSI convention runSingle()'s timing line already uses.
            if detections:
                colour, verdict = bcolors.FAIL, f"DEFECT x{len(detections)}"
            else:
                colour, verdict = bcolors.OKGREEN, "CLEAN"

            st    = infer_stats or {}
            model = str(st.get("name", "model")).replace(".pth", "")
            # None means no runtime override, i.e. the model's own calibrated gate is in
            # force -- worth saying rather than printing a number that is not the one used.
            thr   = self.get_max_probability_threshold()
            thr_s = f"{thr:.3f}" if thr is not None else "model"
            line  = (f"{model} | {verdict} | bg={bg:.3f} | T={thr_s} | "
                     f"step={st.get('step', 0)} | "
                     f"@ {st.get('hz', hz):5.2f} Hz | "
                     f"({self._compact(st.get('tiles', 0))} tiles, "
                     f"{self._compact(st.get('tiles_per_sec', 0))}tile/s){breakdown}")

            # On a terminal this is ONE line that rewrites itself, so a 19 Hz stream stops
            # scrolling everything else off screen. "\x1b[K" erases whatever the previous,
            # possibly longer, frame left to the right of the cursor -- padding to a fixed
            # width would do the same until the line outgrew a narrow terminal, and a
            # wrapped line breaks '\r' (it only returns to the start of the last screen row).
            # Redirected output keeps real newlines -- '\r' would collapse a whole run into
            # a single unreadable, ungreppable line in the log file.
            if sys.stdout.isatty():
                # Truncate to the real terminal width. A busy frame's class breakdown pushes
                # this past 140 columns, and a line that wraps defeats '\r' entirely -- the
                # carriage return only rewinds to the start of the last screen row, so the
                # earlier rows stay and the display scrolls after all. len() is the visible
                # width here because `line` is assembled without any escape codes.
                width = shutil.get_terminal_size(fallback=(120, 24)).columns
                print(f"{colour}{bcolors.BOLD}{line[:width - 1]}{bcolors.ENDC}\x1b[K",
                      end="\r", flush=True)
                self._status_line_open = True
            else:
                print(f"{colour}{bcolors.BOLD}{line}{bcolors.ENDC}", flush=True)

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

    def end_status_line(self):
        """Finish the rewriting one-line status so a block can be printed under it.

        The status line is left without a newline so the next frame can overwrite it.
        Anything multi-line printed on top of it -- the key help, the terminal view,
        the shutdown messages, the shell prompt -- lands on that half-written row
        instead, so every such caller closes it off first.
        """
        if getattr(self, "_status_line_open", False):
            print(flush=True)
            self._status_line_open = False

    def close(self):
        self.end_status_line()
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
  V      draw the visualization window in the terminal (works with no X display)
  v      toggle visualization            p      toggle pause  (set_visualization / pause)
  2      toggle two-stage ensemble       m      toggle majority voting
  3      toggle cascade (needs a preset with a "cascade" stage list; see recommended_configuration.json)
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


# ========================================================
# Terminal ("ASCII") view of the visualization window
# ========================================================
# The OpenCV window needs an X display a deployment box reached over ssh does not
# have, and 'v' switches it off anyway whenever the frame time it costs is wanted
# back for inference. This redraws what that window shows -- WHERE on the frame the
# classifier activated, and in which class colour -- as one screenful of text, so
# the operator keeps the spatial shape of a frame instead of only the running status
# line's counts. It is printed on demand ('V'), not per frame: at ~19 Hz a block
# this size would be unreadable and would scroll everything else off screen.

VIEW_CELL_ASPECT = 2.0    # a terminal cell is roughly twice as tall as it is wide
VIEW_MAX_COLS    = 110
VIEW_MIN_COLS    = 24
VIEW_CLASS_CHARS = "abcdefghijklmnopqrstuvwxyz"
VIEW_DIM         = "\033[2m"


def ansi_fg(colour):
    """One class colour from the window, as a 24-bit ANSI foreground escape.

    class_colors entries are written straight into the heatmap buffer, which
    generate_heatmap has already converted to BGR for cv2.imshow -- so a tuple's
    FIRST channel is the blue the operator actually sees. Reversing it here is what
    makes a class the same colour in this view as it is in the window; taking the
    tuples at their (R,G,B) word would paint every class a different colour from
    the one it has on screen, which defeats the point of colouring them at all.
    """
    b, g, r = (int(max(0, min(255, round(c)))) for c in colour[:3])
    return f"\033[38;2;{r};{g};{b}m"


def view_glyphs():
    """Box-drawing characters, or ASCII stand-ins on a terminal that cannot encode them."""
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" in enc:
        return {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
                "h": "─", "v": "│", "empty": "·"}
    return {"tl": "+", "tr": "+", "bl": "+", "br": "+",
            "h": "-", "v": "|", "empty": "."}


def view_geometry(runtime):
    """(height, width) of the coordinate space detections are reported in.

    Detection x,y are demosaiced (half-res) pixels -- i.e. the heatmap's own
    resolution, which is why the last heatmap's shape is what the character grid is
    scaled against. Before the first inference there is no heatmap, so a raw mosaic
    frame is halved by hand to land in the same space.
    """
    shape = getattr(runtime, "_last_view_shape", None)
    if shape:
        return int(shape[0]), int(shape[1])
    frame = getattr(runtime, "_last_frame", None)
    if frame is None:
        return 0, 0
    h, w = frame.shape[:2]
    if frame.ndim == 3 and frame.shape[2] == 4:
        return h, w
    return h // 2, w // 2


def wrap_segments(prefix, segments, width):
    """Lay out (plain, painted) segments into lines no wider than `width`.

    The painted twin carries ANSI escapes, so its len() is not its printed width;
    the plain twin is what the budget is measured against. Continuation lines are
    indented to the prefix so the labels stay in one column.
    """
    pad, lines = " " * len(prefix), []
    cur_plain, cur_painted, first = prefix, prefix, True
    for plain, painted in segments:
        sep = "" if first else "  "
        if not first and len(cur_plain) + len(sep) + len(plain) > width:
            lines.append(cur_painted)
            cur_plain, cur_painted, first, sep = pad, pad, True, ""
        cur_plain   += sep + plain
        cur_painted += sep + painted
        first = False
    if not first:
        lines.append(cur_painted)
    return lines


def render_ascii_view(runtime):
    """Draw the visualization window as a block of coloured text. Returns the block."""
    g       = view_glyphs()
    term    = shutil.get_terminal_size(fallback=(120, 24))
    # Redirected output gets the same drawing without escapes: a log file keeps a
    # readable map, rather than a map wrapped in codes nothing will interpret.
    colour  = sys.stdout.isatty()
    width   = max(40, term.columns - 1)

    def paint(text, *codes):
        return f"{''.join(codes)}{text}{bcolors.ENDC}" if colour and codes else text

    h_px, w_px = view_geometry(runtime)
    if w_px <= 0 or h_px <= 0:
        return paint("No frame has been classified yet - nothing to draw.", bcolors.WARNING)

    # One consistent snapshot of the last frame's results, taken the way
    # _save_current_frame takes it. No getter may be called while the lock is held.
    with runtime._lock:
        responses  = runtime._last_responses
        tile_size  = runtime._last_tile_size
        stats      = dict(getattr(runtime, "_last_infer_stats", None) or {})
    points = list(responses.get("points",      [])) if responses else []
    ids    = list(responses.get("classIDs",    [])) if responses else []
    names  = list(responses.get("classes",     [])) if responses else []
    confs  = list(responses.get("confidences", [])) if responses else []

    # Keep the drawing in proportion with the frame: a 16:9 frame has to look 16:9
    # here too, or a defect's position is read off the wrong part of the plate. So the
    # terminal's height is spent by NARROWING the map, not by squashing it -- 12 lines
    # are left under it for the summary, the legend and the state.
    cols = max(VIEW_MIN_COLS, min(VIEW_MAX_COLS, term.columns - 4))
    aspect   = (h_px / w_px) / VIEW_CELL_ASPECT
    max_rows = max(3, term.lines - 12)
    rows     = int(round(cols * aspect))
    if rows > max_rows:
        rows = max_rows
        cols = max(VIEW_MIN_COLS, int(round(rows / aspect)))

    # One cell holds however many tiles fall inside it -- at step 8 on a 960px-wide
    # frame that is about a dozen -- so the most confident activation is what it shows.
    cells = {}
    for i, (x, y) in enumerate(points):
        c = min(cols - 1, max(0, int(float(x) * cols / w_px)))
        r = min(rows - 1, max(0, int(float(y) * rows / h_px)))
        cells.setdefault((r, c), []).append(
            (float(confs[i]) if i < len(confs) else 0.0,
             int(ids[i])     if i < len(ids)   else -1))

    palette = list(getattr(runtime._single_classifier, "class_colors", []) or [])

    def class_style(cid):
        return ansi_fg(palette[cid]) if colour and 0 <= cid < len(palette) else ""

    def class_char(cid):
        return VIEW_CLASS_CHARS[cid % len(VIEW_CLASS_CHARS)] if cid >= 0 else "?"

    crowded = False
    body    = []
    for r in range(rows):
        parts = [VIEW_DIM] if colour else []
        for c in range(cols):
            hits = cells.get((r, c))
            if not hits:
                parts.append(g["empty"])
                continue
            _, cid = max(hits)
            char   = class_char(cid)
            if len(hits) > 1:
                char    = char.upper()
                crowded = True
            if colour:
                parts.append(f"{bcolors.ENDC}{class_style(cid)}{bcolors.BOLD}{char}"
                             f"{bcolors.ENDC}{VIEW_DIM}")
            else:
                parts.append(char)
        if colour:
            parts.append(bcolors.ENDC)
        body.append("".join(parts))

    # ---- frame -------------------------------------------------------------
    model  = str(stats.get("name", "model")).replace(".pth", "")
    header = (f"{g['h']} {model}  {w_px}x{h_px}px  tile {tile_size}  "
              f"step {stats.get('step', runtime.get_step_size())} ")[:cols]
    out = [paint(g["tl"] + header + g["h"] * (cols - len(header)) + g["tr"], VIEW_DIM),
           *[paint(g["v"], VIEW_DIM) + line + paint(g["v"], VIEW_DIM) for line in body],
           paint(g["bl"] + g["h"] * cols + g["br"], VIEW_DIM)]

    # ---- summary, same numbers and the same verdict colour as the status line ----
    thr   = runtime.get_max_probability_threshold()
    thr_s = f"{thr:.3f}" if thr is not None else "model"
    bg    = getattr(runtime, "_background_probability", 0.0)
    if points:
        verdict, tint = f"DEFECT x{len(points)}", bcolors.FAIL
    else:
        verdict, tint = "CLEAN", bcolors.OKGREEN
    summary = [(verdict, paint(verdict, tint, bcolors.BOLD)),
               *[(s, s) for s in (
                   f"bg={bg:.3f}",
                   f"T={thr_s}",
                   f"@ {stats.get('hz', 0.0):.2f} Hz",
                   f"{LiveClassifier._compact(stats.get('tiles', 0))} tiles",
                   f"{LiveClassifier._compact(stats.get('tiles_per_sec', 0))} tile/s")]]
    out += wrap_segments("  ", summary, width)

    # ---- legend ------------------------------------------------------------
    per_class = {}
    for i in range(len(points)):
        cid   = int(ids[i]) if i < len(ids) else -1
        entry = per_class.setdefault(cid, {"name": names[i] if i < len(names) else "?",
                                           "count": 0})
        entry["count"] += 1
    if per_class:
        legend = []
        for cid, e in sorted(per_class.items(), key=lambda kv: -kv[1]["count"]):
            det_type, det_class = filter_type(e["name"])
            char  = class_char(cid)
            plain = f"{char} {det_type}/{det_class} x{e['count']}"
            legend.append((plain, f"{class_style(cid)}{bcolors.BOLD}{char}{bcolors.ENDC} "
                                  f"{det_type}/{det_class} x{e['count']}" if colour else plain))
        out += wrap_segments("  classes ", legend, width)
        if crowded:
            note = "(a CAPITAL letter = 2+ activations in one cell)"
            out.append(paint(" " * 10 + note[:width - 10], VIEW_DIM))
    else:
        out.append(paint("  classes  none activated this frame", VIEW_DIM))

    # ---- the state the keys change -----------------------------------------
    def flag(label, on):
        text = f"{label}={'ON' if on else 'off'}"
        if not colour:
            return (text, text)
        tint = bcolors.OKGREEN if on else VIEW_DIM
        return (text, f"{label}={tint}{'ON' if on else 'off'}{bcolors.ENDC}")

    out += wrap_segments("  toggles ", [
        flag("window",   runtime.visualization_enabled()),
        flag("paused",   runtime.inference_paused()),
        flag("2-stage",  runtime.two_stage_enabled()),
        flag("cascade",  runtime.cascade_enabled()),
        flag("voting",   runtime.majority_voting_enabled()),
        flag("limiter",  runtime.frame_limiter_enabled()),
        flag("autosave", runtime.autosave_defect_snapshots_enabled()),
        flag("lasers",   runtime.lasers_enabled()),
    ], width)

    fps   = runtime.get_target_fps()
    knobs = [f"thr={thr_s}", f"step={runtime.get_step_size()}",
             f"erosion={runtime.get_erosion_kernel()}", f"votes={runtime.get_min_votes()}",
             f"fps={'unlimited' if fps <= 0.0 else f'{fps:g}'}",
             f"scale={runtime.get_window_scale():.2f}"]
    out += wrap_segments("  knobs   ", [(k, k) for k in knobs], width)
    out += wrap_segments("  keys    ",
                         [(k, paint(k, VIEW_DIM)) for k in
                          ("V redraw", "h all keys", "v window", "p pause", "s snapshot", "q quit")],
                         width)
    return "\n".join(out)


def handle_key(key, runtime, model_names):
    """Apply one keystroke to the runtime. Returns False when the user asked to quit."""
    if key in ("q", "\x03", "\x1b"):
        return False

    if key in ("h", "?"):
        runtime.end_status_line()
        print(KEY_HELP, flush=True)
    elif key == "V":
        runtime.end_status_line()
        print(render_ascii_view(runtime), flush=True)
    elif key == "v":
        runtime.set_visualization(not runtime.visualization_enabled())
    elif key in ("p", " "):
        runtime.pause_inference(not runtime.inference_paused())
    elif key == "2":
        runtime.set_two_stage(not runtime.two_stage_enabled())
    elif key == "3":
        runtime.set_cascade(not runtime.cascade_enabled())
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


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone live defect classifier (non-ROS twin of liveClassifierTorchROS.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=KEY_HELP)
    parser.add_argument("--config", default=None,
                        help="preset name from recommended_configuration.json (default: first entry)")
    parser.add_argument("--list-configs", action="store_true", help="list the presets and exit")
    parser.add_argument("--model", default=None,
                        help="override the preset's model name (auto-downloaded if missing; "
                             "an engine/<name>.py plug-in runs when the name matches)")
    parser.add_argument("--model-config", default=None,
                        help="path to the engine plug-in's .json config "
                             "(default engine/<model>.json; only used when --model is an engine)")
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
    parser.add_argument("--cascade", dest="cascade", action="store_true", default=None,
                        help="start in cascade mode (needs the active preset's \"cascade\" "
                             "stage list -- see recommended_configuration.json)")
    parser.add_argument("--no-cascade", dest="cascade", action="store_false")
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
    parser.add_argument("--perf-log", action="store_true", help="append every frame's inference timing to perf.csv (off by default -- the file grows unbounded, ~700k lines/day of continuous use, for a per-frame write cost that profiling shows is negligible either way)")
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

    PATH = repo_root()

    # --config NAME selects a preset from recommended_configuration.json; default = first.
    preset = load_recommended_configuration(args.config)
    model_name = args.model or preset["model"]
    runtime.apply_preset(preset)
    runtime.apply_cli_overrides(args)

    # An engine plug-in (engine/<name>.py, a third-party classifier design) has no
    # .pth on the model server, so it skips ensure_model and is built from its own
    # .json config (default engine/<name>.json, override with --model-config).
    from mvc.inference.engine_base import is_engine_name, build_engine
    if is_engine_name(model_name):
        try:
            single_classifier = build_engine(model_name, cfg_path=args.model_config)
        except Exception as e:
            runtime.logger.error(f"Could not build engine '{model_name}': {e}")
            runtime.close()
            return 1
    else:
        if args.model_config:
            runtime.logger.warning("--model-config ignored — only engine plug-ins take a config path")
        # Fetch the model if it is not already here, then load it. ensure_model is a no-op
        # when model_scan() already sees a valid {name}.pth + {name}.json pair.
        from mvc.inference.model_download import ensure_model
        if not ensure_model(model_name, PATH):
            runtime.logger.error(
                f"Could not obtain model '{model_name}'. Check network access to the model "
                f"server, or place {model_name}.pth + {model_name}.json in {PATH}. "
                f"Edit recommended_configuration.json to use a different preset.")
            runtime.close()
            return 1

        # Only this path gets inductor's autotuner: it runs one constant batch (the whole
        # tile grid) every frame, so the one-off compile amortises. The ensemble members
        # deliberately do NOT -- see ClassifierPnm._load_model()'s compile_mode comment.
        single_classifier = ClassifierPnm(
            model_path=os.path.join(PATH, f"{model_name}.pth"),
            cfg_path=os.path.join(PATH, f"{model_name}.json"),
            compile_mode="max-autotune-no-cudagraphs",
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
            f"Fetch them with: python3 -m mvc.inference.model_download {' '.join(_missing)}")
    else:
        try:
            from mvc.inference.ensemble_classifier import EnsembleClassifierPnm
            ensemble_classifier = EnsembleClassifierPnm(
                initial_model_cfg=(os.path.join(PATH, ENSEMBLE_STAGE1 + ".pth"),
                                   os.path.join(PATH, ENSEMBLE_STAGE1 + ".json")),
                model_cfg_list=[(os.path.join(PATH, m + ".pth"),
                                 os.path.join(PATH, m + ".json")) for m in ENSEMBLE_MEMBERS],
            )
        except Exception as e:
            runtime.logger.warning(f"Two-stage ensemble DISABLED — build failed: {e!r}")
            ensemble_classifier = None

    # Cascade is OPTIONAL, same non-fatal-missing-model pattern as the ensemble above.
    # Built only if the active preset actually requested it (runtime.cascade_enabled())
    # -- an unpopulated "cascade" block (the common case) means no stages to resolve, no
    # download attempt, no warning.
    cascade_classifier = None
    _cascade_stages_cfg = runtime.get_cascade_stages()
    if runtime.cascade_enabled():
        if len(_cascade_stages_cfg) < 2:
            runtime.logger.warning(
                f"Cascade requested but the active preset defines "
                f"{len(_cascade_stages_cfg)} stage(s) (need >= 2) — cascade DISABLED")
        else:
            _cascade_needed = [s["model"] for s in _cascade_stages_cfg]
            _cascade_missing = [m for m in _cascade_needed
                                if not (os.path.isfile(os.path.join(PATH, m + ".pth"))
                                        and os.path.isfile(os.path.join(PATH, m + ".json")))]
            if _cascade_missing:
                runtime.logger.warning(
                    f"Cascade DISABLED — missing models: {_cascade_missing}. "
                    f"Fetch them with: python3 -m mvc.inference.model_download "
                    f"{' '.join(_cascade_missing)}")
            else:
                try:
                    from mvc.inference.ensemble_classifier import CascadeClassifierPnm
                    cascade_classifier = CascadeClassifierPnm(stage_cfgs=[
                        dict(model_path=os.path.join(PATH, s["model"] + ".pth"),
                             cfg_path=os.path.join(PATH, s["model"] + ".json"),
                             step=int(s.get("step", 16)),
                             threshold=float((s.get("gate") or {}).get("threshold", 0.0)),
                             gate_mode=(s.get("gate") or {}).get("mode", GATE_DEFECT_MASS),
                             assign_best_defect_class=bool(
                                 (s.get("gate") or {}).get("assign_best_defect_class", True)))
                        for s in _cascade_stages_cfg
                    ])
                    runtime.logger.info(
                        "Cascade built: " + " -> ".join(
                            f"{s['model']}(step={s.get('step', 16)}, "
                            f"thr={(s.get('gate') or {}).get('threshold', 0.0)})"
                            for s in _cascade_stages_cfg))
                except Exception as e:
                    runtime.logger.warning(f"Cascade DISABLED — build failed: {e!r}")
                    cascade_classifier = None

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

    perf_log                 = args.perf_log   # append each frame's inference timing to perf.csv (opt-in)
    last_processed_timestamp = None
    last_pushed_threshold    = None   # only log a gate change, never a per-frame no-op
    _warned_no_ensemble      = False  # log the two-stage fallback once, not every frame
    _warned_no_cascade       = False  # log the cascade fallback once, not every frame
    _warned_channels         = False
    _warned_no_frame         = False  # log a missing frame once per outage, not every 100 ms

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
                if not _warned_no_frame:
                    runtime.logger.warning("Couldn't read frame from Shared Memory")
                    _warned_no_frame = True
                time.sleep(0.1)
                continue
            _warned_no_frame = False   # warn again the next time frames stop

            if runtime.frame_limiter_enabled() and frameTimestamp == last_processed_timestamp:
                time.sleep(0.001)
                continue
            last_processed_timestamp = frameTimestamp

            frame_is_raw_mosaic = frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1)
            frame_is_demosaiced = frame.ndim == 3 and frame.shape[2] == 4
            if not (frame_is_raw_mosaic or frame_is_demosaiced):
                if not _warned_channels:
                    runtime.logger.warning(
                        f"Frame is {frame.shape}, expected either a raw single-channel "
                        f"polarization mosaic (auto-debayered downstream) or an already "
                        f"HxWx4 RGBA array — check the grabber's stream format")
                    _warned_channels = True

            # read_from_shared_memory() already returned a private copy, and nothing below writes to it
            runtime._last_frame = frame
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
                if runtime.cascade_enabled() and cascade_classifier is None:
                    if not _warned_no_cascade:
                        runtime.logger.warning(
                            "cascade requested but unavailable (see the DISABLED warning "
                            "at startup) — falling back to two-stage/single classifier")
                        _warned_no_cascade = True

                if runtime.two_stage_enabled() and ensemble_classifier is None:
                    if not _warned_no_ensemble:
                        runtime.logger.warning(
                            "two_stage requested but the ensemble is unavailable — "
                            "falling back to the single classifier")
                        _warned_no_ensemble = True

                infer_stats = {}   # filled by whichever branch runs; drives the status line
                if runtime.cascade_enabled() and cascade_classifier is not None:
                    # No per-frame step/threshold push here, unlike the two branches below:
                    # a cascade's whole point is a jointly-tuned set of per-stage step/
                    # threshold pairs (chosen offline, see knowledge/7-9-report.md §8.4-
                    # §8.7), not a single knob an operator free-runs live. Retuning means
                    # switching presets, same as a model hot-swap.
                    with runtime._model_lock:
                        tile_size = cascade_classifier.stages[-1].tile_size
                        heatmap, occupancy, responses = cascade_classifier.forward(
                            frame, legend=True, log=perf_log)
                        inference_hz = getattr(cascade_classifier, "hz", 0.0)
                        infer_stats = {"name": "cascade", "step": cascade_classifier.stages[-1].step,
                                       "hz": inference_hz,
                                       "tiles": getattr(cascade_classifier, "_last_tile_count", 0),
                                       "tiles_per_sec": getattr(cascade_classifier, "_last_tile_count", 0)
                                                        / max(getattr(cascade_classifier, "_last_elapsed", 1e-4), 1e-4)}
                elif runtime.two_stage_enabled() and ensemble_classifier is not None:
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
                            log=perf_log,
                        )
                        inference_hz = getattr(ensemble_classifier, "hz", 0.0)
                        infer_stats = {"name": "ensemble", "step": ensemble_classifier.step,
                                       "hz": inference_hz,
                                       "tiles": getattr(ensemble_classifier, "_last_tile_count", 0),
                                       "tiles_per_sec": getattr(ensemble_classifier, "_last_tile_count", 0)
                                                        / max(getattr(ensemble_classifier, "_last_elapsed", 1e-4), 1e-4)}
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
                            log=perf_log,
                            stats=infer_stats,
                        )
                        inference_hz = single_classifier.hz

            # Snapshot responses for _save_current_frame sidecar JSON
            runtime._last_responses = responses
            runtime._last_tile_size = tile_size
            # The heatmap is the window's own image, so its shape is the coordinate
            # space responses["points"] are in -- what the terminal view scales against.
            runtime._last_view_shape  = heatmap.shape[:2] if heatmap is not None else None
            runtime._last_infer_stats = dict(infer_stats)

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

            runtime.flush_frame(frameTimestamp, tile_size, inference_hz, infer_stats)

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
