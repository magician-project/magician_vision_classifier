#!/usr/bin/python3
"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

# --------------------------------------------------------
import os
import sys
import time
import json
import threading
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F

# --------------------------------------------------------
# Everything that is not ROS-specific lives in liveClassifierTorch.py, which is also
# the standalone runner. Importing it here (rather than keeping a second copy) is what
# keeps the ROS node and the standalone node at feature parity: presets, gate/step/vote
# knobs, laser geometry, marker maths and the detection contract have ONE definition.
from mvc.inference.live_torch import (
    ClassifierPnm,
    ENSEMBLE_STAGE1, ENSEMBLE_MEMBERS,
    LASER_XY_PIXELS, LASER_IDW_POWER, idw_depth,
    MARKER_SCAN_DURATION_S, ARUCO_DICT_NAME, DEFAULT_MARKER_LENGTH_M,
    CHESSBOARD_W, CHESSBOARD_H, CHESSBOARD_SQUARE_M,
    estimatePoseSingleMarkers, make_approx_camera_matrix, rvec_to_quaternion,
    resize_to_fit_screen, filter_type, class_to_severity,
)
from mvc.inference.ensemble_classifier import EnsembleClassifierPnm
from mvc.core.shared_memory import SharedMemoryManager
from mvc.paths import repo_root

# --------------------------------------------------------
import rclpy
from rclpy.node import Node

#To make a venv with ROS plus the needed pytorch stuff
#sudo apt install ros-rolling-example-interfaces
#source source /opt/ros/rolling/setup.bash
#python3 -m venv venv --system-site-packages
#source venv/bin/activate
#python3 -m pip install -r requirements.txt 
#python3 -m pip install empy lark

from std_srvs.srv import SetBool
from std_srvs.srv import Trigger
from datetime import datetime

#from example_interfaces.srv import SetInt64
#from example_interfaces.srv import SetFloat64
from magician_vision_classifier.srv import SetInt64
from magician_vision_classifier.srv import SetFloat64
#from magician_vision_classifier.srv import SetString


from std_msgs.msg import Float32, Header
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time as RosTime

from magician_vision_classifier.msg import BackgroundActivations # Custom ROS message
from magician_vision_classifier.msg import Detection             # Custom ROS message
from magician_vision_classifier.msg import DetectionM            # Custom ROS message (uint8 severity, geometry_msgs/Pose location)
from magician_vision_classifier.msg import Marker                # Custom ROS message (string id, geometry_msgs/Pose pose)

def unix_ns_to_ros_time(ns):
    t = RosTime()
    t.sec = int(ns // 1_000_000_000)
    t.nanosec = int(ns % 1_000_000_000)
    return t


# ========================================================
# Deployment presets
# ========================================================
# Which model to run and at what operating point comes from recommended_configuration.json
# (RECOMMENDED_CONFIG_FILE / load_recommended_configuration, imported below from
# classifierPnm). That file is COMMITTED TO GIT, so a deployment site picks up new
# models and thresholds with a plain `git pull` -- deliberately NOT environment variables,
# which are awkward to change on-site.
#
# The FIRST entry is the startup default; pass `--config NAME` to select another.
# The model's .pth/.json are auto-fetched on first run (model_download.ensure_model).

# ENSEMBLE_STAGE1 / ENSEMBLE_MEMBERS name the OPTIONAL two-stage ensemble; if any member
# cannot be resolved the ensemble is skipped and the node still starts on the single
# classifier. 
# The loader lives in classifierPnm so the ROS node, wxAnnotator and any other consumer
# share ONE definition rather than hand-copied variants that drift apart.
from mvc.inference.classifier_pnm import (load_recommended_configuration, FALLBACK_PRESET,
                           RECOMMENDED_CONFIG_FILE)


# Two-stage ensemble members. This path is OPTIONAL: if any member cannot be resolved the
# ensemble is skipped and the node still starts on the single classifier (previously a
# missing member called sys.exit(1) inside ClassifierPnm and killed the node before it
# ever published).
ENSEMBLE_STAGE1 = "binary_small_cnn"
ENSEMBLE_MEMBERS = [
    "allclass_verysmall_cnn",
    "allclass_resnet18",
    "allclass_resnext50",
    "allclass_convnext_tiny",
]


# ========================================================
# Laser fusion globals (project-specific / fixed hardware)
# ========================================================
USE_LASERS = True  # Set to False to disable subscriptions + DetectionM publishing entirely.

LASER_TOPICS = [
    "magician_grabber/distance1",
    "magician_grabber/distance2",
    "magician_grabber/distance3",
]

# LASER_XY_PIXELS (the laser positions in the classifier's 2D image plane) and
# LASER_IDW_POWER are imported from liveClassifierTorch so both runners fuse depth
# identically; only the ROS topics that feed them are node-specific.


# ========================================================
# ROS Node
# ========================================================
class DefectPublisher(Node):
    """
    ROS2 node for real-time defect detection and classification.

    Subscribes to laser depth sensors (optional), publishes Detection messages
    for each detected defect, and optionally publishes DetectionM messages with
    interpolated depth. Exposes runtime tuning services via ROS2 services
    (visualization, pause, FPS, step size, threshold control).

    Supports both single-model and two-stage ensemble inference.
    """

    def __init__(self):
        """Initialize the ROS2 node, publishers, services, and laser subscriptions."""
        super().__init__("magician_vision_classifier")

        # Publishers
        self.publisher_ = self.create_publisher(Detection, "detections", 10)

        self.publisher_m = None
        if USE_LASERS:
            self.publisher_m = self.create_publisher(DetectionM, "detections_m", 10)

        self.publisher_markers = self.create_publisher(Marker, "markers", 10)
        self.publisher_bg = self.create_publisher(BackgroundActivations, "background_activations", 10)


        # Last received frame (for saving)
        self._last_frame = None

        # Where to store images
        self._output_path = "./data"
        os.makedirs(self._output_path, exist_ok=True)
        self._snapshot_path = "./snapshots"
        os.makedirs(self._snapshot_path, exist_ok=True)

        # Internal execution state
        self._visualization_enabled = False
        self._inference_paused = False
        self._two_stage_enabled = False
        self._autosave_defect_snapshots = False
        self._frame_limiter = True

        # Runtime tunables (dynamic via services)
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
        self._last_pushed_threshold = None   # avoid re-assigning an unchanged value every frame
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

        # Laser state (latest samples)
        self._laser_depths = [float("nan"), float("nan"), float("nan")]
        if USE_LASERS:
            if len(LASER_TOPICS) != 3:
                self.get_logger().error("LASER_TOPICS must have exactly 3 entries. Disabling lasers.")
            elif len(LASER_XY_PIXELS) != 3:
                self.get_logger().error("LASER_XY_PIXELS must have exactly 3 (x,y) pairs. Disabling lasers.")
            else:
                self._setup_laser_subscriptions()

        # ------------------------------------------------
        # Services (existing)
        # ------------------------------------------------
        self.create_service(SetBool,    "magician_vision_classifier/set_visualization", self._set_visualization_cb)
        self.create_service(SetBool,    "magician_vision_classifier/pause", self._pause_inference_cb)
        self.create_service(SetBool,    "magician_vision_classifier/set_two_stage", self._set_two_stage_cb)
        self.create_service(SetFloat64, "magician_vision_classifier/set_fps", self._set_fps_cb)
        self.create_service(SetInt64,   "magician_vision_classifier/set_step", self._set_step_cb)
        self.create_service(SetFloat64, "magician_vision_classifier/set_threshold", self._set_threshold_cb)
        self.create_service(SetFloat64, "magician_vision_classifier/set_erosion_kernel", self._set_erosion_kernel_cb)
        self.create_service(SetFloat64, "magician_vision_classifier/set_min_votes", self._set_min_votes_cb)
        self.create_service(Trigger,    "magician_vision_classifier/remember_defect", self._remember_defect_cb)
        self.create_service(Trigger,    "magician_vision_classifier/remember_clean", self._remember_clean_cb)
        self.create_service(Trigger,    "magician_vision_classifier/scan_markers", self._scan_markers_cb)
        self.create_service(SetBool,    "magician_vision_classifier/set_autosave_defect_snapshots", self._set_autosave_defect_snapshots_cb)
        self.create_service(Trigger,    "magician_vision_classifier/snapshot", self._snapshot_cb)
        self.create_service(SetBool,    "magician_vision_classifier/set_frame_limiter", self._set_frame_limiter_cb)
        #self.create_service(SetString,  "magician_vision_classifier/set_model", self._set_model_cb)
        self.create_service(SetBool,    "magician_vision_classifier/set_majority_voting", self._set_majority_voting_cb)

        # ------------------------------------------------
        # Services (NEW): runtime tuning
        # ------------------------------------------------
        self.get_logger().info("Services ready:")
        self.get_logger().info("  magician_vision_classifier/set_visualization (SetBool)")
        self.get_logger().info("  magician_vision_classifier/pause (SetBool)")
        self.get_logger().info("  magician_vision_classifier/set_two_stage (SetBool)")
        self.get_logger().info("  magician_vision_classifier/set_fps (SetFloat64)")
        self.get_logger().info("  magician_vision_classifier/set_step (SetInt64)")
        self.get_logger().info("  magician_vision_classifier/set_threshold (SetFloat64)")
        self.get_logger().info("  magician_vision_classifier/set_erosion_kernel (SetFloat64, int 0..5)")
        self.get_logger().info("  magician_vision_classifier/set_min_votes (SetFloat64, int; accept tile if >=N activated tiles incl. itself in the (2k+1)^2 neighborhood; 0/1 disables voting)")
        self.get_logger().info("  magician_vision_classifier/remember_defect (Trigger)")
        self.get_logger().info("  magician_vision_classifier/remember_clean (Trigger)")
        self.get_logger().info("  magician_vision_classifier/set_autosave_defect_snapshots (SetBool)")
        self.get_logger().info("  magician_vision_classifier/snapshot (Trigger)")
        self.get_logger().info("  magician_vision_classifier/set_frame_limiter (SetBool)")
        #self.get_logger().info("  magician_vision_classifier/set_model (SetString)")
        self.get_logger().info("  magician_vision_classifier/set_majority_voting (SetBool)")


        if USE_LASERS and self.publisher_m is not None:
            self.get_logger().info(f"Laser fusion ENABLED: topics={LASER_TOPICS} xy={LASER_XY_PIXELS} p={LASER_IDW_POWER}")
        else:
            self.get_logger().info("Laser fusion DISABLED")

    def apply_preset(self, preset):
        """Adopt a recommended_configuration.json preset as the node's startup state.

        These are only DEFAULTS -- every one stays overridable at runtime through the
        existing services, so an operator can still retune live without editing the file.
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
        self.get_logger().info(
            f"Preset '{preset.get('name','?')}': model={preset.get('model')} "
            f"gate={gate.get('mode')}@{gate.get('threshold')} step={self._step_size} "
            f"fps={self._target_fps} erosion_kernel={self._erosion_kernel} "
            f"min_votes={self._min_votes} majority_voting={self._majority_voting}")
        if preset.get("description"):
            self.get_logger().info(f"  {preset['description']}")
        if "detected" in m and "false_alarm" in m:
            self.get_logger().info(
                f"  expected (from {m.get('source','curve')}): detects {m['detected']:.1%} "
                f"of defect tiles, false-alarms on {m['false_alarm']:.2%} of clean tiles")

    # -------------------------
    # Laser subscriptions
    # -------------------------
    def _setup_laser_subscriptions(self):
        """Create ROS2 subscriptions for 3 laser depth sensors."""
        def make_cb(i):
            def _cb(msg: Float32):
                with self._lock:
                    self._laser_depths[i] = float(msg.data)
            return _cb

        self.create_subscription(Float32, LASER_TOPICS[0], make_cb(0), 10)
        self.create_subscription(Float32, LASER_TOPICS[1], make_cb(1), 10)
        self.create_subscription(Float32, LASER_TOPICS[2], make_cb(2), 10)

    # -------------------------
    # Service callbacks  
    # -------------------------
    def _set_visualization_cb(self, request, response):
        """Service callback to toggle visualization on/off."""
        with self._lock:
            self._visualization_enabled = bool(request.data)
        response.success = True
        response.message = ("Visualization ENABLED" if request.data else "Visualization DISABLED")
        self.get_logger().info(response.message)
        return response

    def _pause_inference_cb(self, request, response):
        """Service callback to pause/resume inference."""
        with self._lock:
            self._inference_paused = bool(request.data)
        response.success = True
        response.message = ("Inference PAUSED" if request.data else "Inference RESUMED")
        self.get_logger().info(response.message)
        return response

    def _set_two_stage_cb(self, request, response):
        """Service callback to toggle two-stage ensemble mode on/off."""
        with self._lock:
            self._two_stage_enabled = bool(request.data)
        response.success = True
        response.message = ("Two-stage execution ENABLED" if request.data else "Two-stage execution DISABLED")
        self.get_logger().info(response.message)
        return response
 
    def _set_fps_cb(self, request, response):
        """Service callback to set target FPS (0 = no limiting)."""
        fps = float(request.data)
        with self._lock:
            self._target_fps = max(0.0, fps)  # 0 => no limiting
        response.success = True
        response.message = f"Target FPS set to {self._target_fps}"
        self.get_logger().info(response.message)
        return response

    def _set_step_cb(self, request, response):
        """Service callback to set tile step size (minimum 1)."""
        step = int(request.data)
        with self._lock:
            self._step_size = max(1, step)
        response.success = True
        response.message = f"Step size set to {self._step_size}"
        self.get_logger().info(response.message)
        return response

    def _set_threshold_cb(self, request, response):
        """Set the gate score threshold. Semantics depend on the model's gate mode
        (under the default "defect_mass" this thresholds 1 - P(clean), not the max
        softmax probability). A NEGATIVE value clears the override and follows the
        model's own calibrated gate. The expected detection / false-alarm trade-off
        at the new setting is looked up from the model's threshold curve and
        returned in the response."""
        raw = float(request.data)
        thr = None if raw < 0.0 else max(0.0, min(1.0, raw))
        with self._lock:
            self._threshold = thr
        if thr is None:
            response.message = "Threshold override CLEARED — following the model's calibrated gate"
        else:
            response.message = f"Gate threshold set to {thr:.3f}\n" + self._threshold_tradeoff_text(thr)
        response.success = True
        self.get_logger().info(response.message)
        return response

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
        self.get_logger().info(
            f"{context}{self._threshold_tradeoff_text(threshold)}")

    def _set_erosion_kernel_cb(self, request, response):
        """Set the voting neighborhood radius k; votes are counted over the (2k+1)^2 tiles around each activation."""
        k = max(0, min(5, int(request.data)))
        with self._lock:
            self._erosion_kernel = k
        response.success = True
        response.message = f"Erosion kernel set to {self._erosion_kernel} (neighborhood {(2*self._erosion_kernel+1)**2} tiles)"
        self.get_logger().info(response.message)
        return response

    def _set_min_votes_cb(self, request, response):
        """Require N activated tiles (including the tile itself) in the voting neighborhood for an activation to be accepted. 0/1 disables voting."""
        v = max(0, int(request.data))
        with self._lock:
            self._min_votes = v
        response.success = True
        response.message = f"Minimum votes set to {self._min_votes}"
        self.get_logger().info(response.message)
        return response

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

    def _scan_markers_cb(self, request, response):
        """Service callback to activate marker scanning for MARKER_SCAN_DURATION_S seconds."""
        with self._lock:
            self._marker_scan_until = time.monotonic() + MARKER_SCAN_DURATION_S
        response.success = True
        response.message = f"Marker scanning active for {MARKER_SCAN_DURATION_S:.0f} s"
        self.get_logger().info(response.message)
        return response

    def _remember_defect_cb(self, request, response):
        """Service callback to save the current frame tagged as a defect."""
        success, msg = self._save_current_frame("defect")
        response.success = success
        response.message = msg
        self.get_logger().info(msg)
        return response

    def _remember_clean_cb(self, request, response):
        """Service callback to save the current frame tagged as clean."""
        success, msg = self._save_current_frame("clean")
        response.success = success
        response.message = msg
        self.get_logger().info(msg)
        return response

    def _set_autosave_defect_snapshots_cb(self, request, response):
        """Service callback to enable/disable automatic saving of frames when a defect is detected."""
        with self._lock:
            self._autosave_defect_snapshots = bool(request.data)
        response.success = True
        response.message = ("Autosave defect snapshots ENABLED" if request.data else "Autosave defect snapshots DISABLED")
        self.get_logger().info(response.message)
        return response

    def _set_frame_limiter_cb(self, request, response):
        """Service callback to enable/disable the duplicate-frame limiter (False = unlimited framerate)."""
        with self._lock:
            self._frame_limiter = bool(request.data)
        response.success = True
        response.message = ("Frame limiter ENABLED" if request.data else "Frame limiter DISABLED (unlimited framerate)")
        self.get_logger().info(response.message)
        return response

    def _set_majority_voting_cb(self, request, response):
        """Service callback to enable/disable majority voting across inference tiles."""
        with self._lock:
            self._majority_voting = bool(request.data)
        response.success = True
        response.message = ("Majority voting ENABLED" if request.data else "Majority voting DISABLED")
        self.get_logger().info(response.message)
        return response

    def _set_model_cb(self, request, response):
        """Service callback to hot-swap the single classifier model at runtime."""
        name = request.data.strip()

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
            response.success = False
            response.message = f"Model file not found: {model_path}"
            self.get_logger().error(response.message)
            return response

        if not os.path.isfile(cfg_path):
            response.success = False
            response.message = f"Config file not found: {cfg_path}"
            self.get_logger().error(response.message)
            return response

        if self._single_classifier is None:
            response.success = False
            response.message = "Single classifier not yet initialized"
            self.get_logger().error(response.message)
            return response

        self.get_logger().info(f"Hot-swapping model to '{stem}' from {directory} ...")
        with self._model_lock:
            ok = self._single_classifier.reload_model(directory, stem)

        if ok:
            with self._lock:
                self._model_dir = directory

        response.success = ok
        response.message = (f"Model reloaded: {stem}" if ok
                            else f"Failed to reload model: {stem}")
        self.get_logger().info(response.message)
        return response

    def _snapshot_cb(self, request, response):
        """Service callback to save the current frame on demand to the snapshots directory."""
        with self._lock:
            frame = self._last_frame

        if frame is None:
            response.success = False
            response.message = "No frame available to save."
            self.get_logger().warning(response.message)
            return response

        full_path = os.path.join(
            self._snapshot_path,
            self._make_timestamped_basename("snapshot") + ".png",
        )
        try:
            cv2.imwrite(full_path, frame)
            response.success = True
            response.message = f"Saved: {full_path}"
        except Exception as e:
            response.success = False
            response.message = str(e)
        self.get_logger().info(response.message)
        return response

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

    def get_laser_depths(self):
        """Thread-safe getter for the latest laser depth readings."""
        with self._lock:
            return list(self._laser_depths)

    def is_marker_scanning(self):
        """Check whether marker scanning is currently active."""
        with self._lock:
            return time.monotonic() < self._marker_scan_until

    # -------------------------
    # Publishers
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
        Publish a Marker ROS2 message with 3D position and orientation.

        Converts the rvec (Rodrigues vector) to a quaternion for the pose message.
        """
        msg = Marker()
        msg.id = str(marker_id)

        qx, qy, qz, qw = rvec_to_quaternion(rvec)
        pose = Pose()
        pose.position.x = float(tvec[0])
        pose.position.y = float(tvec[1])
        pose.position.z = float(tvec[2])
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        msg.pose = pose
        self.publisher_markers.publish(msg)

    def scan_and_publish_markers(self, frame):
        """
        Detect ArUco markers in the current frame and publish Marker messages.

        Uses the cached camera matrix for pose estimation. Chessboard detection
        is present but disabled (too slow for real-time).
        """
        self.get_logger().debug("Scanning frame for markers...")

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            gray = frame if len(frame.shape) == 2 else frame[:, :, 0]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        K, dist = self._get_camera_matrix(frame)

        # --- ArUco ---
        self.get_logger().debug("Running ArUco detection...")
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is not None:
            self.get_logger().info(f"Found {len(ids)} ArUco marker(s): {ids.flatten().tolist()}")
            rvecs, tvecs = estimatePoseSingleMarkers(corners, DEFAULT_MARKER_LENGTH_M, K, dist)
            for marker_id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs):
                tvec_flat = tvec.flatten()
                self.get_logger().info(
                    f"  id={marker_id}  tvec=[{tvec_flat[0]:.3f}, {tvec_flat[1]:.3f}, {tvec_flat[2]:.3f}] m"
                )
                self.publish_marker(str(int(marker_id)), tvec, rvec)
        else:
            self.get_logger().info("No ArUco markers found.")

        # --- Chessboard (disabled: too slow) ---
        # print(f"[Markers] Running chessboard detection ({CHESSBOARD_W}x{CHESSBOARD_H})...")
        # pattern = (CHESSBOARD_W, CHESSBOARD_H)
        # flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        # found, cb_corners = cv2.findChessboardCorners(gray, pattern, flags)
        # if found and cb_corners is not None:
        #     print("[Markers] Chessboard found, refining corners...")
        #     cv2.cornerSubPix(gray, cb_corners, (20, 20), (-1, -1), self._cb_criteria)
        #
        #     objp = np.zeros((CHESSBOARD_W * CHESSBOARD_H, 3), np.float32)
        #     objp[:, :2] = np.mgrid[0:CHESSBOARD_W, 0:CHESSBOARD_H].T.reshape(-1, 2)
        #     objp *= CHESSBOARD_SQUARE_M
        #
        #     print("[Markers] Solving PnP for chessboard pose...")
        #     ok, rvec_cb, tvec_cb = cv2.solvePnP(
        #         objp, cb_corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        #     )
        #     if ok:
        #         t = tvec_cb.reshape(3)
        #         print(f"[Markers] Chessboard pose: tvec=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
        #         self.publish_marker("chessboard", t, rvec_cb.reshape(3))
        #         self.get_logger().debug(
        #             f"Chessboard tvec={t.tolist()}"
        #         )
        #     else:
        #         print("[Markers] PnP solve failed for chessboard.")
        # else:
        #     print("[Markers] No chessboard found.")
        self.get_logger().debug("Marker scan complete.")

    def publish_detection(self, x, y, w, h, det_type, det_class, probability, depth_z=0.0, ts=0):
        """Publish a Detection message with 2D box, type, class, and optional depth.

        COORDINATE CONTRACT (see msg/Detection.msg): x,y are the tile CENTRE, NOT
        the top-left corner, in DEMOSAICED (half-resolution) pixels. w,h are the
        tile size, so the covered box is [x-w/2, y-h/2]..[x+w/2, y+h/2]. This
        matches responses["points"], which generate_heatmap and
        process_predictions_erode already emit as centres -- do NOT add half a
        tile again at the call site.
        """
        try:
            msg = Detection()
            msg.header.stamp    = unix_ns_to_ros_time(ts)
            msg.header.frame_id = "camera"
            msg.x           = int(x)
            msg.y           = int(y)
            msg.w           = int(w)
            msg.h           = int(h)
            msg.depth       = float(depth_z)
            msg.type        = det_type
            msg.class_name  = det_class  # 'class' is a reserved keyword in Python
            msg.probability = float(probability)
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish detection: {e}")

    def publish_detection_m(self, cx, cy, severity, depth_z, ts):
        """Publish a DetectionM message with severity, pixel position, and interpolated depth.

        cx, cy are the tile CENTRE in demosaiced (half-res) pixels -- the same
        coordinate carried by Detection.x/y. See publish_detection.
        """
        if (not USE_LASERS) or (self.publisher_m is None):
            return
        try:
            msg = DetectionM()
            msg.header.stamp    = unix_ns_to_ros_time(ts)
            msg.header.frame_id = "camera"
            msg.severity = int(severity) #Detection_M accepts 1,2,3 here

            pose = Pose()
            pose.position.x    = float(cx)
            pose.position.y    = float(cy)
            pose.position.z    = float(depth_z)
            pose.orientation.w = 1.0  # identity quaternion
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0

            msg.location = pose
            self.publisher_m.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish detection_m: {e}")

    def publish_background_activations(self, avg_prob, ts):
        """Publish average softmax probability of clean (non-activated) tiles."""
        try:
            msg = BackgroundActivations()
            msg.header.stamp    = unix_ns_to_ros_time(ts)
            msg.header.frame_id = "camera"
            msg.background_probability = 1.0 - float(avg_prob)
            self.publisher_bg.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish background_activations: {e}")


# ========================================================
# Main
# ========================================================
def main():
    # Configure PyTorch global settings before any model is loaded.
    # TF32 on Ampere+ GPUs gives ~3× matmul throughput with negligible accuracy loss.
    torch.set_float32_matmul_precision("high")
    # cuDNN auto-tunes the fastest convolution algorithm for each fixed input shape.
    # Since tile_size is constant per model, this pays off after the first forward pass.
    torch.backends.cudnn.benchmark = True

    # Initialize ROS2
    rclpy.init()
    ros_node = DefectPublisher()
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    #PATH="./magician_vision_classifier"
    PATH = repo_root()

    # --config NAME selects a preset from recommended_configuration.json; default = first.
    preset_name = None
    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        if i + 1 < len(sys.argv):
            preset_name = sys.argv[i + 1]
    # Engine plug-ins (engine/<name>.py) take a config path; default
    # engine/<model>.json next to the plug-in, override with --model-config.
    model_config_path = None
    if "--model-config" in sys.argv:
        i = sys.argv.index("--model-config")
        if i + 1 < len(sys.argv):
            model_config_path = sys.argv[i + 1]
    preset = load_recommended_configuration(preset_name)
    model_name = preset["model"]
    ros_node.apply_preset(preset)

    # An engine plug-in (engine/<name>.py, a third-party classifier design) has no
    # .pth on the model server, so it skips ensure_model and is built from its own
    # .json config (default engine/<name>.json, override with --model-config).
    from mvc.inference.engine_base import is_engine_name, build_engine
    if is_engine_name(model_name):
        try:
            single_classifier = build_engine(model_name, cfg_path=model_config_path)
        except Exception as e:
            ros_node.get_logger().error(f"Could not build engine '{model_name}': {e}")
            rclpy.shutdown()
            return
    else:
        if model_config_path:
            ros_node.get_logger().warning("--model-config ignored — only engine plug-ins take a config path")
        # Fetch the model if it is not already here, then load it. ensure_model is a no-op
        # when model_scan() already sees a valid {name}.pth + {name}.json pair.
        from mvc.inference.model_download import ensure_model
        if not ensure_model(model_name, PATH):
            ros_node.get_logger().error(
                f"Could not obtain model '{model_name}'. Check network access to the model "
                f"server, or place {model_name}.pth + {model_name}.json in {PATH}. "
                f"Edit recommended_configuration.json to use a different preset.")
            rclpy.shutdown()
            return

        single_classifier = ClassifierPnm(
            model_path=os.path.join(PATH, f"{model_name}.pth"),
            cfg_path=os.path.join(PATH, f"{model_name}.json"),
        )
    # The preset's gate wins over the model json's own calibration, since the preset is
    # the deployment decision. Mode too -- a threshold means nothing without its mode.
    single_classifier.gateMode = preset["gate"].get("mode", single_classifier.gateMode)
    single_classifier.assignBestDefectClass = bool(
        preset["gate"].get("assign_best_defect_class", single_classifier.assignBestDefectClass))

    # Expose classifier to the ROS node for hot-swap via service
    ros_node._model_dir = PATH
    ros_node._single_classifier = single_classifier

    # State the gate's expected trade-off once, up front. The runtime default is
    # deliberately stricter than the model's KPI-optimal gate to suppress false
    # alarms; this makes the cost in missed defects explicit instead of implicit.
    ros_node._log_threshold_tradeoff(
        ros_node.get_max_probability_threshold(),
        "Startup gate setting — ")

    # Two-stage ensemble is OPTIONAL. Build it only if every member resolves; a
    # missing member used to reach ClassifierPnm's sys.exit(1) and kill the node
    # before it published anything, even though two-stage mode is off by default.
    ensemble_classifier = None
    _needed = [ENSEMBLE_STAGE1] + ENSEMBLE_MEMBERS
    _missing = [m for m in _needed
                if not (os.path.isfile(os.path.join(PATH, m + ".pth"))
                        and os.path.isfile(os.path.join(PATH, m + ".json")))]
    if _missing:
        ros_node.get_logger().warning(
            f"Two-stage ensemble DISABLED — missing models: {_missing}. "
            f"Fetch them with: python3 -m mvc.inference.model_download {' '.join(_missing)}")
    else:
        try:
            ensemble_classifier = EnsembleClassifierPnm(
                initial_model_cfg=(os.path.join(PATH, ENSEMBLE_STAGE1 + ".pth"),
                                   os.path.join(PATH, ENSEMBLE_STAGE1 + ".json")),
                model_cfg_list=[(os.path.join(PATH, m + ".pth"),
                                 os.path.join(PATH, m + ".json")) for m in ENSEMBLE_MEMBERS],
            )
        except Exception as e:
            ros_node.get_logger().warning(f"Two-stage ensemble DISABLED — build failed: {e!r}")
            ensemble_classifier = None

    tile_size = 0  # updated inside the inference block each iteration

    # Shared memory frame source
    stream_name = "stream1"
    smm = SharedMemoryManager(
        "./libSharedMemoryVideoBuffers.so",
        descriptor="video_frames.shm",
        frameName=stream_name,
        connect=True,
    )

    last_processed_timestamp = None
    _warned_no_ensemble = False   # log the two-stage fallback once, not every frame

    try:
        while True:
            loop_start = time.perf_counter()

            if ros_node.frame_limiter_enabled():
                ts = smm.get_timestamp()
                if ts is not None and ts == last_processed_timestamp:
                    time.sleep(0.001)
                    continue

            frame          = smm.read_from_shared_memory()
            frameTimestamp = smm.unix_timestamp

            # Get image to work on
            if frame is None or smm.frame_size == 0:
                ros_node.get_logger().warning("Couldn't read frame from Shared Memory")
                time.sleep(0.1)
                continue

            if ros_node.frame_limiter_enabled() and frameTimestamp == last_processed_timestamp:
                time.sleep(0.001)
                continue
            last_processed_timestamp = frameTimestamp

            ros_node._last_frame = frame.copy()
            ros_node._last_frame_timestamp = frameTimestamp

            # Marker scanning (runs regardless of inference pause state)
            if ros_node.is_marker_scanning():
                ros_node.scan_and_publish_markers(frame)

            # Pause inference
            if ros_node.inference_paused():
                time.sleep(0.01)
                continue

            # Run the neural network
            majority_voting = ros_node.majority_voting_enabled()
            with torch.inference_mode():
                if ros_node.two_stage_enabled() and ensemble_classifier is None:
                    if not _warned_no_ensemble:
                        ros_node.get_logger().warning(
                            "two_stage requested but the ensemble is unavailable — "
                            "falling back to the single classifier")
                        _warned_no_ensemble = True

                if ros_node.two_stage_enabled() and ensemble_classifier is not None:
                    with ros_node._model_lock:
                        ensemble_classifier.step = ros_node.get_step_size()
                        thr = ros_node.get_max_probability_threshold()
                        if thr is not None and thr != ensemble_classifier.maxProbabilityThreshold:
                            ensemble_classifier.maxProbabilityThreshold = thr
                        tile_size = ensemble_classifier.tile_size

                        heatmap, occupancy, responses = ensemble_classifier.forward(
                            frame,
                            majorityVote=majority_voting,
                            parallel=True,
                            multimodel=True,
                        )
                else:
                    with ros_node._model_lock:
                        single_classifier.step = ros_node.get_step_size()
                        # Push the override only when it CHANGED. Assigning every frame
                        # meant a model hot-swap silently lost the new model's calibrated
                        # gate (reload_model re-reads it, then the next frame overwrote
                        # it again). A None override leaves the model's own gate alone.
                        thr = ros_node.get_max_probability_threshold()
                        if thr is not None and thr != single_classifier.maxProbabilityThreshold:
                            single_classifier.maxProbabilityThreshold = thr
                            ros_node._log_threshold_tradeoff(thr, "Gate threshold changed: ")
                        tile_size = single_classifier.tile_size

                        heatmap, occupancy, responses = single_classifier.forward(
                            frame,
                            majorityVote=majority_voting,
                            erosion_kernel=ros_node.get_erosion_kernel(),
                            erosion_threshold=ros_node.get_min_votes(),
                        )

            # Snapshot responses for _save_current_frame sidecar JSON
            ros_node._last_responses = responses
            ros_node._last_tile_size = tile_size

            # Publish detections
            points      = responses.get("points",      [])
            classes     = responses.get("classes",     [])
            confidences = responses.get("confidences", [])

            for (x, y), description, confidence in zip(points, classes, confidences):
                det_type, det_class = filter_type(description)


                z = 0.0
                # DetectionM with interpolated depth
                if USE_LASERS:
                    # responses["points"] already holds the tile CENTRE --
                    # generate_heatmap / process_predictions_erode append
                    # (x + tile_size//2, y + tile_size//2). Adding half a tile again
                    # here shifted every DetectionM location and every laser-depth
                    # lookup by tile_size/2 (24 px at tile_size 48).
                    cx = float(x)
                    cy = float(y)

                    depths = ros_node.get_laser_depths()
                    if all(np.isfinite(d) for d in depths):
                        z = idw_depth(cx, cy, LASER_XY_PIXELS, depths, p=LASER_IDW_POWER)
                    else:
                        z = float("nan")

                    severity = class_to_severity(det_class)
                    #Detection_M accepts severities 1,2,3
                    ros_node.publish_detection_m(cx, cy, severity, z, frameTimestamp)


                # Existing 2D detection
                ros_node.publish_detection(
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
            if ros_node.autosave_defect_snapshots_enabled() and points:
                ros_node._save_current_frame("autosaved_defect")

            # Publish average background (clean-tile) softmax probability
            ros_node.publish_background_activations(
                responses.get("background_avg_prob", 0.0),
                frameTimestamp,
            )

            # Visualization
            if ros_node.visualization_enabled():
                heatmapForAWindow,scale = resize_to_fit_screen(heatmap) 
                cv2.imshow("Classifier Output",heatmapForAWindow)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            # Target FPS limiting
            target_fps = ros_node.get_target_fps()
            if target_fps > 0.0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, (1.0 / target_fps) - elapsed)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        ros_node.get_logger().info("Interrupted by user.")

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()


