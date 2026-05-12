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
from liveClassifierTorch import ClassifierPnm
from EnsembleClassifier import EnsembleClassifierPnm
from SharedMemoryManager import SharedMemoryManager

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
# Laser fusion globals (project-specific / fixed hardware)
# ========================================================
USE_LASERS = True  # Set to False to disable subscriptions + DetectionM publishing entirely.

LASER_TOPICS = [
    "magician_grabber/distance1",
    "magician_grabber/distance2",
    "magician_grabber/distance3",
]

# Laser locations in the classifier's 2D image plane (pixels)
# (x0,y0), (x1,y1), (x2,y2)
LASER_XY_PIXELS = [
    (120.0, 200.0),
    (320.0, 200.0),
    (520.0, 200.0),
]

LASER_IDW_POWER = 2.0  # IDW interpolation power

# ========================================================
# Marker scanning globals
# ========================================================
MARKER_SCAN_DURATION_S   = 3.0          # seconds each scan_markers call stays active
ARUCO_DICT_NAME          = "DICT_6X6_250"
DEFAULT_MARKER_LENGTH_M  = 0.05         # 5 cm default ArUco marker side length
CHESSBOARD_W             = 9            # inner corner columns
CHESSBOARD_H             = 6            # inner corner rows
CHESSBOARD_SQUARE_M      = 0.024        # 24 mm square size


# ========================================================
# Helpers
# ========================================================

def resize_to_fit_screen(img, max_w=1280, max_h=720, only_shrink=True):
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

        # Runtime tunables (dynamic via services)
        self._target_fps = 23.0
        self._step_size = 18
        self._threshold = 0.6

        self._lock = threading.Lock()

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
        self.create_service(Trigger,    "magician_vision_classifier/remember_defect", self._remember_defect_cb)
        self.create_service(Trigger,    "magician_vision_classifier/remember_clean", self._remember_clean_cb)
        self.create_service(Trigger,    "magician_vision_classifier/scan_markers", self._scan_markers_cb)
        self.create_service(SetBool,    "magician_vision_classifier/set_autosave_defect_snapshots", self._set_autosave_defect_snapshots_cb)
        self.create_service(Trigger,    "magician_vision_classifier/snapshot", self._snapshot_cb)

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
        self.get_logger().info("  magician_vision_classifier/remember_defect (Trigger)")
        self.get_logger().info("  magician_vision_classifier/remember_clean (Trigger)")
        self.get_logger().info("  magician_vision_classifier/set_autosave_defect_snapshots (SetBool)")
        self.get_logger().info("  magician_vision_classifier/snapshot (Trigger)")


        if USE_LASERS and self.publisher_m is not None:
            self.get_logger().info(f"Laser fusion ENABLED: topics={LASER_TOPICS} xy={LASER_XY_PIXELS} p={LASER_IDW_POWER}")
        else:
            self.get_logger().info("Laser fusion DISABLED")

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
        """Service callback to set the confidence threshold for low-activation tiles."""
        thr = float(request.data)
        with self._lock:
            self._threshold = thr
        response.success = True
        response.message = f"Max probability threshold set to {self._threshold}"
        self.get_logger().info(response.message)
        return response

    def _save_current_frame(self, prefix: str):
        """
        Save the last received frame as a PNG with a timestamped filename.

        Thread-safe: acquires the lock to read the frame pointer.
        Returns (success: bool, message: str).
        """
        # Take a snapshot under the lock so the main thread can't replace the frame
        # while we are reading it (data race between main loop and ROS spin thread).
        with self._lock:
            frame = self._last_frame

        if frame is None:
            return False, "No frame available to save."

        now = datetime.now()
        filename = (
            f"{prefix}_"
            f"{now.year:04d}_{now.month:02d}_{now.day:02d}_"
            f"{now.hour:02d}_{now.minute:02d}_{now.second:02d}_"
            f"{int(now.microsecond / 1000):03d}.png"
        )
        full_path = os.path.join(self._output_path, filename)

        try:
            cv2.imwrite(full_path, frame)
            return True, f"Saved: {full_path}"
        except Exception as e:
            return False, str(e)

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

    def _snapshot_cb(self, request, response):
        """Service callback to save the current frame on demand to the snapshots directory."""
        with self._lock:
            frame = self._last_frame

        if frame is None:
            response.success = False
            response.message = "No frame available to save."
            self.get_logger().warning(response.message)
            return response

        now = datetime.now()
        filename = (
            f"snapshot_"
            f"{now.year:04d}_{now.month:02d}_{now.day:02d}_"
            f"{now.hour:02d}_{now.minute:02d}_{now.second:02d}_"
            f"{int(now.microsecond / 1000):03d}.png"
        )
        full_path = os.path.join(self._snapshot_path, filename)
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
        self.get_logger().info("Scanning frame for markers...")

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
        self.get_logger().info("Marker scan complete.")

    def publish_detection(self, x, y, w, h, det_type, det_class, probability, depth_z=0.0, ts=0):
        """Publish a Detection message with 2D bounding box, type, class, and optional depth."""
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
        """Publish a DetectionM message with severity, pixel position, and interpolated depth."""
        if (not USE_LASERS) or (self.publisher_m is None):
            return
        try:
            msg = DetectionM()
            msg.header.stamp    = unix_ns_to_ros_time(ts)
            msg.header.frame_id = "camera"
            msg.severity = int(severity)

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

        """
        1. Inference (classify_tiles, line ~872): The model runs a forward pass on all tiles, then softmax is applied over the class dimension:
          probs = torch.nn.functional.softmax(preds.float(), dim=1)
          max_probs, predictions = torch.max(probs, dim=1)
          1. max_probs is the winning class's softmax probability for each tile (range 0–1).
          2. Filtering to background tiles (line ~619–621 in generate_heatmap):
          else:  # predicted_class == cleanClassID
              bg_prob_sum += float(confidences[idx])  # confidences = max_probs
              bg_count    += 1
          2. Only tiles predicted as cleanClassID contribute. Their confidence is their max softmax probability (i.e., the probability assigned to the clean/background class by the
           model).
          3. Averaging (line 628):
          responses["background_avg_prob"] = bg_prob_sum / bg_count if bg_count > 0 else 0.0

          So what does 0.98 mean?

          It means the background tiles are classified as "clean" with 98% average confidence. This is completely normal — when most of the scene is background, the model is very
          sure those tiles are clean, so their softmax scores cluster near 1.0. A lower value (e.g. 0.60–0.75) would indicate the background tiles are ambiguous, which could be a
          signal of noise or a degraded model.
        """
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
    PATH="."

    # Initialize Neural Networks (both modes)
    single_classifier = ClassifierPnm(
        model_path=f"{PATH}/allclass_convnext_tiny.pth",
        cfg_path=f"{PATH}/allclass_convnext_tiny.json",
    )

    ensemble_classifier = EnsembleClassifierPnm(
        initial_model_cfg=(
            f"{PATH}/binary_small_cnn.pth",
            f"{PATH}/binary_small_cnn.json",
        ),
        model_cfg_list=[
            (f"{PATH}/allclass_verysmall_cnn.pth", f"{PATH}/allclass_verysmall_cnn.json"),
            (f"{PATH}/allclass_resnet18.pth",       f"{PATH}/allclass_resnet18.json"),
            (f"{PATH}/allclass_resnext50.pth",      f"{PATH}/allclass_resnext50.json"),
            # (f"{PATH}/allclass_efficientnet_v2_s.pth", f"{PATH}/allclass_efficientnet_v2_s.json"),  # slowest
            (f"{PATH}/allclass_convnext_tiny.pth",  f"{PATH}/allclass_convnext_tiny.json"),
        ],
    )

    tile_size = 0  # updated inside the inference block each iteration

    # Shared memory frame source
    stream_name = "stream1"
    smm = SharedMemoryManager(
        "./libSharedMemoryVideoBuffers.so",
        descriptor="video_frames.shm",
        frameName=stream_name,
        connect=True,
    )

    majority_voting = True

    try:
        while True:
            loop_start = time.perf_counter()

            frame          = smm.read_from_shared_memory()
            frameTimestamp = smm.unix_timestamp

            # Get image to work on
            if frame is None or smm.frame_size == 0:
                ros_node.get_logger().warning("Couldn't read frame from Shared Memory")
                time.sleep(0.1)
                continue

            ros_node._last_frame = frame.copy()

            # Marker scanning (runs regardless of inference pause state)
            if ros_node.is_marker_scanning():
                ros_node.scan_and_publish_markers(frame)

            # Pause inference
            if ros_node.inference_paused():
                time.sleep(0.01)
                continue

            # Run the neural network
            with torch.inference_mode():
                if ros_node.two_stage_enabled():
                    ensemble_classifier.step = ros_node.get_step_size()
                    ensemble_classifier.maxProbabilityThreshold = ros_node.get_max_probability_threshold()
                    tile_size = ensemble_classifier.tile_size

                    heatmap, occupancy, responses = ensemble_classifier.forward(
                        frame,
                        majorityVote=majority_voting,
                        parallel=True,
                        multimodel=True,
                    )
                else:
                    single_classifier.step = ros_node.get_step_size()
                    single_classifier.maxProbabilityThreshold = ros_node.get_max_probability_threshold()
                    tile_size = single_classifier.tile_size

                    heatmap, occupancy, responses = single_classifier.forward(
                        frame,
                        majorityVote=majority_voting,
                        erosion_kernel=0,
                        erosion_threshold=0,
                    )

            # Publish detections
            points      = responses.get("points",      [])
            classes     = responses.get("classes",     [])
            confidences = responses.get("confidences", [])

            for (x, y), description, confidence in zip(points, classes, confidences):
                det_type, det_class = filter_type(description)


                z = 0.0
                # DetectionM with interpolated depth
                if USE_LASERS:
                    cx = float(x) + 0.5 * float(tile_size)
                    cy = float(y) + 0.5 * float(tile_size)

                    depths = ros_node.get_laser_depths()
                    if all(np.isfinite(d) for d in depths):
                        z = idw_depth(cx, cy, LASER_XY_PIXELS, depths, p=LASER_IDW_POWER)
                    else:
                        z = float("nan")

                    severity = class_to_severity(det_class)
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


