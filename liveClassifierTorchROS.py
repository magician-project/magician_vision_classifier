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

from std_srvs.srv import SetBool
from example_interfaces.srv import SetInt64
from example_interfaces.srv import SetFloat64

from std_msgs.msg import Float32
from geometry_msgs.msg import Pose

from magician_classifier.msg import Detection      # Custom ROS message
from magician_classifier.msg import DetectionM     # Custom ROS message (uint8 severity, geometry_msgs/Pose location)

# ========================================================
# Laser fusion globals (project-specific / fixed hardware)
# ========================================================
USE_LASERS = True  # Set to False to disable subscriptions + DetectionM publishing entirely.

LASER_TOPICS = [
    "/magician_grabber/dist0",
    "/magician_grabber/dist1",
    "/magician_grabber/dist2",
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
# Helpers
# ========================================================
def filterType(det_type: str):
    """
    det_type might be 'class_NegativeDentClassB'
    returns: ('NegativeDent', 'ClassB')
    """
    det_class = "Unknown"
    clean_type = det_type

    if clean_type.startswith("class_"):
        clean_type = clean_type[len("class_"):]

    for cls in ("ClassA", "ClassB", "ClassC"):
        if clean_type.endswith(cls):
            det_class = cls
            clean_type = clean_type[:-len(cls)]
            break

    return clean_type, det_class


def class_to_severity(det_class: str) -> int:
    # DetectionM.msg convention (as you described):
    #   SEVERITY_CLASS_A = 1
    #   SEVERITY_CLASS_B = 2
    #   SEVERITY_CLASS_C = 3
    if det_class == "ClassA":
        return 1
    if det_class == "ClassB":
        return 2
    if det_class == "ClassC":
        return 3
    return 0


def idw_depth(x: float, y: float, xy_list, d_list, p: float = 2.0) -> float:
    """
    Inverse Distance Weighting interpolation using 3 samples.
    xy_list: [(x0,y0),(x1,y1),(x2,y2)]
    d_list:  [d0,d1,d2]
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
# ROS Node
# ========================================================
class DefectPublisher(Node):
    """ROS2 node that publishes Detection (+ optional DetectionM) and exposes runtime tuning services."""

    def __init__(self):
        super().__init__("magician_vision_classifier_publisher")

        # Publishers
        self.publisher_ = self.create_publisher(Detection, "detections", 10)

        self.publisher_m = None
        if USE_LASERS:
            self.publisher_m = self.create_publisher(DetectionM, "detections_m", 10)

        # Internal execution state
        self._visualization_enabled = False
        self._inference_paused = False
        self._two_stage_enabled = False

        # Runtime tunables (dynamic via services)
        self._target_fps = 23.0
        self._step_size = 18
        self._threshold = 0.6

        self._lock = threading.Lock()

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
        self.create_service(SetBool, "set_visualization", self._set_visualization_cb)
        self.create_service(SetBool, "pause", self._pause_inference_cb)
        self.create_service(SetBool, "set_two_stage", self._set_two_stage_cb)

        # ------------------------------------------------
        # Services (NEW): runtime tuning
        # ------------------------------------------------
        self.create_service(SetFloat64, "set_fps", self._set_fps_cb)
        self.create_service(SetInt64, "set_step", self._set_step_cb)
        self.create_service(SetFloat64, "set_threshold", self._set_threshold_cb)

        self.get_logger().info("Services ready:")
        self.get_logger().info("  /set_visualization (SetBool)")
        self.get_logger().info("  /pause (SetBool)")
        self.get_logger().info("  /set_two_stage (SetBool)")
        self.get_logger().info("  /set_fps (SetFloat64)")
        self.get_logger().info("  /set_step (SetInt64)")
        self.get_logger().info("  /set_threshold (SetFloat64)")

        if USE_LASERS and self.publisher_m is not None:
            self.get_logger().info(f"Laser fusion ENABLED: topics={LASER_TOPICS} xy={LASER_XY_PIXELS} p={LASER_IDW_POWER}")
        else:
            self.get_logger().info("Laser fusion DISABLED")

    # -------------------------
    # Laser subscriptions
    # -------------------------
    def _setup_laser_subscriptions(self):
        def make_cb(i):
            def _cb(msg: Float32):
                with self._lock:
                    self._laser_depths[i] = float(msg.data)
            return _cb

        self.create_subscription(Float32, LASER_TOPICS[0], make_cb(0), 10)
        self.create_subscription(Float32, LASER_TOPICS[1], make_cb(1), 10)
        self.create_subscription(Float32, LASER_TOPICS[2], make_cb(2), 10)

    # -------------------------
    # Service callbacks (existing)
    # -------------------------
    def _set_visualization_cb(self, request, response):
        with self._lock:
            self._visualization_enabled = bool(request.data)
        response.success = True
        response.message = ("Visualization ENABLED" if request.data else "Visualization DISABLED")
        self.get_logger().info(response.message)
        return response

    def _pause_inference_cb(self, request, response):
        with self._lock:
            self._inference_paused = bool(request.data)
        response.success = True
        response.message = ("Inference PAUSED" if request.data else "Inference RESUMED")
        self.get_logger().info(response.message)
        return response

    def _set_two_stage_cb(self, request, response):
        with self._lock:
            self._two_stage_enabled = bool(request.data)
        response.success = True
        response.message = ("Two-stage execution ENABLED" if request.data else "Two-stage execution DISABLED")
        self.get_logger().info(response.message)
        return response

    # -------------------------
    # Service callbacks (NEW): runtime tuning
    # -------------------------
    def _set_fps_cb(self, request, response):
        fps = float(request.data)
        with self._lock:
            self._target_fps = max(0.0, fps)  # 0 => no limiting
        response.success = True
        response.message = f"Target FPS set to {self._target_fps}"
        self.get_logger().info(response.message)
        return response

    def _set_step_cb(self, request, response):
        step = int(request.data)
        with self._lock:
            self._step_size = max(1, step)
        response.success = True
        response.message = f"Step size set to {self._step_size}"
        self.get_logger().info(response.message)
        return response

    def _set_threshold_cb(self, request, response):
        thr = float(request.data)
        with self._lock:
            self._threshold = thr
        response.success = True
        response.message = f"Max probability threshold set to {self._threshold}"
        self.get_logger().info(response.message)
        return response

    # -------------------------
    # Thread-safe getters
    # -------------------------
    def get_step_size(self):
        with self._lock:
            return self._step_size

    def get_max_probability_threshold(self):
        with self._lock:
            return self._threshold

    def get_target_fps(self):
        with self._lock:
            return self._target_fps

    def visualization_enabled(self):
        with self._lock:
            return self._visualization_enabled

    def inference_paused(self):
        with self._lock:
            return self._inference_paused

    def two_stage_enabled(self):
        with self._lock:
            return self._two_stage_enabled

    def get_laser_depths(self):
        with self._lock:
            return list(self._laser_depths)

    # -------------------------
    # Publishers
    # -------------------------
    def publish_detection(self, x, y, w, h, det_type, det_class, probability):
        msg = Detection()
        msg.x = int(x)
        msg.y = int(y)
        msg.w = int(w)
        msg.h = int(h)
        msg.type = det_type
        msg.class_ = det_class  # 'class' is reserved in Python
        msg.probability = float(probability)
        self.publisher_.publish(msg)

    def publish_detection_m(self, cx, cy, severity, depth_z):
        if (not USE_LASERS) or (self.publisher_m is None):
            return

        msg = DetectionM()
        msg.severity = int(severity)

        pose = Pose()
        pose.position.x = float(cx)
        pose.position.y = float(cy)
        pose.position.z = float(depth_z)
        pose.orientation.w = 1.0  # identity
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0

        msg.location = pose
        self.publisher_m.publish(msg)


# ========================================================
# Main
# ========================================================
def main():
    # Initialize ROS2
    rclpy.init()
    ros_node = DefectPublisher()
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    # Initialize Neural Networks (both modes)
    SingleClassifier = ClassifierPnm(
        model_path="./magician_vision_classifier/allclass_resnet18.pth",
        cfg_path="./magician_vision_classifier/allclass_resnet18.json",
    )

    EnsembleClassifier = EnsembleClassifierPnm(
        initial_model_cfg=(
            "./magician_vision_classifier/binary_small_cnn.pth",
            "./magician_vision_classifier/binary_small_cnn.json",
        ),
        model_cfg_list=[
            ("./magician_vision_classifier/allclass_verysmall_cnn.pth", "./magician_vision_classifier/allclass_verysmall_cnn.json"),
            ("./magician_vision_classifier/allclass_resnet18.pth", "./magician_vision_classifier/allclass_resnet18.json"),
            ("./magician_vision_classifier/allclass_resnext50.pth", "./magician_vision_classifier/allclass_resnext50.json"),
            # ("./magician_vision_classifier/allclass_efficientnet_v2_s.pth", "./magician_vision_classifier/allclass_efficientnet_v2_s.json"),  # slowest
            ("./magician_vision_classifier/allclass_convnext_tiny.pth", "./magician_vision_classifier/allclass_convnext_tiny.json"),
        ],
    )

    torch.set_float32_matmul_precision("medium")
    tile_size = 0  # invalid value to ensure it is observed

    # Shared memory frame source
    stream_name = "stream1"
    smm = SharedMemoryManager(
        "./libSharedMemoryVideoBuffers.so",
        descriptor="video_frames.shm",
        frameName=stream_name,
        connect=True,
    )

    majorityVotingConfiguration = True

    try:
        while True:
            loop_start = time.perf_counter()

            frame = smm.read_from_shared_memory()

            # Get image to work on
            if frame is None or smm.frame_size == 0:
                print("Error: Couldn't read frame from Shared Memory")
                time.sleep(0.1)
                continue

            # Pause inference
            if ros_node.inference_paused():
                time.sleep(0.01)
                continue

            # Run the neural network
            with torch.inference_mode():
                if ros_node.two_stage_enabled():
                    EnsembleClassifier.step = ros_node.get_step_size()
                    EnsembleClassifier.maxProbabilityThreshold = ros_node.get_max_probability_threshold()
                    tile_size = EnsembleClassifier.tile_size

                    heatmap, occupancy, responses = EnsembleClassifier.forward(
                        frame,
                        majorityVote=majorityVotingConfiguration,
                        parallel=True,
                        multimodel=True,
                    )
                else:
                    SingleClassifier.step = ros_node.get_step_size()
                    SingleClassifier.maxProbabilityThreshold = ros_node.get_max_probability_threshold()
                    tile_size = SingleClassifier.tile_size

                    heatmap, occupancy, responses = SingleClassifier.forward(
                        frame,
                        majorityVote=majorityVotingConfiguration,
                        erosion_kernel=0,
                        erosion_threshold=0,
                    )

            # Publish detections
            points = responses.get("points", [])
            classes = responses.get("classes", [])
            confidences = responses.get("confidences", [])

            for (x, y), description, confidence in zip(points, classes, confidences):
                det_type, det_class = filterType(description)

                # Existing 2D detection
                ros_node.publish_detection(
                    x=x,
                    y=y,
                    w=tile_size,
                    h=tile_size,
                    det_type=det_type,
                    det_class=det_class,
                    probability=confidence,
                )

                # Optional DetectionM with interpolated depth
                if USE_LASERS:
                    cx = float(x) + 0.5 * float(tile_size)
                    cy = float(y) + 0.5 * float(tile_size)

                    depths = ros_node.get_laser_depths()
                    if all(np.isfinite(d) for d in depths):
                        z = idw_depth(cx, cy, LASER_XY_PIXELS, depths, p=LASER_IDW_POWER)
                    else:
                        z = float("nan")

                    severity = class_to_severity(det_class)
                    ros_node.publish_detection_m(cx, cy, severity, z)

            # Visualization
            if ros_node.visualization_enabled():
                cv2.imshow("Classifier Output", heatmap)
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
        print("Interrupted by user.")

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()

