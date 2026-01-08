#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

#--------------------------------------------------------
import os
import sys
import time
import json
import threading
import cv2
import numpy as np
import torch
from torch.nn import functional as F
#--------------------------------------------------------
from trainClassifierTorch import Classifier
from liveClassifierTorch import ClassifierPnm, runSingle
from SharedMemoryManager import SharedMemoryManager
#--------------------------------------------------------
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from example_interfaces.srv import SetInt64
from example_interfaces.srv import SetFloat64
from magician_classifier.msg import Detection  # Custom ROS message
#--------------------------------------------------------

class DefectPublisher(Node):
    """ROS2 node that publishes Detection messages and controls execution state."""

    def __init__(self):
        super().__init__('magician_vision_classifier_publisher')

        self.publisher_ = self.create_publisher(Detection, 'detections', 10)

        # -------------------------
        # Internal execution state
        # -------------------------
        self._visualization_enabled = False
        self._inference_paused      = False
        self._two_stage_enabled     = False

        self._lock = threading.Lock()

        # -------------------------
        # Services
        # -------------------------
        self.create_service(SetBool, 'set_visualization', self._set_visualization_cb)
        self.create_service(SetBool, 'pause',   self._pause_inference_cb)
        self.create_service(SetBool, 'set_two_stage',     self._set_two_stage_cb)

        self.get_logger().info("Services ready:")
        self.get_logger().info("  /set_visualization")
        self.get_logger().info("  /pause")
        self.get_logger().info("  /set_two_stage")

    # -------------------------
    # Service callbacks
    # -------------------------
    def _set_visualization_cb(self, request, response):
        with self._lock:
            self._visualization_enabled = bool(request.data)

        response.success = True
        response.message = ( "Visualization ENABLED" if request.data else "Visualization DISABLED" )
        self.get_logger().info(response.message)
        return response

    def _pause_inference_cb(self, request, response):
        with self._lock:
            self._inference_paused = bool(request.data)

        response.success = True
        response.message = ("Inference PAUSED" if request.data else "Inference RESUMED" )
        self.get_logger().info(response.message)
        return response

    def _set_two_stage_cb(self, request, response):
        with self._lock:
            self._two_stage_enabled = bool(request.data)

        response.success = True
        response.message = ( "Two-stage execution ENABLED" if request.data else "Two-stage execution DISABLED" )
        self.get_logger().info(response.message)
        return response

    # -------------------------
    # Thread-safe getters
    # -------------------------
    def get_step_size(self):
        with self._lock:
            return 18 #TODO: Make this dynamic

    def get_max_probability_threshold(self):
        with self._lock:
            return 0.6 #TODO: Make this dynamic

    def get_target_fps(self):
        with self._lock:
            return 23 #TODO: Make this dynamic

    def visualization_enabled(self):
        with self._lock:
            return self._visualization_enabled

    def inference_paused(self):
        with self._lock:
            return self._inference_paused

    def two_stage_enabled(self):
        with self._lock:
            return self._two_stage_enabled
    # -------------------------

    # -------------------------
    # Neural Network Detection Publisher
    # -------------------------
    def publish_detection(self, x, y, w, h, det_type, det_class, probability):
        msg = Detection()
        msg.x           = int(x)
        msg.y           = int(y)
        msg.w           = int(w)
        msg.h           = int(h)
        msg.type        = det_type
        msg.class_      = det_class  # 'class' is reserved in Python
        msg.probability = float(probability)
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published detection: ({x},{y},{w},{h}) {det_type}:{det_class} p={probability:.2f}")


def filterType(det_type)
    # Default values
    det_class = "Unknown"
    clean_type = det_type

    # -------------------------
    # Remove leading "class_"
    # -------------------------
    if clean_type.startswith("class_"):
        clean_type = clean_type[len("class_"):]

    # -------------------------
    # Extract and remove class suffix
    # -------------------------
    for cls in ("ClassA", "ClassB", "ClassC"):
        if clean_type.endswith(cls):
            det_class = cls
            clean_type = clean_type[:-len(cls)]
            break

    # Final cleaned det_type
    det_type_clean = clean_type
    return det_type_clean, det_class

def main():
    # Initialize ROS2
    rclpy.init()
    ros_node = DefectPublisher()
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    # Initialize Neural Network
    modelC = ClassifierPnm(model_path='last.pth', cfg_path='last.json')
    model = modelC.model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()

    # Compile the model for inference
    torch.set_float32_matmul_precision('medium')
    model = torch.compile(model)

    # Initialize Shared Memory mechanism
    stream_name = "stream1"
    smm = SharedMemoryManager( 
                              "./libSharedMemoryVideoBuffers.so",
                              descriptor="video_frames.shm",
                              frameName=stream_name,
                              connect=True
                             )
   
    majorityVotingConfiguration = True

    try:
        while True:
            loop_start = time.perf_counter()

            frame = smm.read_from_shared_memory()

            # -------------------------
            # Get image to work on
            # -------------------------
            if frame is None or smm.frame_size == 0:
                print("Error: Couldn't read frame from Shared Memory")
                time.sleep(0.1)
                continue

            # -------------------------
            # Pause inference
            # -------------------------
            if ros_node.inference_paused():
                    time.sleep(0.01)
                    continue

            # -------------------------
            # Actually run the neural network
            # -------------------------
            with torch.inference_mode():
                modelC.step                    = ros_node.get_step_size()
                modelC.maxProbabilityThreshold = rosnode.get_max_probability_threshold()
                heatmap, occupancy, responses  = modelC.forward(
                                                                frame, 
                                                                majorityVote=majorityVotingConfiguration
                                                                parallel=ros_node.two_stage_enabled(),
                                                                multimodel=ros_node.two_stage_enabled()
                                                               )
                #print("Responses:", responses)
            
            # -------------------------
            # Publish detections
            # -------------------------
            points      = responses.get('points',  [])
            classes     = responses.get('classes', [])
            confidences = responses.get('confidences', [])

            for (x, y), description , confidence in zip(points, classes, confidences):
                det_class = "Unknown"

                #det_type might be class_NegativeDentClassB
                det_type , det_class = filterType(description)
                #det_type should now be NegativeDent and det_class ClassB
                
                ros_node.publish_detection(
                                           x=x,
                                           y=y,
                                           w=modelC.tile_size,
                                           h=modelC.tile_size,
                                           det_type=det_type,
                                           det_class=det_class,
                                           probability=confidence
                                          )

            # -------------------------
            # Visualization
            # -------------------------
            if ros_node.visualization_enabled():
               cv2.imshow('Classifier Output', heatmap)
               if cv2.waitKey(1) & 0xFF == ord('q'):
                  break

            # -------------------------
            # Target FPS limiting
            # -------------------------
            target_fps = ros_node.target_fps()
            if target_fps > 0.0:
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = max(0.0, (1.0 / target_fps) - elapsed)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
            #====================================

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()

