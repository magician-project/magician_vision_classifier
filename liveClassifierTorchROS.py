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
from magician_classifier.msg import Detection  # Custom ROS message
#--------------------------------------------------------


class DefectPublisher(Node):
    """ROS2 node that publishes Detection messages."""

    def __init__(self):
        super().__init__('wx_folder_streamer_publisher')
        self.publisher_ = self.create_publisher(Detection, 'detections', 10)

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
    # model.half()  # Optional: Use half precision

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

    DETECTION_TYPE       = "defect"
    DETECTION_CLASS      = ""
    DEFAULT_W, DEFAULT_H = modelC.tile_size, modelC.tile_size  # Default bounding box size
    DEFAULT_PROB         = 0.95  # Placeholder probability

    try:
        while True:
            frame = smm.read_from_shared_memory()

            if frame is None or smm.frame_size == 0:
                print("Error: Couldn't read frame from Shared Memory")
                time.sleep(0.1)
                continue

            with torch.inference_mode():
                heatmap, occupancy, responses = modelC.forward(frame, majorityVote=True)

            cv2.imshow('Classifier Output', heatmap)
            #print("Responses:", responses)

            # Publish detections
            points  = responses.get('points',  [])
            classes = responses.get('classes', [])

            for (x, y), det_type in zip(points, classes):
                det_class = DETECTION_CLASS
                #det_type might be class_NegativeDentClassB
                if   ("ClassA" in det_type):
                       det_class = "ClassA"
                elif ("ClassB" in det_type):
                       det_class = "ClassB"
                elif ("ClassC" in det_type):
                       det_class = "ClassC"

                ros_node.publish_detection(
                                           x=x,
                                           y=y,
                                           w=DEFAULT_W,
                                           h=DEFAULT_H,
                                           det_type=det_type,
                                           det_class=det_class,
                                           probability=DEFAULT_PROB
                                          )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()

