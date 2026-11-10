#!/usr/bin/python3
"""
Dumps a live classifier session as a magician_grabber / magician_grabber_annotator dataset,
plus periodic screenshots of the desktop documenting what else was running.

Layout of one recording (same files as e.g. Datasets/FORTH_POSC_650):

    info.json                     capture settings; dataset_creator keys datasets on it
    camera.csv                    timestamp (us since record_start), frameID
    controller.csv                one row PER RECORDED FRAME, see add_frame
    colorFrame_0_NNNNN.png        packed polarization (H/2 x W/2 x 4), as in the datasets
    colorFrame_0_NNNNN.json       the classifier's detections, in annotator schema
    desktop/desktop_NNNNN.jpg     desktop screenshot. In a SUBDIRECTORY because both the
                                  annotator and the folder streamer treat every image
                                  directly inside a dataset as a camera frame
    desktop/desktop.csv           timestamp, unix_ns, frameID, file: the latest camera
                                  frame recorded when each screenshot was taken (-1 = none)

Nothing here may stall the inference loop: encoding happens on writer threads behind a
bounded queue, and a frame that does not fit is dropped and counted rather than waited for.
"""

import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

from mvc.core.read_data import readPolarPNMToRGBALive
from mvc.inference.live_torch import filter_type

# PNG level 1 costs ~0.1 s per packed frame and is only ~1.5% larger than level 3.
PNG_COMPRESSION = 1
# Three writers keep up with the camera's 23 fps at ~0.1 s per frame.
WRITER_THREADS = 3
# Frames waiting to be encoded, ~5 MB each, so a slow disk costs at most ~320 MB of RAM.
MAX_QUEUED_FRAMES = 64

# A 4K capture takes ~70 ms as JPEG but ~240 ms as PNG, so JPEG is what makes this rate
# reachable. If a capture overruns the period, the next one simply starts immediately.
DESKTOP_SCREENSHOT_PERIOD_S = 0.1
DESKTOP_JPEG_QUALITY = 80
DESKTOP_SCREENSHOT_TIMEOUT_S = 5.0

CONTROLLER_COLUMNS = ["Button1", "Button2", "Distance1", "Distance2", "Distance3",
                      "Light1", "Light2", "Light3", "Light4", "Light5", "Light6"]

# Same mapping as the annotator's _classifierDetectionsForImage. Names not listed
# here are already spelled the way the annotator spells them.
_ANNOTATOR_CLASS_NAMES = {"NegativeDent": "Negative Dent",
                          "PositiveDent": "Positive Dent",
                          "MaterialDefect": "Material Defect"}
_ANNOTATOR_SEVERITIES = {"ClassA": "Class A", "ClassB": "Class B", "ClassC": "Class C"}


def _finite_or_none(value):
    """NaN is not valid JSON, and a laser that has not reported yet is NaN."""
    value = float(value)
    return value if np.isfinite(value) else None


def annotation_from_responses(mosaic_w, mosaic_h, frame_ts_ns, responses, tile_size, lasers):
    """Annotator-schema dict for one frame. `responses` is None when inference was paused.

    pointClicks are in FULL MOSAIC pixels, the annotator's convention for clicks, while
    responses["points"] are tile centres in the half-res demosaic, hence the x2.
    "classifier" carries what the annotator schema has no field for.
    """
    points, classes, severities, probabilities = [], [], [], []
    if responses is not None:
        for (x, y), label, confidence in zip(responses.get("points", []),
                                             responses.get("classes", []),
                                             responses.get("confidences", [])):
            name, severity = filter_type(label)
            points.append([int(x) * 2, int(y) * 2])
            classes.append(_ANNOTATOR_CLASS_NAMES.get(name, name))
            # "AI" is what the annotator marks a classifier point of unknown severity with
            severities.append(_ANNOTATOR_SEVERITIES.get(severity, "AI"))
            probabilities.append(float(confidence))

    return {
        "width": int(mosaic_w),
        "height": int(mosaic_h),
        "md5hash": "",
        "regionClicks": [],
        "pointClicks": points,
        "pointClasses": classes,
        "pointSeverities": severities,
        "pointSources": ["classifier"] * len(points),
        "classifier": {
            "timestamp_ns": int(frame_ts_ns),
            "inference_ran": responses is not None,
            "tile_size": int(tile_size),
            "background_avg_prob": (float(responses.get("background_avg_prob", 0.0))
                                    if responses is not None else None),
            "probabilities": probabilities,
            "laser_depths": [_finite_or_none(d) for d in lasers],
        },
    }


class SessionRecorder:
    """One recording, from construction until stop(). Not reusable."""

    def __init__(self, directory, target_fps=0.0, logger=None):
        self.directory = directory
        self._logger = logger
        self._desktop_dir = os.path.join(directory, "desktop")
        os.makedirs(self._desktop_dir)  # raises if this recording already exists

        with open(os.path.join(directory, "info.json"), "w") as f:
            # exposure/gain/blackLevel are deliberately absent: they belong to the grabber
            # and this node cannot know them. dataset_creator accepts a missing exposure,
            # whereas an invented value would pass or fail its exposure window wrongly.
            json.dump({"delay": 0, "maxFramesToGrab": 0, "frameRate": float(target_fps)}, f)

        self._start_ns = time.time_ns()
        self._lock = threading.Lock()
        self._accepting = True
        self._next_frame = 0
        self._last_ts = None
        self.dropped = 0
        self.write_failures = 0
        self.screenshots = 0
        self.screenshot_failures = 0

        self._camera_file = open(os.path.join(directory, "camera.csv"), "w", newline="")
        self._camera_csv = csv.writer(self._camera_file)
        self._camera_csv.writerow(["timestamp", "frameID"])
        self._controller_file = open(os.path.join(directory, "controller.csv"), "w", newline="")
        self._controller_csv = csv.writer(self._controller_file)
        self._controller_csv.writerow(["timestamp", "dev_timestamp"] + CONTROLLER_COLUMNS)

        self._queue = queue.Queue(maxsize=MAX_QUEUED_FRAMES)
        self._writers = [threading.Thread(target=self._writer_loop, daemon=True)
                         for _ in range(WRITER_THREADS)]
        for thread in self._writers:
            thread.start()

        self._stop_desktop = threading.Event()
        self._desktop_thread = threading.Thread(target=self._desktop_loop, daemon=True)
        self._desktop_thread.start()

    def _log(self, method, message):
        if self._logger is not None:
            getattr(self._logger, method)(message)

    def add_frame(self, frame, frame_ts_ns, responses, tile_size, lasers, controller):
        """Queue one camera frame with its classification. Never blocks on the disk.

        `controller` is the latest value of each CONTROLLER_COLUMNS entry. controller.csv
        gets exactly one row per recorded frame, not one per serial line as the grabber
        writes, because the annotator looks rows up by frame number and this node does not
        see every camera frame, so an independent stream would drift out of step.
        """
        # With the frame limiter off the main loop re-reads the same frame repeatedly.
        # A zero timestamp means the source does not stamp frames, so it cannot be compared.
        with self._lock:
            if not self._accepting or (frame_ts_ns and frame_ts_ns == self._last_ts):
                return
            frame_id = self._next_frame
            item = (frame_id, frame, frame_ts_ns, responses, tile_size, lasers)
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # No frame number or CSV row is consumed, so numbering stays gapless.
                self.dropped += 1
                return
            self._last_ts = frame_ts_ns
            self._next_frame += 1

            elapsed_us = max(0, ((frame_ts_ns or time.time_ns()) - self._start_ns) // 1000)
            self._camera_csv.writerow([elapsed_us, frame_id])
            # The ROS topics carry neither the device clock nor Button2
            self._controller_csv.writerow([elapsed_us, 0] + [int(v) for v in controller])

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._write_frame(*item)
            except Exception as e:
                with self._lock:
                    self.write_failures += 1
                    first_failure = self.write_failures == 1
                if first_failure:  # e.g. disk full: would otherwise repeat for every frame
                    self._log("error", f"Recording: failed writing a frame: {e!r}")

    def _write_frame(self, frame_id, frame, frame_ts_ns, responses, tile_size, lasers):
        packed = readPolarPNMToRGBALive(frame)
        stem = os.path.join(self.directory, f"colorFrame_0_{frame_id:05d}")
        if not cv2.imwrite(stem + ".png", packed, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
            raise IOError(f"cv2.imwrite could not write {stem}.png")
        annotation = annotation_from_responses(packed.shape[1] * 2, packed.shape[0] * 2,
                                               frame_ts_ns, responses, tile_size, lasers)
        with open(stem + ".json", "w") as f:
            json.dump(annotation, f)

    def _desktop_loop(self):
        if shutil.which("scrot") is None:
            self._log("error", "Recording: scrot is not installed, so there will be no "
                               "desktop screenshots (sudo apt install scrot)")
            return
        with open(os.path.join(self._desktop_dir, "desktop.csv"), "w", newline="") as f:
            desktop_csv = csv.writer(f)
            desktop_csv.writerow(["timestamp", "unix_ns", "frameID", "file"])
            sequence = 0
            while True:
                started = time.monotonic()
                name = f"desktop_{sequence:05d}.jpg"
                if self._take_screenshot(os.path.join(self._desktop_dir, name)):
                    now = time.time_ns()
                    with self._lock:
                        latest_frame = self._next_frame - 1
                    desktop_csv.writerow([(now - self._start_ns) // 1000, now, latest_frame, name])
                    f.flush()
                    self.screenshots += 1
                    sequence += 1
                wait = DESKTOP_SCREENSHOT_PERIOD_S - (time.monotonic() - started)
                if self._stop_desktop.wait(max(0.0, wait)):
                    return

    def _take_screenshot(self, path):
        try:
            subprocess.run(["scrot", "--silent", "-q", str(DESKTOP_JPEG_QUALITY), path],
                           check=True, capture_output=True,
                           timeout=DESKTOP_SCREENSHOT_TIMEOUT_S)
            return True
        except (subprocess.SubprocessError, OSError) as e:
            self.screenshot_failures += 1
            if self.screenshot_failures == 1:  # e.g. no DISPLAY: would repeat every second
                detail = getattr(e, "stderr", b"") or b""
                self._log("warning", f"Recording: desktop screenshot failed, is DISPLAY set? "
                                     f"{e!r} {detail.decode(errors='replace').strip()}")
            return False

    def stop(self):
        """Finish writing everything queued. Returns a one-line summary."""
        with self._lock:
            self._accepting = False
        self._stop_desktop.set()
        self._desktop_thread.join()
        for _ in self._writers:
            self._queue.put(None)
        for thread in self._writers:
            thread.join()
        self._camera_file.close()
        self._controller_file.close()
        return (f"{self.directory}: {self._next_frame} frames, {self.dropped} dropped, "
                f"{self.write_failures} failed writes, {self.screenshots} desktop screenshots, "
                f"{self.screenshot_failures} failed screenshots")
