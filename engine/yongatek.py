"""Yongatek engine — YOLO26m whole-image defect detector.

Runs with ``python3 -m mvc.inference.live_torch --model yongatek`` (or
live_torch_ros).  The design is a whole-image object detector, not a tile
classifier:

  1. The polarization frame (HxWx4 RGBA, or a raw single-channel mosaic which
     is debayered first) is collapsed to a single intensity image — the mean of
     the four polarization channels, replicated to 3 channels.  This is exactly
     the preprocessing the training set was generated with
     (``convert_intensity``: 2x2 mosaic average, +brightness, clamp, uint8).
  2. A YOLO26m detector (ultralytics, trained at imgsz 640) finds defects.
  3. Box centres are returned as detection points; the boxes themselves are
     drawn on the heatmap.

Settings live under the ``"yongatek"`` key of engine/yongatek.json.  Requires
``ultralytics==8.4.71`` (the version the checkpoint was produced with).

The plug-in contract lives in ``mvc/inference/engine_base.py``.
"""

import os
import sys

import cv2
import numpy as np
import torch

# Run as a script (python3 engine/yongatek.py): put the repo root on sys.path so
# the mvc package resolves.  No-op when imported as engine.yongatek.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvc.core.read_data import readPolarPNMToRGBALive
from mvc.inference.engine_base import EngineClassifier
from mvc.paths import repo_root


# Keyword arguments EngineClassifier.forward accepts; anything else the live
# loops pass (e.g. live_torch's ``stats=``) is handled/dropped by our override.
_BASE_FORWARD_KWARGS = ("majorityVote", "legend", "erosion_kernel",
                        "erosion_threshold", "log", "majority_window")


class YongatekEngine(EngineClassifier):
    """Yongatek plug-in.  Runtime plumbing (config loading, legend, live_*
    interface) is inherited from EngineClassifier; the detector, the intensity
    preprocessing and the box drawing are yongatek-specific."""

    def load_model(self):
        """Load the YOLO detector and validate its class mapping against the
        engine config.  Returns the underlying torch module so the base class's
        ``.to(device)`` / ``.eval()`` calls have something to act on; inference
        itself goes through ``self.yolo.predict``."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("engine 'yongatek' needs ultralytics: "
                              "pip install ultralytics==8.4.71") from e

        cfg = self.cfg.get("yongatek", {})
        weights = cfg.get("model_weights", "engine/yongatek_weights.pt")
        if not os.path.isabs(weights):
            weights = os.path.join(repo_root(), weights)
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"[engine:{self.name}] weights not found: {weights}")

        self.imgsz    = int(cfg.get("imgsz", 640))
        self.conf     = float(cfg.get("conf", 0.25))
        self.iou      = float(cfg.get("iou", 0.7))
        self.max_det  = int(cfg.get("max_det", 300))
        self.half     = bool(cfg.get("half", True)) and self.device == "cuda"
        self.class_conf = {k: float(v) for k, v in cfg.get("class_conf", {}).items()}
        intensity_cfg   = cfg.get("intensity", {})
        self.brightness = float(intensity_cfg.get("brightness", 0))
        self.contrast   = float(intensity_cfg.get("contrast", 0))

        self.yolo = YOLO(weights, task="detect")

        # YOLO class id -> engine class name.  Validated up front: publishing a
        # wrong class name is worse than refusing to start.
        class_map = {int(k): v for k, v in cfg.get("yolo_class_map", {}).items()}
        if set(class_map) != set(self.yolo.names):
            raise ValueError(f"[engine:{self.name}] yolo_class_map ids {sorted(class_map)} "
                             f"do not match the model's classes {self.yolo.names}")
        unknown = [v for v in class_map.values() if v not in self.classes]
        if unknown:
            raise ValueError(f"[engine:{self.name}] yolo_class_map targets {unknown} "
                             f"are not in 'classes' {self.classes}")
        self.class_map = class_map

        publish = cfg.get("publish_classes")
        self.publish_classes = set(class_map.values() if publish is None else publish)

        # Lowest per-class threshold goes to YOLO; per-class cuts are applied after.
        self.predict_conf = min([self.conf] + list(self.class_conf.values()))
        self._last_boxes = []

        if cfg.get("warmup", True):
            self._predict(np.zeros((1024, 1224, 3), dtype=np.uint8))

        return self.yolo.model

    def forward(self, image, **kwargs):
        """Runtime entry point.  Wraps EngineClassifier.forward to (a) debayer a
        raw single-channel mosaic frame to RGBA (the base whole-image path
        assumes RGBA) and (b) accept live_torch's ``stats=`` dict, which the
        base signature does not take, and fill it for the status line."""
        image = readPolarPNMToRGBALive(image)
        stats = kwargs.pop("stats", None)
        out = super().forward(image, **{k: v for k, v in kwargs.items()
                                         if k in _BASE_FORWARD_KWARGS})
        if stats is not None:
            stats.update({"name": self.name, "step": self.step, "hz": self.hz,
                          "tiles": 0, "tiles_per_sec": 0.0})
        return out

    def _to_intensity(self, rgba):
        """HxWx4 uint8 RGBA -> HxWx3 uint8 intensity, identical to the training
        preprocessing (convert_intensity): float mean of the four polarization
        channels, +10*brightness, clamp, uint8 truncation, replicate to 3ch."""
        x = torch.from_numpy(rgba).to(self.device, non_blocking=True).float()
        intensity = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2] + x[:, :, 3]) / 4.0
        intensity = torch.clamp(intensity + 10 * self.brightness, 0, 255).to(torch.uint8)
        img = intensity.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy()
        if self.contrast != 0:
            # NOTE: must match the adjust_contrast used when the training set was
            # generated; trained with contrast=0, so this path is off by default.
            img = cv2.convertScaleAbs(img, alpha=1.0 + self.contrast / 10, beta=0)
        return img

    def _predict(self, img):
        return self.yolo.predict(img, imgsz=self.imgsz, conf=self.predict_conf,
                                 iou=self.iou, max_det=self.max_det, half=self.half,
                                 device=self.device, verbose=False)[0]

    def run_whole_image(self, frame):
        """Detect defects on the whole frame.  ``frame`` is HxWx4 uint8 RGBA;
        returns (points, classes, confidences) with points = box centres in the
        frame's pixel coordinates (the intensity image has the frame's HxW and
        ultralytics maps boxes back to the input size)."""
        # Inference runs on the intensity image only: the 4 polarization channels
        # are averaged into one grey image (see _to_intensity) before YOLO sees it.
        intensity = self._to_intensity(frame)
        result = self._predict(intensity)
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        points, classes, confidences, kept_boxes = [], [], [], []
        for (x1, y1, x2, y2), cid, conf in zip(xyxy, cls_ids, confs):
            name = self.class_map[int(cid)]
            if name not in self.publish_classes:
                continue
            if conf < self.class_conf.get(name, self.conf):
                continue
            points.append((int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))))
            classes.append(name)
            confidences.append(float(conf))
            kept_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        self._last_boxes = kept_boxes
        return points, classes, confidences

    def _forward_whole_image(self, image):
        """Same contract as EngineClassifier._forward_whole_image, but draws the
        detection boxes with class/confidence labels instead of centre circles."""
        with torch.inference_mode():
            points, classes, confidences = self.run_whole_image(image)

        heatmap = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        for (x1, y1, x2, y2), cls, conf in zip(self._last_boxes, classes, confidences):
            color = self.class_colors[self.classes.index(cls)]
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, 2)
            label = f"{cls.removeprefix('class_')} {conf:.2f}"
            cv2.putText(heatmap, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        occupancy = np.zeros((1, 1), dtype=np.uint8)
        responses = {"points": list(points),
                     "classes": list(classes),
                     "classIDs": [self.classes.index(c) for c in classes],
                     "confidences": [float(c) for c in confidences],
                     "background_avg_prob": 0.0}
        return heatmap, occupancy, responses


def build_classifier(cfg_path):
    """Entry point the loader (mvc.inference.engine_base.build_engine) imports
    by name — KEEP this name and signature.  ``cfg_path`` is the engine's .json
    config (default engine/yongatek.json; override with --model-config)."""
    return YongatekEngine(cfg_path, name="yongatek")


if __name__ == "__main__":
    # Standalone test on a single raw polarization frame:
    #   python3 engine/yongatek.py [image.pnm] [--config engine/yongatek.json] [--out out.png] [--show]
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Run the yongatek engine on one frame")
    parser.add_argument("image", nargs="?",
                        default="/home/cvteam/workplace/magician/magician_grabber/"
                                "day3_neg_dent_6_1/colorFrame_0_00179.pnm",
                        help="raw polarization mosaic (.pnm) or HxWx4 RGBA image")
    parser.add_argument("--config", default=os.path.join(repo_root(), "engine", "yongatek.json"))
    parser.add_argument("--out", default="yongatek_result.png", help="where to save the drawn result")
    parser.add_argument("--show", action="store_true", help="also open an OpenCV window")
    args = parser.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if frame is None:
        sys.exit(f"Could not read image: {args.image}")

    engine = build_classifier(args.config)
    start = time.time()
    heatmap, _, responses = engine.forward(frame, log=False)
    print(f"inference {1000 * (time.time() - start):.1f} ms, {len(responses['points'])} detection(s)")
    for (x, y), cls, conf, box in zip(responses["points"], responses["classes"],
                                      responses["confidences"], engine._last_boxes):
        print(f"  {cls:22s} conf={conf:.3f} centre=({x}, {y}) box={box}")

    cv2.imwrite(args.out, heatmap)
    print(f"saved {args.out}")
    if args.show:
        cv2.imshow("yongatek", heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
