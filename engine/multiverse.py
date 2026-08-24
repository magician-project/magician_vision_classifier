"""Multiverse engine — template for a third-party classifier design.

This is a RUNNABLE SKELETON: as shipped it runs end-to-end with
``python3 -m mvc.inference.live_torch --model multiverse``, powered by the tiny
placeholder network below.  Replace the placeholder with the real design by
following the numbered comments.

The plug-in contract lives in ``mvc/inference/engine_base.py`` — in short:
  * ``build_classifier(cfg_path)`` is the entry point the loader imports.
  * ``load_model()`` returns a module mapping uint8 (N, 4, tile, tile) tiles to
    logits (N, num_classes)  [tile input mode].
  * ``run_whole_image(frame)`` classifies the entire frame and returns
    (points, classes, confidences)  [whole-image input mode].
  * The input mode is switched in ``engine/multiverse.json`` under
    ``"input": {"mode": "tiles" | "whole_image", "tile_size": X, "step": Z}``.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvc.inference.engine_base import EngineClassifier


# --------------------------------------------------------------------------
# STEP 1 (REPLACE ME): the multiverse network itself.
# --------------------------------------------------------------------------
# Replace this class with the real multiverse architecture.  Its forward MUST
# satisfy the tile-model contract: uint8 (N, 4, tile_size, tile_size) in,
# logits (N, num_classes) out.  classifier_pnm feeds uint8 tiles by design, so
# dequantise inside forward (x.float() / 255.0) — keep that even when you
# replace the body.  num_classes and the tile size come from the engine's
# .json config; read them (and any arch hyperparameters) from the config
# inside load_model() below, not from hardcoded values.
# --------------------------------------------------------------------------
class _MultiversePlaceholder(nn.Module):
    """Tiny stand-in network so the engine runs before the real design lands.
    (Two stages + pooling to show where a deeper architecture goes.)"""

    def __init__(self, num_classes, channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.out = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.out(x)


class MultiverseEngine(EngineClassifier):
    """Multiverse plug-in.  All runtime plumbing (config loading, tiling,
    gating, heatmap, responses, live_* interface) is inherited from
    EngineClassifier — only the two hooks below are multiverse-specific."""

    def load_model(self):
        """HOOK — implement the real multiverse network here.

        Instantiate your architecture and load its weights.  Everything this
        engine needs at build time is in ``self.cfg``; engine-specific settings
        belong under the ``"multiverse"`` key of engine/multiverse.json, e.g.::

            cfg = self.cfg.get("multiverse", {})
            weights = cfg.get("model_weights")          # e.g. "engine/multiverse_weights.pt"
            model = _YourRealNetwork(num_classes=len(self.classes), ...)
            if weights:
                model.load_state_dict(torch.load(weights, map_location=self.device,
                                                 weights_only=False))

        The returned module MUST map uint8 (N, 4, tile_size, tile_size) to
        logits (N, len(self.classes)) for tile mode (see STEP 1 above); the
        base class moves it to self.device and calls .eval() itself.
        """
        # TODO(multiverse): replace with the real network + weights (see above).
        return _MultiversePlaceholder(len(self.classes))

    def run_whole_image(self, frame):
        """HOOK — implement the real multiverse whole-image design here.

        Only called when engine/multiverse.json sets ``"input": {"mode":
        "whole_image"}``.  ``frame`` is the raw numpy HxWx4 uint8 RGBA frame
        from shared memory (runs under torch.inference_mode()).  Return::

            (points, classes, confidences)

        points      = list of (x, y) pixel coordinates of detection centres,
        classes     = class-name strings taken from self.classes,
        confidences = float score per detection, one entry per detection.

        The default below is a working stand-in: it downscales the whole frame
        to one tile, classifies it with the tile model, and — if the top class
        is not the clean class — reports a single detection at the frame
        centre.  Replace it with the real design (e.g. a detector or segmenter
        returning per-defect points/boxes).  For detections you want to
        suppress, simply omit them from the lists — the gate is a tile-mode
        concept and does not apply here.
        """
        # TODO(multiverse): replace with the real whole-image design (see above).
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.tile_size, self.tile_size),
                           interpolation=cv2.INTER_AREA)
        x = torch.as_tensor(small, device=self.device, dtype=torch.uint8)
        x = x.permute(2, 0, 1).unsqueeze(0)
        probs = F.softmax(self.model(x).float(), dim=1)
        conf, cls = torch.max(probs, dim=1)
        cls, conf = int(cls.item()), float(conf.item())
        if self.classes[cls].lower() in ("class_clean", "clean"):
            return [], [], []
        return [(w // 2, h // 2)], [self.classes[cls]], [conf]


def build_classifier(cfg_path):
    """Entry point the loader (mvc.inference.engine_base.build_engine) imports
    by name — KEEP this name and signature.  ``cfg_path`` is the engine's .json
    config (default engine/multiverse.json; override with --model-config)."""
    return MultiverseEngine(cfg_path, name="multiverse")
