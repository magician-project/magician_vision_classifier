"""Plug-in mechanism for third-party classifier designs ("engines").

How a plug-in works
-------------------
Drop a module ``engine/<name>.py`` next to this package (the top-level
``engine/`` directory) that defines ``build_classifier(cfg_path)`` returning an
:class:`EngineClassifier` subclass.  ``--model <name>`` on live_torch.py /
live_torch_ros.py then builds that engine instead of downloading a
``<name>.pth`` from the model server — no other change needed.  An engine name
must be a valid, non-keyword Python identifier and the file must exist;
anything else falls through to the normal built-in model path.

Engine .json config schema
--------------------------
The engine reads its own configuration file (path passed to ``build_classifier``
/ the ``--model-config`` flag; default ``engine/<name>.json``)::

    {
      "classes": ["class_clean", "class_bubble"],   # REQUIRED; label order = output order.
                                                    # The clean class MUST be named
                                                    # "class_clean" or "clean" (the gate and
                                                    # the template whole-image default rely on it).
      "input": {
        "mode": "tiles",          # "tiles" (XxY tile_size grid, Z step) or "whole_image"
        "tile_size": 48,          # tile mode: square tile edge in px
        "step": 16,               # tile mode: tile grid stride in px
        "resize": null            # whole_image mode only, optional: fixed input size
                                  #   [W, H] for designs that need one; absent/None means
                                  #   the model receives the ENTIRE frame at full camera
                                  #   resolution (e.g. 1024x1224x4 uint8 RGBA).
      },
      "gate": {                   # optional; same semantics as a built-in model's gate
        "mode": "defect_mass",    #   "defect_mass" | "max_prob" | "off"
        "threshold": 0.655,       #   gate score cut; 0.0 leaves the gate OFF
        "assign_best_defect_class": true
      },
      "<engine>": { }             # free-form engine-specific settings (weights path,
                                  # arch params, ...) read by your load_model().
    }

Note on step: in live mode the loop overwrites ``classifier.step`` from the
runtime/preset every frame (same philosophy as the preset gate overriding the
model json), so the config's ``input.step`` is the engine's initial value and
standalone default, not a live-mode override.

Tile-mode model contract
------------------------
``load_model()`` must return a ``torch.nn.Module`` whose ``forward`` maps uint8
``(N, 4, tile_size, tile_size)`` tiles to logits ``(N, len(classes))``.
classifier_pnm feeds uint8 by design, so dequantise inside the module
(``x.float() / 255.0``) — the entire existing pipeline (tiling, gating, majority
vote, erosion, heatmap, responses) is then reused as-is via ``runSingle``.
"""

import importlib
import json
import keyword
import os
import time

import cv2
import numpy as np
import torch

from mvc.paths import repo_root
from mvc.inference.classifier_pnm import (
    GATE_DEFECT_MASS,
    ClassifierPnm,          # used only for the unbound add_legend() call
    getNDifferentColors,
    runSingle,
)

ENGINE_DIR = os.path.join(repo_root(), "engine")


def is_engine_name(name):
    """True when `name` names a plug-in: a valid, non-keyword Python identifier
    with ``engine/<name>.py`` on disk.  Non-identifiers and names without a
    matching module are not engines, so ``--model`` falls through to the
    normal download path."""
    if not isinstance(name, str) or not name:
        return False
    if not name.isidentifier() or keyword.iskeyword(name):
        return False
    return os.path.isfile(os.path.join(ENGINE_DIR, name + ".py"))


def build_engine(name, cfg_path=None):
    """Import ``engine.<name>`` and build its classifier via its module-level
    ``build_classifier(cfg_path)``.  ``cfg_path`` defaults to
    ``engine/<name>.json``.

    Raises ValueError (not an engine), FileNotFoundError (missing config),
    RuntimeError (module import failed), or TypeError (no ``build_classifier``
    or it returned a non-EngineClassifier).  Callers (the live_* scripts) catch
    these and shut down cleanly.
    """
    if not is_engine_name(name):
        raise ValueError(f"'{name}' is not an engine: no engine/{name}.py")
    if cfg_path is None:
        cfg_path = os.path.join(ENGINE_DIR, name + ".json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Engine config not found: {cfg_path}")
    try:
        mod = importlib.import_module(f"engine.{name}")
    except Exception as e:
        raise RuntimeError(f"importing engine/{name}.py failed: {e!r}") from e
    build = getattr(mod, "build_classifier", None)
    if not callable(build):
        raise TypeError(f"engine/{name}.py must define build_classifier(cfg_path)")
    obj = build(cfg_path)
    if not isinstance(obj, EngineClassifier):
        raise TypeError(f"engine/{name}.py: build_classifier() must return an "
                        f"EngineClassifier subclass, got {type(obj).__name__}")
    return obj


class EngineClassifier:
    """Base class for engine plug-ins; satisfies the live_* runtime contract
    (the attributes and ``forward()`` the loops touch on a ClassifierPnm).

    Subclass contract:
      * REQUIRED hook:  ``load_model(self)`` -> torch.nn.Module mapping uint8
        ``(N, 4, tile_size, tile_size)`` -> logits ``(N, len(classes))``.
      * OPTIONAL hook:  ``run_whole_image(self, frame)`` — see its docstring;
        only used when ``input.mode == "whole_image"``.
      * OPTIONAL:       override ``forward()`` entirely for a fully custom
        pipeline that does not fit either input mode.
    """

    def __init__(self, cfg_path, name, step=16):
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Engine config not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            self.cfg = json.load(f)

        if "classes" not in self.cfg:
            raise ValueError(f"Engine config {cfg_path} is missing the required "
                             f"'classes' list (label order = model output order) "
                             f"— see mvc/inference/engine_base.py for the schema")
        self.classes = list(self.cfg["classes"])

        input_cfg = self.cfg.get("input", {})
        self.input_mode = input_cfg.get("mode", "tiles")
        if self.input_mode not in ("tiles", "whole_image"):
            raise ValueError(f"Engine config {cfg_path}: input.mode must be "
                             f"'tiles' or 'whole_image', got {self.input_mode!r}")
        self.tile_size = int(input_cfg.get("tile_size",
                                           self.cfg.get("hparams", {}).get("tile_size", 64)))
        self.step = int(input_cfg.get("step", step))
        # Whole-image mode only: fixed input size [W, H] for designs that need one.
        # Absent/None = the model receives the entire frame at full resolution.
        self.whole_image_resize = input_cfg.get("resize")
        if self.whole_image_resize is not None:
            try:
                self.whole_image_resize = (int(self.whole_image_resize[0]),
                                           int(self.whole_image_resize[1]))
            except (TypeError, ValueError, IndexError) as e:
                raise ValueError(f"Engine config {cfg_path}: input.resize must be "
                                 f"[W, H] in pixels, got {input_cfg.get('resize')!r}") from e

        # Tile decision gate — same semantics as ClassifierPnm.__init__ (see its
        # "gate" comment in classifier_pnm.py). 0.0 leaves the gate OFF (plain argmax).
        gate_cfg = self.cfg.get("gate", {}) if isinstance(self.cfg, dict) else {}
        self.gateMode                = gate_cfg.get("mode", GATE_DEFECT_MASS)
        self.maxProbabilityThreshold = float(gate_cfg.get("threshold", 0.0))
        self.assignBestDefectClass   = bool(gate_cfg.get("assign_best_defect_class", True))

        self.name       = name
        self.model_path = os.path.join(ENGINE_DIR, name + ".py")  # read by handle_key's basename()[:-4]
        self.hz         = 0.0
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_colors = getNDifferentColors(len(self.classes))

        print(f"[engine:{self.name}] classes={self.classes} mode={self.input_mode} "
              f"tile_size={self.tile_size} step={self.step}")
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()
        print(self.format_threshold_tradeoff(self.maxProbabilityThreshold))

    # ------------------------------------------------------------------ hooks
    def load_model(self):
        """HOOK (required, no base default).  Return the torch module whose
        forward maps uint8 (N, 4, tile_size, tile_size) tiles to logits
        (N, len(self.classes)).  Dequantise inside the module
        (x.float() / 255.0) — classifier_pnm feeds uint8 tiles by design.
        Architecture/weights settings come from self.cfg's engine section
        (e.g. cfg["yongatek"])."""
        raise NotImplementedError(f"{type(self).__name__} must implement load_model()")

    def run_whole_image(self, frame):
        """HOOK (optional).  Whole-image input mode.  `frame` is the ENTIRE
        camera frame at full resolution — a raw numpy HxWx4 uint8 RGBA array
        from shared memory (e.g. 1024x1224x4).  Return
            (points, classes, confidences)
        where points = list of (x, y) pixel coordinates of detection centres
        IN THE ORIGINAL FRAME'S COORDINATES, classes = class-name strings from
        self.classes, confidences = floats (one entry per detection).  If the
        design needs a fixed input size, resize internally (or use the config's
        ``input.resize``) and scale your points back to the original frame.
        Base default: no detections (safe for engines that only implement tile
        mode)."""
        return [], [], []

    # ------------------------------------------------------- runtime plumbing
    def forward(self, image, majorityVote=False, legend=True, erosion_kernel=0, erosion_threshold=0):
        """Runtime entry point, called every frame by live_*.  Returns
        (heatmap, occupancy, responses) with the exact contract the loops
        consume (responses keys: points/classes/classIDs/confidences/
        background_avg_prob).  Tile mode delegates to classifier_pnm.runSingle
        — the full existing pipeline.  Whole-image mode calls
        _forward_whole_image.  Sets self.hz."""
        start = time.time()

        if self.input_mode == "tiles":
            heatmap, occupancy, responses = runSingle(
                image, self.model, self.device,
                self.classes, self.class_colors,
                self.tile_size, self.step,
                majorityVote=majorityVote,
                maxProbabilityThreshold=self.maxProbabilityThreshold,
                gateMode=self.gateMode,
                assignBestDefectClass=self.assignBestDefectClass,
                erosion_kernel=erosion_kernel,
                erosion_threshold=erosion_threshold,
                name=self.name)
        else:
            heatmap, occupancy, responses = self._forward_whole_image(image)

        if legend:
            heatmap = ClassifierPnm.add_legend(self, heatmap)

        self.hz = 1 / (time.time() - start + 0.0001)
        return heatmap, occupancy, responses

    def _forward_whole_image(self, image):
        """Turn run_whole_image() into the runtime (heatmap, occupancy, responses)
        triple.  `image` is the raw numpy HxWx4 uint8 RGBA frame.  Erosion and
        majority vote are tile-grid concepts and are ignored here; implement
        your own suppression inside run_whole_image() if you need it."""
        with torch.inference_mode():
            points, classes, confidences = self.run_whole_image(image)

        heatmap = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        radius = max(4, self.tile_size // 4)
        for (x, y), cls in zip(points, classes):
            class_id = self.classes.index(cls)
            cv2.circle(heatmap, (int(x), int(y)), radius,
                       self.class_colors[class_id], 2)

        # No tile grid exists in whole-image mode; the live loops unpack
        # occupancy but never read it.
        occupancy = np.zeros((1, 1), dtype=np.uint8)
        responses = {"points": list(points),
                     "classes": list(classes),
                     "classIDs": [self.classes.index(c) for c in classes],
                     "confidences": [float(c) for c in confidences],
                     "background_avg_prob": 0.0}
        return heatmap, occupancy, responses

    def format_threshold_tradeoff(self, threshold, gateMode=None):
        """Human-readable gate summary, mirroring ClassifierPnm's shape.
        Engines have no threshold curve, so the trade-off is stated as unknown.
        Handles threshold=None (the runtime's "cleared" default)."""
        at = "model default" if threshold is None else f"{float(threshold):.3f}"
        return (f"gate {gateMode or self.gateMode} @ {at} "
                f"(engine plug-in — no threshold curve, trade-off unknown)")

    def reload_model(self, directoryPath, name):
        """Hot-swap stub: engine plug-ins are module-based and have no .pth to
        reload.  Pressing 'r' with an engine active logs this and the process
        keeps running (set_model's existing failure path reports it)."""
        print(f"Hot-swap is not supported for engine plug-ins — restart with "
              f"--model {name} to switch away from {self.name}")
        return False
