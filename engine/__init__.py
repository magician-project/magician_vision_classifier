"""Third-party classifier plug-ins.

Drop ``engine/<name>.py`` here (defining ``build_classifier(cfg_path)`` and a
class derived from ``mvc.inference.engine_base.EngineClassifier``) and
``--model <name>`` on live_torch.py / live_torch_ros.py picks it up — no other
change needed.  The engine's own config lives at ``engine/<name>.json``
(override with ``--model-config PATH``).

See ``mvc/inference/engine_base.py`` for the full plug-in contract: the
required/optional hooks, the tile-mode model contract, and the .json schema.
``engine/yongatek.py`` and ``engine/multiverse.py`` are working templates to
copy from.
"""
