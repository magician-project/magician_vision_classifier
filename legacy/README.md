# Legacy

Retired code, kept for reference. Nothing here is on a supported path — these
files are not imported by the training, inference, or analysis pipelines and are
not expected to run as-is.

| File | Why it is here |
|------|----------------|
| `client.py` | Keras-era pygame viewer. Imports `evaluate`, a module deleted long ago, so it cannot start. |
| `dumpKerasDataset.py` | Keras-era `.pnm` → `keras_dataset/` tile dumper. Still called by `scripts/createMergedDataset.sh`, which is itself Keras-era (it references the removed `trainClassifierKeras.py` and `evaluate.py`). |
| `EnsembleClassifierOG.py` | First cut of the two-stage ensemble. Superseded by `EnsembleClassifier.py`; nothing imports it. |
| `evaluateTorch.py` | Keras-era single-frame heatmap CLI, broken twice over: it imports the renamed `trainClassifierTorch`, and its single-image branch calls `runSingle` with 6 of 7 required arguments. 18 of its 25 functions are duplicated in `classifierPnm.py`; the other 7 are numpy-era predecessors of functions that live there. Use `evaluateClassifierNew.py` (checkpoint evaluation) or `classifierPnm.runSingle` (single-frame heatmap) instead. |
| `ModelAnalysisOld.py` | Predecessor of the current analysis tool. Superseded by `ModelAnalysis.py`, which orchestrates `ModelAnalysisEvaluate.py` + `ModelAnalysisReport.py`. |
| `principleComponentAnalysis.py` | 2022 body-pose PCA library. Unrelated to the polarization classifier; nothing in this repository imports it. |
| `randomCameraPoses.py` | Scratch generator that writes a fixed `camera_poses.csv` ring. Test input for `integrator3D.py`. |
| `randomHeatmap.py` | Scratch generator that writes synthetic `heatmap_*.png` blobs. Test input for `integrator3D.py`. |

The current equivalents live in the repository root; see the main `README.md`.
