# Magician Vision Classifier

<<<<<<< HEAD
**Authors:** Ammar Qammaz, Nikos Vasilikopoulos
**Institution:** Foundation for Research and Technology – Hellas (FORTH), Institute of Computer Science, Greece
**Copyright:** © 2025 FORTH, Computer Science Department, Greece

This work has been developed within the context of research funded by the European Union's Horizon 2020 research and innovation programme.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Related Projects](#related-projects)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Training](#training)
7. [Inference](#inference)
8. [ROS2 Integration](#ros2-integration)
9. [Model Analysis and Evaluation](#model-analysis-and-evaluation)
10. [Shared Memory System](#shared-memory-system)
11. [Project Structure](#project-structure)
12. [References](#references)
13. [Contact](#contact)

---

## Abstract

The Magician Vision Classifier System provides a complete pipeline for training, evaluating, and deploying real-time, vision-based defect classifiers on a ROS2-enabled robotic platform. The system is designed around a polarization camera that captures four-channel images (0°, 45°, 90°, 135° polarization angles), which encode surface properties that are not visible in conventional RGB imagery. This makes it particularly effective for detecting subtle manufacturing defects on specular or transparent surfaces.

The core classifier uses a tile-based approach: each frame is decomposed into overlapping patches, which are independently classified by a convolutional neural network. A two-stage ensemble architecture enables real-time performance: a fast binary prefilter discards clean tiles, while a pool of pretrained backbones votes only on the remaining candidate defect regions. Results are spatially smoothed via 2D majority voting before being published as ROS2 messages with optional depth data from fused laser sensors.

The system supports sixteen pretrained vision backbones (including ResNet, ResNeXt, ConvNeXt, EfficientNet-V2, Swin-V2, and RegNet variants) alongside a lightweight custom CNN, all adapted for four-channel input. Training is managed via PyTorch Lightning with support for focal loss, false-clean penalization, balanced batch sampling, and optional Weights & Biases or TensorBoard logging.

---

## Related Projects

This repository is one component of a larger acquisition-to-inference pipeline. The two upstream repositories are:

| Repository | Role |
|------------|------|
| [magician_grabber](https://github.com/magician-project/magician_grabber) | Multi-modal data acquisition node. Drives the GigE polarization camera, captures raw `.pnm`/`.png` frames, streams them via POSIX shared memory, and publishes sensor data over ROS2. This is the live frame source consumed by `liveClassifierTorchROS.py` at inference time. |
| [magician_grabber_annotator](https://github.com/magician-project/magician_grabber_annotator) | Desktop GUI for annotating raw polarization captures. Annotators label defect class and severity per frame; the tool exports training-ready tile datasets (RGBA PNGs organized by class directory) that feed directly into the training pipeline described below. |

The full pipeline is:

```
magician_grabber          magician_grabber_annotator      magician_vision_classifier
─────────────────         ──────────────────────────      ──────────────────────────
Capture raw frames   -->  Annotate & export tiles    -->  Train classifier
Stream via shm                                            Run inference (consumes shm)
=======
**Authors:** Ammar Qammaz, Nikos Vasilikopoulos  
**Copyright:** © 2025 Foundation of Research and Technology – Hellas (FORTH), Computer Science Department, Greece  
**License:** See `license.txt`

---

## Overview

The **Magician Vision Classifier** is a complete pipeline for training, evaluating, and deploying a real-time tile-based vision classifier within a ROS 2 ecosystem. It is designed for industrial surface-defect detection on a conveyor or robotic inspection system.

The system operates by dividing each incoming frame into a grid of small tiles, classifying each tile independently with a deep neural network, and publishing the results as structured ROS 2 messages. Both single-model and two-stage ensemble inference modes are supported.

---

## Architecture

```
Camera / Shared Memory
        │
        ▼
 SharedMemoryManager          ← reads RGBA frames from shared memory
        │
        ▼
 ClassifierPnm  ──or──  EnsembleClassifierPnm
 (single model)           (binary stage + multi-model vote)
        │
        ▼
 Tile classification  →  Heatmap overlay  →  Detection responses
        │
        ├──► /detections              (Detection per defect tile)
        ├──► /detections_m            (DetectionM with interpolated depth)
        ├──► /markers                 (ArUco marker poses)
        └──► /background_activations  (clean-tile confidence)
>>>>>>> b25a3bdd9acd41e79c030f0d1923dfcc54f52d80
```

---

<<<<<<< HEAD
## System Architecture

```
  +-------------------+     +----------------+     +---------------------+
  | Polarization      |     | Debayering     |     | Shared Memory       |
  | Camera            |-->  | (4-ch RGBA)    |-->  | (zero-copy mmap)    |
  | 0°,45°,90°,135°  |     | readData.py    |     | libSharedMemory     |
  +-------------------+     +----------------+     +----------+----------+
                                                              |
                                                              v
  +-------------------+     +----------------+     +---------------------+
  | ROS2 Publications | <-- | Spatial        | <-- | Stage 2 Ensemble    |
  | Detection msgs    |     | Smoothing      |     | (multi-model vote)  |
  +-------------------+     +----------------+     +----------+----------+
                                                              ^
                                                              |
                                                  +-----------+-----------+
                                                  | Stage 1 Prefilter      |
                                                  | (binary: clean/defect) |
                                                  +-----------------------+
```

**Pipeline stages:**

1. **Capture:** A polarization camera with a Bayer-like filter acquires images encoding four polarization angles.
2. **Debayering:** `readData.py` extracts the four channels into a 4-channel RGBA image. Optional derived channels (AoLP, DoLP) can be computed.
3. **Shared Memory Transfer:** Frames are passed to the classifier via POSIX shared memory (`libSharedMemoryVideoBuffers.so`), avoiding costly IPC serialization.
4. **Tile Extraction:** Each frame is tiled into overlapping patches (e.g., 48×48 pixels, step 18) using GPU-accelerated `unfold` operations.
5. **Stage 1 Prefilter:** A lightweight binary classifier rapidly identifies clean tiles, which are discarded.
6. **Stage 2 Ensemble:** Multiple backbone models run in parallel (via async CUDA streams) on the remaining non-clean tile subset, producing a majority vote.
7. **Spatial Smoothing:** A 2D majority-vote filter suppresses isolated misclassifications.
8. **Output:** Detections are published as ROS2 messages with bounding boxes, class labels, confidence scores, and optional depth data fused from laser sensors via inverse-distance-weighted interpolation.

---

## Installation

### Prerequisites

- Linux (Ubuntu 22.04 or later)
- NVIDIA GPU with CUDA support (for GPU inference)
- ROS2 Rolling (for the ROS2 node)
- Python 3.10+

### Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The main dependencies are: `torch`, `torchvision`, `pytorch-lightning`, `torchmetrics`, `opencv-python`, `numpy`, `h5py`, `wandb`, `matplotlib`, `seaborn`, `numba`.

### ROS2 Workspace

Place this repository inside a ROS2 workspace `src/` directory and build:

```bash
cd /path/to/ros2/workspace
colcon build --packages-select magician_vision_classifier
source install/setup.bash
```

### Shared Memory Library

The `SharedMemoryVideoBuffers/` subdirectory contains the C library source for zero-copy frame transfer. Build it according to the instructions in that directory, ensuring `libSharedMemoryVideoBuffers.so` is placed in this repository root.

---

## Dataset Preparation

### Image Format

The training dataset consists of RGBA PNG images where each channel corresponds to a polarization angle (R = 0°, G = 45°, B = 90°, A = 135°). Datasets are typically produced by the [magician_grabber_annotator](https://github.com/magician-project/magician_grabber_annotator), which exports annotated tiles in this format directly. Raw polarization images can also be split into channels manually using `splitChannels.py`.

### Directory Structure

Organize images in subdirectories named by class:

```
dataset/
  ClassA/
    img001.png
    img002.png
  ClassB/
    img001.png
  Clean/
    img001.png
```

### HDF5 Conversion

For large datasets, convert PNG images to HDF5 format for faster I/O during training:

```bash
python3 DatasetConverter.py <config.json>
```

This produces a `dataset.h5` file that the training script can load directly. The HDF5 format stores data as uint8 (four times smaller than float32) and supports per-sample JSON metadata.

### Utility Scripts

- `scripts/createBinaryDataset.sh` — create a binary clean/defect dataset from multi-class data (for Stage 1 training)
- `scripts/createMergedDataset.sh` — merge multiple dataset directories into one

=======
## Requirements

### System

- ROS 2 (Rolling or compatible distro)
- Python 3.10+
- CUDA-capable GPU (recommended)

### Python dependencies

Create a virtual environment that inherits ROS 2 system packages:

```bash
sudo apt install ros-rolling-example-interfaces
source /opt/ros/rolling/setup.bash

python3 -m venv venv --system-site-packages
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install empy lark
```

### Shared memory library

The inference node reads frames from a shared memory ring buffer. The library must be present at runtime:

```bash
git clone https://github.com/AmmarkoV/SharedMemoryVideoBuffers
cd SharedMemoryVideoBuffers && make && cd ..
ln -s SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so .
SharedMemoryVideoBuffers/server --nokb &
```

---

## Repository Structure

```
├── liveClassifierTorchROS.py            # ROS 2 inference node (main entry point)
├── liveClassifierTorch.py               # ClassifierPnm wrapper (single model)
├── EnsembleClassifier.py                # EnsembleClassifierPnm (two-stage)
├── trainMagicianVisionClassifierTorch.py # Model definitions (PyTorch Lightning)
├── trainClassifierTorch.py              # Training & evaluation script
├── evaluateTorch.py                     # Standalone evaluation script
├── SharedMemoryManager.py               # Shared memory frame reader
├── readData.py                          # PNM / RGBA image utilities
├── plotTool.py                          # Confusion matrix plotting
├── scripts/                             # Convenience shell scripts (see below)
├── msg/                                 # Custom ROS 2 message definitions
├── srv/                                 # Custom ROS 2 service definitions
├── models/                              # Archived training results (.zip)
├── configs/                             # Example training configuration files
├── data/                                # Saved defect / clean frames
├── snapshots/                           # On-demand snapshots
├── CMakeLists.txt
├── package.xml
├── requirements.txt
└── license.txt
```

>>>>>>> b25a3bdd9acd41e79c030f0d1923dfcc54f52d80
---

## Training

<<<<<<< HEAD
### Quick Start

```bash
python3 trainMagicianVisionClassifierTorch.py configs/stage1.json
```

### Supported Architectures

The training module supports the following backbone networks, all adapted for four-channel RGBA input:

| Model | Parameters | Category |
|-------|-----------|----------|
| `resnet18` | ~11M | Medium |
| `resnext50` | ~27M | Heavy |
| `convnext_tiny` | ~28M | Heavy |
| `efficientnet_v2_s` | ~21M | Heavy |
| `efficientnet_v2_b0` | ~8M | Medium |
| `swin_v2_t` | ~27M | Heavy |
| `regnet_y_800mf` | ~5M | Medium |
| `regnet_y_400mf` | ~2M | Lightweight |
| `mobilenet_v3_small` | ~2.5M | Lightweight |
| `mobilenet_v3_large` | ~11M | Medium |
| `shufflenet_v2_x0_5` | ~1M | Lightweight |
| `shufflenet_v2_x1_0` | ~2M | Lightweight |
| `squeezenet1_1` | ~1M | Lightweight |
| `densenet121` | ~7M | Medium |
| `mnasnet_0_5` | ~1M | Lightweight |
| `mnasnet_1_0` | ~3M | Lightweight |
| `custom` | Configurable | Custom CNN |

### Configuration Files

Training is controlled via JSON configuration files. Example configurations are provided in `configs/`:

| Config | Description |
|--------|-------------|
| `configs/stage1.json` | Binary prefilter (clean vs defect), ResNet18, false-clean penalty |
| `configs/stage1_small.json` | Smaller binary prefilter |
| `configs/stage1_verysmall.json` | Very small binary prefilter |
| `configs/smallmodel.json` | Medium backbones with balanced sampling |
| `configs/bigmodel.json` | Heavy pretrained backbones |
| `configs/verysmallmodel.json` | Lightweight models for edge deployment |

### Configuration Reference

Key configuration fields:

```json
{
  "hparams": {
    "tile_size": 48,
    "batch_size": 64,
    "training_epochs": 25,
    "dropout_rate": 0.25,
    "seed": 42,
    "gradient_clip_value": 1.0,
    "base_channels": 32,
    "final_dense_layer": 128,
    "AoLP": false,
    "DoLP": false,
    "unpolarized": false
  },
  "optimizer": {
    "type": "AdamW",
    "learning_rate": 5e-4
  },
  "dataloader": {
    "validation_split": 0.05,
    "cacheAllDataToRAM": false,
    "num_workers": 12,
    "balanced_sampling": true
  },
  "wandb": {
    "project": "Classifier",
    "name": "experiment_name",
    "use_wandb": false
  },
  "loss": "focal",
  "penalize_false_clean": 0.0,
  "model": "resnet18",
  "devices": 1,
  "selected_classes": [],
  "name": "allclass",
  "training_dataset": "path/to/dataset"
}
```

**Important fields:**

- `loss` — `"focal"` or `"cross_entropy"`. Focal loss is recommended for imbalanced datasets.
- `penalize_false_clean` — penalty weight for misclassifying defects as clean. Use a high value (e.g., 16.0) for the Stage 1 prefilter.
- `balanced_sampling` — ensures every class appears in each batch.
- `cacheAllDataToRAM` — preload entire dataset into RAM for faster training.
- `selected_classes` — filter the dataset to specific classes (empty array means all classes).
- `AoLP`, `DoLP`, `unpolarized` — enable derived polarization channels as additional input features.

### Training Output

After training completes, the following files are produced:

```
<model_name>.pth             — Model checkpoint (PyTorch state dict)
<model_name>.json            — Configuration + training metrics
<model_name>_confusion.json  — Confusion matrix data
models/<timestamp>.zip       — Archived training run
last.pth                     — Symlink to latest model weights
last.json                    — Symlink to latest model config
```

### Training Scripts

- `scripts/fullTraining.sh` — train Stage 1 binary prefilter, then small and big models sequentially
- `scripts/secondaryTraining.sh` — train all backbone variants (heavy, medium, lightweight)

### Metrics

The training module logs the following validation metrics:

- `val_loss`, `val_accuracy`
- `val_precision`, `val_recall`
- `val_auroc` (area under ROC curve)
- Confusion matrix (numeric and plotted)

---

## Inference

### Standalone Mode

Run the classifier outside of ROS2, reading frames from shared memory and displaying a live heatmap:

```bash
python3 liveClassifierTorch.py
```

### ROS2 Mode

Launch the full ROS2 inference node:

```bash
scripts/runROSMagicianVisionClassifier.sh
```

Or directly:

```bash
python3 liveClassifierTorchROS.py
```

### Two-Stage Ensemble

The `EnsembleClassifier.py` module implements a two-stage ensemble classifier:

1. **Stage 1 (Prefilter):** A fast binary classifier (typically ResNet18) classifies all tiles as clean or non-clean. Clean tiles are discarded.
2. **Stage 2 (Ensemble):** Multiple backbone models run in parallel on the remaining non-clean tiles. Each model casts a vote, and the majority vote determines the final class.

**Execution modes:**
- **Async CUDA streams** (default): All models execute concurrently on separate CUDA streams with `channels_last` memory format for maximum throughput.
- **CPU thread pool:** Parallel classification via `ThreadPoolExecutor` for CPU-only systems.
- **Serial:** Sequential execution for low-VRAM systems.

### Runtime Parameters

The following can be adjusted at runtime via ROS2 services:

- Target FPS
- Tile step size (controls overlap)
- Classification confidence threshold
- Visualization toggle

---

## ROS2 Integration

### Package

The `magician_vision_classifier` package uses `ament_cmake` and is compatible with ROS2 Rolling.

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `detections` | `magician_vision_classifier/Detection` | 2D bounding box with class label and confidence |
| `detections_m` | `magician_vision_classifier/DetectionM` | Detection with severity level and 3D pose |
| `markers` | `magician_vision_classifier/Marker` | ArUco marker detections with pose |

### Message Definitions

**Detection:**
```
std_msgs/Header header
int32 x, y          — Bounding box top-left corner
int32 w, h          — Bounding box dimensions
float32 depth        — Depth in meters (from laser fusion)
string type          — Detection type string
string class_name    — Predicted class
float32 probability  — Classification confidence
```

**DetectionM:**
```
int32 severity       — ClassA=1, ClassB=2, ClassC=3
geometry_msgs/Pose   — 3D pose with depth interpolation
```

**Marker:**
```
string id            — ArUco marker identifier
geometry_msgs/Pose   — Position and quaternion orientation
```

### Services

| Service | Type | Description |
|---------|------|-------------|
| `set_fps` | `SetFloat64` | Set target inference frame rate |
| `set_step` | `SetInt64` | Set tile step size |
| `set_threshold` | `SetFloat64` | Set classification confidence threshold |
| `set_visualization` | `std_srvs/Trigger` | Toggle OpenCV visualization |
| `remember_defect` | `std_srvs/Trigger` | Save current frame as defect sample |
| `remember_clean` | `std_srvs/Trigger` | Save current frame as clean sample |
| `scan_markers` | `std_srvs/Trigger` | Trigger ArUco marker scan |
| `pause` | `std_srvs/Trigger` | Pause/resume inference |

### Service Definitions

**SetFloat64:**
```
float64 value
---
bool success
```

**SetInt64:**
```
int64 value
---
bool success
```

### Utility Scripts

- `scripts/setFPS.sh` — adjust target FPS
- `scripts/setStepSize.sh` — adjust tile step size
- `scripts/visualizationOn.sh` / `scripts/visualizationOff.sh` — toggle visualization
- `scripts/scanForMarkers.sh` — trigger ArUco marker scanning
- `scripts/echoROSMagicianVisionClassifierDetections.sh` — print detections to console

### Depth Fusion

The node subscribes to three laser depth topics (`magician_grabber/distance{1,2,3}`) and fuses depth measurements at detection pixel positions using inverse-distance-weighted interpolation.

### Marker Detection

ArUco marker detection uses the `DICT_6X6_250` dictionary. Detected markers are published with 6-DOF pose estimates derived from camera calibration.

---

## Model Analysis and Evaluation

### Single-Model Evaluation

Evaluate a single trained checkpoint on one or more dataset directories:

```bash
python3 evaluateClassifierNew.py model.pth config.json /path/to/eval_dir
python3 evaluateClassifierNew.py model.pth config.json /path/to/eval_dir <batch_size>
python3 evaluateClassifierNew.py model.pth config.json /path/to/eval_dir 16 Class1,Class2
```

For a quick single-model pass without a config file:

```bash
python3 evaluateTorch.py
```

### Batch Evaluation

Evaluate one or more trained models on a dataset:

```bash
python3 ModelAnalysisEvaluate.py
```

This produces per-model accuracy, precision, recall, F1-score, inference speed (FPS), and parameter count.

Alternatively, evaluate directly on a raw PNG dataset (without HDF5 conversion):

```bash
python3 ModelAnalysisEvaluateRawDataset.py
```

### HTML Report Generation

Generate a comprehensive HTML report from evaluation data:

```bash
python3 ModelAnalysisReport.py <analysis_directory>
```

The report includes:
- Per-model accuracy, balanced accuracy, FPS, parameter count, and latency
- Per-class precision, recall, and F1-score tables
- Confusion matrix heatmaps
- Efficiency plots (accuracy vs parameters)
- Ensemble optimization via greedy forward selection and backward elimination
- Pareto front computation
- Benchmark 3D scatter plots (batch size × step size vs FPS)

### Confusion Matrix Plotting

```bash
python3 plotTool.py <confusion_matrix.json>
```

Generates four visualization variants: raw counts, row-normalized, total-normalized, and hybrid.

### Ensemble Optimization

```bash
python3 calculateOptimalEnsemble.py
```

Computes optimal ensemble weights and model subsets for maximizing accuracy under latency constraints.

To evaluate the resulting ensemble combinations across different voting strategies (soft averaging, majority vote, confidence-weighted):

```bash
python3 evaluateOptimalEnsemble.py
```

### Defect–Clean Confusion Analysis

Identify defect samples that the classifier is misclassifying as clean — useful for auditing the Stage 1 prefilter and finding hard negatives for retraining:

```bash
python3 identifyDefectsConfusedWithClean.py
```

### Benchmarking

- `scripts/bench_to_csv.sh` — convert benchmark output to CSV
- `scripts/run_benchmark_plotting.sh` — run benchmarks and generate Gnuplot comparisons
- Benchmark result files for each architecture are in `scripts/benchmark_allclass_*.txt`

---

## Shared Memory System

The classifier receives frames from the camera grabber via a zero-copy shared memory mechanism implemented in `libSharedMemoryVideoBuffers.so`. This library uses POSIX shared memory (`mmap`) to transfer video frames without serialization, enabling sub-millisecond frame delivery.

### Components

| File | Description |
|------|-------------|
| `libSharedMemoryVideoBuffers.so` | Compiled C library for shared memory buffers |
| `SharedMemoryVideoBuffers/` | C library source code with Python binding examples |
| `SharedMemoryManager.py` | Python ctypes bindings to the shared memory library |
| `SharedMemoryServer.py` | Standalone shared memory server for testing |
| `video_frames.shm` | Shared memory frame descriptor file |
| `shared_memory_context.shm` | Shared memory context descriptor file |

### Python Interface

The `SharedMemoryManager.py` module provides Python bindings for:
- Creating and connecting to shared memory context descriptors
- Mapping remote frame buffers into local address space
- Lock-free read/write access with mutex protection
- Timestamp handling for frame synchronization

---

## Project Structure

```
magician_vision_classifier/
  ├── trainMagicianVisionClassifierTorch.py   Main training script
  ├── liveClassifierTorch.py                  Standalone inference engine
  ├── liveClassifierTorchROS.py               ROS2 inference node
  ├── EnsembleClassifier.py                   Two-stage ensemble classifier
  ├── EnsembleClassifierOG.py                 Legacy ensemble variant
  ├── DatasetConverter.py                     PNG to HDF5 dataset converter
  ├── DataLoader.py                           Dataset utilities
  ├── readData.py                             Polarization image loading and debayering
  ├── evaluateClassifierNew.py                Evaluate a checkpoint on one or more dataset dirs
  ├── evaluateTorch.py                        Quick single-model evaluation
  ├── evaluateOptimalEnsemble.py              Evaluate ensemble voting strategies
  ├── identifyDefectsConfusedWithClean.py     Find defects misclassified as clean
  ├── ModelAnalysisEvaluate.py                Batch model evaluation
  ├── ModelAnalysisEvaluateRawDataset.py      Batch evaluation on raw PNG datasets
  ├── ModelAnalysisReport.py                  HTML report generator
  ├── plotTool.py                             Confusion matrix plotting
  ├── calculateOptimalEnsemble.py             Ensemble weight optimization
  ├── integrator3D.py                         OpenGL 3D visualization of detections
  ├── SharedMemoryManager.py                  Shared memory Python bindings
  ├── SharedMemoryServer.py                   Standalone shared memory server
  ├── viewer.py                               Live polarization image viewer
  ├── splitChannels.py                        Polarization channel splitter
  ├── CMakeLists.txt                          ROS2 package build configuration
  ├── requirements.txt                        Python dependencies
  ├── last.pth                                Latest model checkpoint (symlink, updated after training)
  ├── configs/                                Training configuration files
  ├── msg/                                    ROS2 message definitions
  |   ├── Detection.msg
  |   ├── DetectionM.msg
  |   ├── Marker.msg
  ├── srv/                                    ROS2 service definitions
  |   ├── SetInt64.srv
  |   ├── SetFloat64.srv
  ├── scripts/                                Shell scripts for training and operations
  ├── SharedMemoryVideoBuffers/               Shared memory C library source
  ├── models/                                 Trained model archives
  ├── sounds/                                 Audio feedback files
```

---

## References

1. PyTorch: https://pytorch.org
2. PyTorch Lightning: https://lightning.ai
3. ROS2: https://www.ros.org
4. Pretrained backbones from `torchvision.models`: ResNet, ResNeXt, ConvNeXt, EfficientNet-V2, Swin-V2, RegNet, MobileNet-V3, ShuffleNet-V2, SqueezeNet, DenseNet, MNASNet

This work has been developed within the context of research funded by the European Union's Horizon 2020 research and innovation programme.
=======
### Supported architectures

| Key | Architecture |
|---|---|
| `resnet18` | ResNet-18 |
| `resnext50` | ResNeXt-50 |
| `convnext_tiny` | ConvNeXt-Tiny |
| `efficientnet_b0` | EfficientNet-B0 |
| `efficientnet_v2_s` | EfficientNet-V2-S |
| `densenet121` | DenseNet-121 |
| `mobilenet_v3_large` | MobileNet-V3-Large |
| `mobilenet_v3_small` | MobileNet-V3-Small |
| `shufflenet_v2_x1_0` | ShuffleNet-V2-x1.0 |
| `regnet_y_400mf` | RegNet-Y-400MF |
| `regnet_y_800mf` | RegNet-Y-800MF |
| `mnasnet1_0` | MNASNet-1.0 |
| `squeezenet1_1` | SqueezeNet-1.1 |
| `swin_v2_t` | Swin Transformer V2-Tiny |
| `small_cnn` | Lightweight custom CNN |
| `verysmall_cnn` | Ultra-lightweight custom CNN |
| `custom` | Configurable custom CNN |

All architectures accept **4-channel RGBA input** tiles.

### Training features

- **Focal Loss** and **Cross Entropy Loss** with optional class weighting
- **False-clean penalization** — extra loss term discouraging defect-as-clean misclassification
- **Polarization channels** — optional AoLP, DoLP, and unpolarized representations
- **Early stopping** with best-weight restoration
- **Automatic confusion matrix** generation and export
- **W&B and TensorBoard** logging support
- **Model packaging** — checkpoint, config, and confusion matrix are zipped and archived under `models/`

### Configuration

Training is driven by a JSON configuration file:

```json
{
    "hparams": {
        "tile_size": 48,
        "batch_size": 64,
        "training_epochs": 25,
        "dropout_rate": 0.25,
        "seed": 42,
        "gradient_clip_value": 1.0,
        "AoLP": false,
        "DoLP": false,
        "unpolarized": false
    },
    "early_stopping": {
        "monitor": "loss",
        "mode": "min",
        "patience": 16,
        "min_delta": 0.0005,
        "verbose": 1,
        "restore_best_weights": true
    },
    "optimizer": {
        "type": "AdamW",
        "learning_rate": 5e-4
    },
    "dataloader": {
        "seed": 42,
        "validation_split": 0.05,
        "shuffle": false,
        "num_workers": 12
    },
    "wandb": {
        "project": "Classifier",
        "name": "run_name",
        "use_wandb": false
    },
    "tensorboard_log_dir": "tile_classifier/tensorboard/",
    "directory": "dataset/",
    "class_weight": false,
    "loss": "focal",
    "penalize_false_clean": 0.0,
    "accelerator": "auto",
    "devices": 1,
    "name": "allclass",
    "model": "convnext_tiny",
    "selected_classes": []
}
```

`selected_classes` filters the dataset to only the listed class names; an empty list uses all classes found in the dataset directory.

### Running training

```bash
python3 trainClassifierTorch.py config.json
```

### Output files

| File | Description |
|---|---|
| `<name>.pth` | Model checkpoint |
| `<name>.json` | Configuration and validation metrics |
| `<name>_confusion.json` | Confusion matrix data |
| `models/<timestamp>.zip` | Archived training artefacts |
| `last.pth`, `last.json` | Symlinks to the most recent checkpoint |

### Validation metrics

`val_loss`, `val_accuracy`, `val_precision`, `val_recall`, `val_auroc`, confusion matrix (raw counts and plotted image).

---

## Live Inference (ROS 2 Node)

### Running

The node is intended to be launched from the ROS 2 workspace root:

```bash
scripts/runROSMagicianVisionClassifier.sh
```

Or directly (after sourcing the workspace):

```bash
python3 liveClassifierTorchROS.py
```

### Inference modes

| Mode | Description |
|---|---|
| **Single model** (default) | One classifier runs on every frame. Fast and deterministic. |
| **Two-stage ensemble** | A lightweight binary model filters tiles; a pool of classifiers votes on positives. Enabled via the `set_two_stage` service. |

### Published topics

| Topic | Type | Description |
|---|---|---|
| `/detections` | `Detection` | One message per defect tile: bounding box, type, class, confidence, depth |
| `/detections_m` | `DetectionM` | Defect with IDW-interpolated depth from laser sensors (when `USE_LASERS=True`) |
| `/markers` | `Marker` | ArUco marker ID and 6-DoF pose (active during marker scan window) |
| `/background_activations` | `BackgroundActivations` | Average softmax confidence of clean tiles |

### Detection message fields

| Field | Type | Description |
|---|---|---|
| `x`, `y` | int | Top-left pixel of the defect tile |
| `w`, `h` | int | Tile dimensions in pixels |
| `type` | string | Defect type (e.g. `NegativeDent`) |
| `class_name` | string | Severity class (`ClassA`, `ClassB`, `ClassC`) |
| `probability` | float | Model confidence (0–1) |
| `depth` | float | IDW-interpolated depth in metres (0 if lasers disabled) |

### ROS 2 services

All services are prefixed with `/magician_vision_classifier/`.

| Service | Type | Description |
|---|---|---|
| `set_visualization` | `SetBool` | Open / close the live heatmap display window |
| `pause` | `SetBool` | Pause or resume inference |
| `set_two_stage` | `SetBool` | Toggle two-stage ensemble mode |
| `set_fps` | `SetFloat64` | Set target frame rate (0 = unlimited) |
| `set_step` | `SetInt64` | Set tile stride in pixels (larger = faster, coarser) |
| `set_threshold` | `SetFloat64` | Set minimum confidence threshold; low-confidence tiles are reclassified as clean |
| `set_frame_limiter` | `SetBool` | Skip duplicate frames from shared memory (disable for unlimited rate) |
| `set_model` | `SetString` | Hot-swap the single classifier model by name or path stem (e.g. `allclass_resnet18`) |
| `remember_defect` | `Trigger` | Save the current frame + detection JSON to `data/` tagged as a defect |
| `remember_clean` | `Trigger` | Save the current frame + detection JSON to `data/` tagged as clean |
| `set_autosave_defect_snapshots` | `SetBool` | Automatically save a frame + JSON every time a defect is detected |
| `snapshot` | `Trigger` | Save the current frame to `snapshots/` on demand |
| `scan_markers` | `Trigger` | Activate ArUco marker detection for 3 seconds |

### Detection JSON sidecar

Whenever a frame is saved (via `remember_defect`, `remember_clean`, or auto-save), a JSON file is written alongside the PNG with the same basename:

```json
{
  "timestamp_ns": 1747123456789000000,
  "tile_size": 48,
  "background_avg_prob": 0.971,
  "detections": [
    {
      "x": 312,
      "y": 128,
      "w": 48,
      "h": 48,
      "type": "NegativeDent",
      "class_name": "ClassB",
      "probability": 0.934
    }
  ]
}
```

`timestamp_ns` is the Unix nanosecond timestamp of the source frame from shared memory, matching the stamp in the published ROS 2 messages.

### Hot-swapping the model at runtime

The active single classifier can be replaced without restarting the node:

```bash
scripts/SetModel.sh allclass_resnet18
```

The service checks that both the `.pth` checkpoint and `.json` configuration exist before loading. Inference is blocked for the duration of the reload.

---

## Scripts

All scripts are intended to be run from the **ROS 2 workspace root** (one level above this package). They source `install/setup.bash` automatically.

| Script | Description |
|---|---|
| `runROSMagicianVisionClassifier.sh` | Launch the inference node |
| `visualizationOn.sh` / `visualizationOff.sh` | Toggle the live display window |
| `setFPS.sh <fps>` | Set the target frame rate |
| `setStepSize.sh <step>` | Set the tile stride |
| `SetModel.sh <name>` | Hot-swap the single classifier model |
| `rememberDefect.sh` | Save the current frame as a defect sample |
| `rememberClean.sh` | Save the current frame as a clean sample |
| `LogEncounteredDefects.sh` | Enable automatic saving of defect frames |
| `scanMarkers.sh` / `scanForMarkers.sh` | Trigger a 3-second ArUco marker scan |
| `fullTraining.sh` | Run a full training sweep across configurations |
| `secondaryTraining.sh` | Run secondary / fine-tuning training |
>>>>>>> b25a3bdd9acd41e79c030f0d1923dfcc54f52d80

---

## Contact

<<<<<<< HEAD
**Ammar Qammaz**
Foundation for Research and Technology – Hellas (FORTH)
Institute of Computer Science, Greece
ammarkov@gmail.com
=======
**Ammar Qammaz**  
Foundation for Research and Technology – Hellas (FORTH)  
Computer Science Department, Greece
>>>>>>> b25a3bdd9acd41e79c030f0d1923dfcc54f52d80
