# Magician Vision Classifier

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
```

---

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

---

## Training

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

---

## Contact

**Ammar Qammaz**  
Foundation for Research and Technology – Hellas (FORTH)  
Computer Science Department, Greece
