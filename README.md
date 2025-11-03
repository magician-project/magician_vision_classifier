# Magician Vision Classifier System

**Authors:** Ammar Qammaz, Nikos Vasilikopoulos
**Copyright:** © 2025 Foundation of Research and Technology – Hellas (FORTH), Computer Science Department, Greece

---

## 🧠 Overview

The **Magician Vision Classifier System** provides a complete pipeline for **training**, **evaluating**, and **deploying** a real-time vision-based classifier within a ROS2 ecosystem.

It consists of:
- A **training module** (`trainClassifierTorch.py`) using **PyTorch Lightning** for robust supervised learning on RGBA image datasets.
- A **live ROS2 publisher** (`magician_vision_classifier_publisher.py`) that performs real-time inference using shared memory video streams and publishes detection results as ROS2 messages.

---

## 📦 Components


### 0. Setup
Please place the repository in your ROS package directory.
To get its python requirements use:

```
python3 -m venv venv && source venv/bin/activate && python3 -m pip install -r requirements.txt
```


### 1. `trainClassifierTorch.py` — Model Training & Evaluation

This script handles the **end-to-end training process** for the defect classifier, including:
- Dataset preparation and filtering
- Model configuration and training
- Validation and metric computation
- Confusion matrix generation and saving
- Exporting trained model and configuration files
- Archiving model checkpoints and metrics for deployment

#### 🧩 Supported Architectures
You can specify any of the following models in the configuration JSON:
- `resnet18`
- `resnext50`
- `convnext_tiny`
- `efficientnet_v2_s`
- `swin_v2_t`
- `regnet_y_800mf`
- `custom` (custom CNN defined in the code)

All networks support **4-channel RGBA inputs**.

#### 🧮 Features
- **Focal Loss** and **Cross Entropy Loss** options  
- **Class imbalance handling** via weighting  
- **Polarization features**: AoLP (Angle of Linear Polarization), DoLP (Degree of Linear Polarization), Unpolarized input  
- **False clean penalization**: extra loss term to discourage misclassifying defects as “clean”
- **Automatic confusion matrix generation**
- **Optional W&B and TensorBoard logging**
- **Automatic model packaging** (`.zip`) and symlink updates (`last.pth`, `last.json`)

#### 🧰 Dataset Handling
- Expects a directory with RGBA `.png` or `.jpg` images.
- Filenames or folder structure define the class labels.
- Dataset class: `RGBAImageFolder`
- Optionally filters only selected classes via:
  ```json
  "selected_classes": ["ClassA", "ClassB", "Clean"]


#### ⚙️ Example Configuration (config.json)
```
{
    "hparams":{
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
        "monitor":"loss",
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
        "label_mode": "categorical",
        "num_workers": 12
        
    },
    "wandb": {
        "project": "Classifier",
        "name": "AmmarConfig",
        "use_wandb": false
    },
    "tensorboard_log_dir": "tile_classifier/tensorboard/",
    "directory": "keras_dataset/",
    "class_weight": false,
    "loss": "focal",
    "penalize_false_clean": 0.0,
    "accelerator": "auto",
    "name": "allclass",
    "model": "custom",
    "devices": 1,
    "selected_classes": [ ]
  }
```

#### 🚀 Run Training
```
python3 trainClassifierTorch.py config.json
```


#### 🧾 Output Files

After training, the following files are created:
```
<model_name>.pth             # Model checkpoint
<model_name>.json            # Configuration + results
<model_name>_confusion.json  # Confusion matrix data
models/<timestamp>.zip       # Archived training results
last.pth, last.json          # Symlinks to latest model
```



#### 📊 Metrics Logged

    val_loss

    val_accuracy

    val_precision

    val_recall

    val_auroc

    Confusion matrix (numpy + plotted version)



#### 🧩 Shared Memory Library

The node expects:
```
libSharedMemoryVideoBuffers.so
video_frames.shm
```


#### 📂 Repository Structure
```
├── magician_vision_classifier_publisher.py  # ROS2 inference node
├── trainClassifierTorch.py                  # Model training & evaluation
├── liveClassifierTorch.py                   # Runtime model wrapper
├── SharedMemoryManager.py                   # Shared memory interface
├── libSharedMemoryVideoBuffers.so           # Shared memory library
├── video_frames.shm                         # Shared memory descriptor
├── last.pth                                 # Latest trained model weights
├── last.json                                # Latest trained model configuration
├── config.json                              # Example training configuration
├── plotTool.py                              # Confusion matrix plotting
├── models/                                  # Saved model archives
└── license.txt
```


#### ✉️ Contact

Ammar Qammaz
Foundation for Research and Technology – Hellas (FORTH)
Computer Science Department, Greece

