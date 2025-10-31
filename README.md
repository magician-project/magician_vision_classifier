# Magician Vision Classifier System

**Authors:** Ammar Qammaz, Nikos Vasilikopoulos  
**Copyright:** © 2025 Foundation of Research and Technology – Hellas (FORTH), Computer Science Department, Greece  
**License:** FORTH License (see `license.txt`)  

---

## 🧠 Overview

The **Magician Vision Classifier System** provides a complete pipeline for **training**, **evaluating**, and **deploying** a real-time vision-based classifier within a ROS2 ecosystem.

It consists of:
- A **training module** (`trainClassifierTorch.py`) using **PyTorch Lightning** for robust supervised learning on RGBA image datasets.
- A **live ROS2 publisher** (`magician_vision_classifier_publisher.py`) that performs real-time inference using shared memory video streams and publishes detection results as ROS2 messages.

---

## 📦 Components

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
  "name": "defect_detector",
  "model": "resnet18",
  "loss": "focal",
  "hparams": {
    "batch_size": 32,
    "training_epochs": 50,
    "dropout_rate": 0.2,
    "tile_size": 64,
    "gradient_clip_value": 0.5,
    "seed": 42
  },
  "dataloader": {
    "validation_split": 0.2,
    "num_workers": 4
  },
  "optimizer": {
    "learning_rate": 0.0001
  },
  "class_weight": true,
  "penalize_false_clean": 0.1,
  "wandb": {
    "use_wandb": false,
    "project": "magician-vision",
    "name": "run_"
  },
  "accelerator": "gpu",
  "devices": 1,
  "directory": "dataset/"
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


#### 🧾 License

This project is licensed under the FORTH License.
See license.txt for full details.

#### ✉️ Contact

Ammar Qammaz
Foundation for Research and Technology – Hellas (FORTH)
Computer Science Department, Greece

