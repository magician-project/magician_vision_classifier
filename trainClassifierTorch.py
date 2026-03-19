#!/usr/bin/python3

""" 
Author : "Nikos Vasilikopoulos, Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics import Accuracy, Recall, Precision, AUROC,ConfusionMatrix
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import random
import numpy as np
import datetime
import json
from collections import Counter
import sys
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision.models import (
    resnet18, ResNet18_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    regnet_y_800mf, RegNet_Y_800MF_Weights,
    regnet_y_400mf, RegNet_Y_400MF_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    squeezenet1_1, SqueezeNet1_1_Weights,
    densenet121, DenseNet121_Weights,
    mnasnet0_5, MNASNet0_5_Weights,
    mnasnet1_0, MNASNet1_0_Weights,
)
from torch.nn import functional as F
import torchvision
from pytorch_lightning.loggers import WandbLogger
import wandb


torch.set_float32_matmul_precision('high')


def filter_dataset_classes(dataset, keep_classes):
    """
    Keeps only the specified classes in the dataset and removes the rest.

    Args:
        dataset (RGBAImageFolder): The dataset to filter.
        keep_classes (list[str]): A list of class names to keep.
    """
    # Get indices of the classes to keep
    keep_indices = [dataset.class_to_idx[c] for c in keep_classes if c in dataset.class_to_idx]

    # Filter samples and targets
    filtered_samples = []
    filtered_targets = []

    for sample, target in dataset.samples:
        if target in keep_indices:
            filtered_samples.append((sample, target))
            filtered_targets.append(target)

    dataset.samples = filtered_samples
    dataset.targets = filtered_targets

    # Update class lists
    dataset.classes = keep_classes
    dataset.class_to_idx = {cls: i for i, cls in enumerate(keep_classes)}

    print(f"Filtered dataset to {len(dataset.samples)} samples across {len(keep_classes)} classes.")



def evaluate_dumped_tiles(model, tiles_dir, classes, device='cuda', batch_size=16):
    """
    Evaluates dumped PNG tiles against the current model.

    Args:
        model (torch.nn.Module): Trained classifier model.
        tiles_dir (str): Path to the directory containing dumped tiles.
        classes (list): List of class names (order must match model outputs).
        device (str): 'cuda' or 'cpu'.
        batch_size (int): Batch size for inference.

    Returns:
        dict: metrics containing accuracy, confusion matrix, and report.
    """

    import re
    from tqdm import tqdm
    from sklearn.metrics import confusion_matrix, classification_report
    model.eval()
    model.to(device)

    # Transformation: RGBA -> Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Collect all .png tiles
    all_files = [os.path.join(tiles_dir, f) for f in os.listdir(tiles_dir) if f.lower().endswith('.png')]
    if not all_files:
        print(f"No PNG files found in {tiles_dir}")
        return None

    y_true = []
    y_pred = []

    batch = []
    batch_gt = []

    print(f"Evaluating {len(all_files)} dumped tiles from '{tiles_dir}'...")

    for fpath in tqdm(all_files):
        # Parse filename pattern: tile_000001_y0_x0_cls2_dust.png
        match = re.search(r"_cls(-?\d+)_", os.path.basename(fpath))
        if not match:
            print(f"Could not extract class ID from {fpath}")
            continue
        gt_cls = int(match.group(1))
        if gt_cls < 0 or gt_cls >= len(classes):
            continue  # skip unknown class

        # Load RGBA image
        img = load_rgba_image(fpath)
        img = transform(img)
        batch.append(img)
        batch_gt.append(gt_cls)

        # Process in batches
        if len(batch) == batch_size:
            inputs = torch.stack(batch).to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch_gt)
            batch.clear()
            batch_gt.clear()

    # Handle remainder
    if batch:
        inputs = torch.stack(batch).to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(batch_gt)

    # Compute metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)

    print(f"\nEvaluation complete — Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Return results as dictionary
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report
    }



#-------------------------------------------------------------------------------
def checkIfPathExists(filename):
    return os.path.exists(filename)
#-------------------------------------------------------------------------------
def checkIfPathIsDirectory(filename):
    return os.path.isdir(filename) 
#-------------------------------------------------------------------------------
def checkIfFileExists(filename):
    return os.path.isfile(filename)
#-------------------------------------------------------------------------------

def load_hyperparameters(config_file):
    if not checkIfFileExists(config_file):
        print("Config file not found")
        sys.exit(1)
    with open(config_file) as json_file:
        data = json.load(json_file)
    return data


class CategoricalFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Categorical Focal Loss for multi-class classification.

        :param gamma: Focusing parameter (higher values focus more on hard examples)
        :param alpha: Class weighting (list or tensor of shape [num_classes] or None)
        :param reduction: Reduction mode: 'none' | 'mean' | 'sum'
        """
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Computes the focal loss.
        
        :param logits: Tensor of shape [batch_size, num_classes] (raw model outputs before softmax)
        :param targets: Tensor of shape [batch_size] with class indices (integer labels)
        :return: Scalar loss value
        """
        # Convert class indices to one-hot encoding
        num_classes = logits.shape[1]
        #target_one_hot = F.one_hot(targets, num_classes).float() #<-Original 
        target_one_hot = F.one_hot(targets, num_classes).to(torch.float32)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Gather the probabilities corresponding to the true class
        pt = (probs * target_one_hot).sum(dim=1)  # Shape: [batch_size]

        # Compute focal loss term
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Apply class weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # Select alpha value for each target
            ce_loss = alpha_t * ce_loss

        # Compute final loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Return per-sample loss if 'none'



class CustomCNN(nn.Module):
    def __init__(self, in_channels=4, intended_tile_size=64, num_classes=4, dropout_rate=0.5, base_channels=32, final_dense_layer=512):
        super(CustomCNN, self).__init__()
  
        print("Custom CNN (",base_channels,",",final_dense_layer,") constructor")
        self.channels  = in_channels
        self.tile_size = intended_tile_size
 
        c1 = base_channels
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(2)

        c2 = base_channels * 2
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(2)

        c3 = base_channels * 4
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(c3)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(c3, c3, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(c3)
        self.pool4 = nn.MaxPool2d(2)

        prefinalLayerChannels     = int(final_dense_layer * 4.5) 
        intermediateLayerChannels = int(final_dense_layer * 1.5)   # new FC layer
        finalLayerChannels        = final_dense_layer              # output to final FC

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(c3, prefinalLayerChannels)
        self.fc2 = nn.Linear(prefinalLayerChannels, intermediateLayerChannels)  # new layer
        self.bn_dense1 = nn.InstanceNorm1d(prefinalLayerChannels)
        self.bn_dense2 = nn.InstanceNorm1d(intermediateLayerChannels)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(intermediateLayerChannels, num_classes)

    def forward(self, x):
        if x.shape[1:] != (self.channels, self.tile_size, self.tile_size):  # Sanity check on desired input size
          raise ValueError(f"Input size must be {self.channels}x{self.tile_size}x{self.tile_size}, got {x.shape[1:]}")
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # First FC
        x = F.relu(self.bn_dense1(self.fc1(x)))
        x = self.dropout(x)

        # Second FC
        x = F.relu(self.bn_dense2(self.fc2(x)))
        x = self.dropout(x)

        x = self.out(x)
        return x



class Classifier(pl.LightningModule):
    def __init__(self,
                      model='resnet18',
                      loss='focal',
                      tile_size=64,
                      num_classes=4,
                      dropout_rate=0.1,
                      train=True,
                      lr=1e-4,
                      AoLP=False,
                      DoLP=False,
                      Unpolarized=False,
                      MaxPolarization=False,
                      MinPolarization=False,
                      RangePolarization=False,
                      load_checkpoint=None,
                      penalize_false_clean=0.0,
                      base_channels=32,
                      final_dense_layer=512,
                      clean_class=0,
                      noise_std=0.0,
                      noise_clip=None
                 ):
        super(Classifier, self).__init__()
        #-----------------------------------------
        self.type              = model
        self.lr                = lr
        self.tile_size         = tile_size
        self.num_classes       = num_classes
        self.dropout_rate      = dropout_rate
        self.base_channels     = base_channels
        self.final_dense_layer = final_dense_layer
        #-----------------------------------------
        self.AoLP = AoLP
        self.DoLP = DoLP
        self.Unpolarized = Unpolarized
        self.MaxPolarization   = MaxPolarization
        self.MinPolarization   = MinPolarization
        self.RangePolarization = RangePolarization
        #-----------------------------------------
        self.clean_class  = clean_class
        self.penalize_false_clean = penalize_false_clean
        #-----------------------------------------
        self.noise_std  = noise_std
        self.noise_clip = noise_clip

        # Dynamic input channels (base 4 polarization channels + optional derived channels)
        extra_channels = 0
        if self.DoLP: extra_channels += 1
        if self.AoLP: extra_channels += 1
        if self.Unpolarized: extra_channels += 1
        if self.MaxPolarization: extra_channels += 1
        if self.MinPolarization: extra_channels += 1
        if self.RangePolarization: extra_channels += 1

        self.base_input_channels = 4
        self.in_channels = self.base_input_channels + extra_channels

        #RESNEXT
        if self.type == 'resnext50':
            self.model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(2048, num_classes)
        elif self.type == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512, num_classes)
        elif self.type == 'convnext_tiny':
            self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 96, kernel_size=(4, 4), stride=(4, 4))
            self.model.classifier[2]  = nn.Linear(768, num_classes)
        elif self.type == 'efficientnet_v2_s':
            self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1]  = nn.Linear(1280, num_classes, bias=True)
        elif self.type == 'swin_v2_t':
            self.model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 96, kernel_size=(4, 4), stride=(4, 4))
            self.model.head = nn.Linear(768, num_classes)
        elif self.type == 'regnet_y_800mf':
            self.model = regnet_y_800mf(weights=RegNet_Y_800MF_Weights.DEFAULT)
            self.model.stem[0] = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.fc = nn.Linear(784, num_classes)
        elif self.type == 'regnet_y_400mf':
            self.model = torchvision.models.regnet_y_400mf(weights=RegNet_Y_400MF_Weights.DEFAULT)
            self.model.stem[0] = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.fc = nn.Linear(440, num_classes)
        elif self.type == 'mobilenet_v3_small':
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[3] = nn.Linear(1024, num_classes)
        elif self.type == 'mobilenet_v3_large':
            self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[3] = nn.Linear(1280, num_classes)
        elif self.type == 'shufflenet_v2_x0_5':
            self.model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
            self.model.conv1[0] = nn.Conv2d(self.in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.fc = nn.Linear(1024, num_classes)
        elif self.type == 'shufflenet_v2_x1_0':
            self.model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
            self.model.conv1[0] = nn.Conv2d(self.in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.fc = nn.Linear(1024, num_classes)
        elif self.type == 'squeezenet1_1':
            self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
            self.model.features[0] = nn.Conv2d(self.in_channels, 64, kernel_size=(3, 3), stride=(2, 2))
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
            self.model.num_classes = num_classes
        elif self.type == 'efficientnet_b0':
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif self.type == 'densenet121':
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            self.model.features.conv0 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier = nn.Linear(1024, num_classes)
        elif self.type == 'mnasnet0_5':
            self.model = mnasnet0_5(weights=MNASNet0_5_Weights.DEFAULT)
            self.model.layers[0] = nn.Conv2d(self.in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif self.type == 'mnasnet1_0':
            self.model = mnasnet1_0(weights=MNASNet1_0_Weights.DEFAULT)
            self.model.layers[0] = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif ('custom' in self.type) or ('cnn' in self.type):
            self.model = CustomCNN(
                                   in_channels=self.in_channels,
                                   intended_tile_size=tile_size,
                                   num_classes=num_classes,
                                   dropout_rate=dropout_rate,
                                   base_channels=self.base_channels,
                                   final_dense_layer=self.final_dense_layer
                                  )
        else:
            raise ValueError(f"Unsupported model type: {model}. Supported types are 'resnext50', 'resnet18', 'convnext_tiny', 'efficientnet_v2_s', 'swin_v2_t', 'regnet_y_800mf'.")

        if (load_checkpoint is not None):
           self.model = Classifier.load_from_checkpoint(load_checkpoint)

        if loss == 'focal':
            self.criterion = CategoricalFocalLoss(gamma=2.0, alpha=None)
        elif loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss}. Supported types are 'focal' and 'cross_entropy'.")

        self.accuracy  = Accuracy(task='MULTICLASS',  num_classes=num_classes)
        self.recall    = Recall(task='MULTICLASS',    num_classes=num_classes)
        self.precision = Precision(task='MULTICLASS', num_classes=num_classes)
        self.auroc     = AUROC(task='MULTICLASS',     num_classes=num_classes)

    def add_input_noise(self, x):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(x) * self.noise_std
            if self.noise_clip is not None:
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            x = x + noise
        return x

    def build_input_features(self, x):
        """
        Build model input by appending derived channels.
        All polarization-derived features are computed from the original 4 channels only.
        Expects x shape: [B, >=4, H, W] at input (normally [B,4,H,W]).
        Returns x shape: [B, self.in_channels, H, W]
        """
        if x.shape[1] < 4:
            raise ValueError(f"Expected at least 4 channels for polarization input, got {x.shape[1]}")

        pol = x[:, 0:4, :, :]  # original polarization channels only

        # AoLP / DoLP from Stokes computed on the original channels
        if self.AoLP or self.DoLP:
            stokes = self.calculate_stokes(pol)

            if self.DoLP:
                dolp = self.calculate_DoLP(stokes).unsqueeze(1)  # [B,1,H,W]
                x = torch.cat((x, dolp), dim=1)

            if self.AoLP:
                aolp = self.calculate_AoLP(stokes).unsqueeze(1)  # [B,1,H,W]
                x = torch.cat((x, aolp), dim=1)

        # Unpolarized = mean over original 4 channels
        if self.Unpolarized:
            mon = pol.mean(dim=1, keepdim=True)
            x = torch.cat((x, mon), dim=1)

        # Max / Min / Range over original 4 channels
        if self.MaxPolarization:
            max_pol = pol.max(dim=1, keepdim=True)[0]
            x = torch.cat((x, max_pol), dim=1)

        if self.MinPolarization:
            min_pol = pol.min(dim=1, keepdim=True)[0]
            x = torch.cat((x, min_pol), dim=1)

        if self.RangePolarization:
            max_pol = pol.max(dim=1, keepdim=True)[0]
            min_pol = pol.min(dim=1, keepdim=True)[0]
            range_pol = max_pol - min_pol
            x = torch.cat((x, range_pol), dim=1)

        # Final sanity: ensure model sees the expected channel count
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Feature builder produced {x.shape[1]} channels, expected {self.in_channels}. "
                             f"(Flags: DoLP={self.DoLP}, AoLP={self.AoLP}, Unpolarized={self.Unpolarized}, "
                             f"MaxPolarization={self.MaxPolarization}, MinPolarization={self.MinPolarization}, "
                             f"RangePolarization={self.RangePolarization})")
        return x

    def forward(self, x):
        x = self.build_input_features(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = self.add_input_noise(x)
        x = self.build_input_features(x)

        y_hat = self.model(x)
        base_loss  = self.criterion(y_hat, y)

        if self.penalize_false_clean > 0.0:
            pred_probs = F.softmax(y_hat, dim=1)
            non_clean_mask = (y != self.clean_class)
            penalty_strength = float(self.penalize_false_clean)

            if non_clean_mask.any():
                p_clean = pred_probs[non_clean_mask, self.clean_class]
                false_clean_loss = -torch.log(1.0 - p_clean + 1e-8).mean()
                loss = base_loss + penalty_strength * false_clean_loss
            else:
                loss = base_loss
        else:
            loss = base_loss

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = self.build_input_features(x)

        y_hat = self.model(x)
        loss  = self.criterion(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy.update(y_hat, y)
        self.log('val_accuracy',  self.accuracy.compute(),  prog_bar=True, sync_dist=True)

        self.recall.update(y_hat, y)
        self.log('val_recall',    self.recall.compute(),    prog_bar=True, sync_dist=True)

        self.precision.update(y_hat, y)
        self.log('val_precision', self.precision.compute(), prog_bar=True, sync_dist=True)

        self.auroc.update(y_hat, y)
        self.log('val_auroc', self.auroc.compute(), prog_bar=True, sync_dist=True)

        return loss

    def calculate_stokes(self, x):
        """
        Calculate the Stokes parameters from the input tensor.
        Assuming x is a tensor of shape (batch_size, 4, height, width).
        """
        S0 = x[:, 0, :, :]
        S1 = x[:, 1, :, :] - x[:, 2, :, :]
        S2 = x[:, 1, :, :] + x[:, 2, :, :]
        S3 = x[:, 3, :, :]
        return torch.stack((S0, S1, S2, S3), dim=1)

    def calculate_DoLP(self, x):
        """
        Calculate the Degree of Linear Polarization (DoLP) from the Stokes parameters.
        Assuming x is a tensor of shape (batch_size, 4, height, width).
        """
        S0 = x[:, 0, :, :]
        S1 = x[:, 1, :, :]
        S2 = x[:, 2, :, :]
        #S3 = x[:, 3, :, :]  # Not used in DoLP
        DoLP = torch.sqrt(S1**2 + S2**2) / (S0 + 1e-8)
        return DoLP

    def calculate_AoLP(self, x):
        """
        Calculate the Angle of Linear Polarization (AoLP) from the Stokes parameters.
        Assuming x is a tensor of shape (batch_size, 4, height, width).
        """
        S1 = x[:, 1, :, :]
        S2 = x[:, 2, :, :]
        AoLP = 0.5 * torch.atan2(S2, S1)
        return AoLP

    def on_validation_epoch_end(self):
        self.accuracy.reset()

    def on_train_epoch_end(self):
        self.accuracy.reset()

    def configure_optimizers(self):
        self.hparams.lr = self.lr
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        return optimizer


def load_png_comment_metadata(image_path):
    """
    Read JSON metadata stored in PNG text/comment fields.

    Returns:
        dict: parsed metadata dictionary, or {} if unavailable / invalid.
    """
    try:
        from PIL import Image
        import json

        with Image.open(image_path) as img:
            candidates = []

            # Classic Pillow info dict
            if hasattr(img, "info") and isinstance(img.info, dict):
                for key in ("comment", "Comment", "description", "Description"):
                    if key in img.info and img.info[key] is not None:
                        candidates.append(img.info[key])

            # PNG text chunks
            if hasattr(img, "text") and isinstance(img.text, dict):
                for key in ("comment", "Comment", "description", "Description"):
                    if key in img.text and img.text[key] is not None:
                        candidates.append(img.text[key])

                # Also try every text chunk, in case the metadata was stored under another key
                for key, value in img.text.items():
                    if value is not None:
                        candidates.append(value)

            # Deduplicate while preserving order
            seen = set()
            unique_candidates = []
            for c in candidates:
                if isinstance(c, bytes):
                    c = c.decode("utf-8", errors="ignore")
                elif not isinstance(c, str):
                    c = str(c)

                c = c.strip()
                if c and c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)

            # Try to parse any candidate as JSON
            for c in unique_candidates:
                try:
                    parsed = json.loads(c)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

            return {}

    except Exception:
        return {}

def metadata_collate_fn(batch):
    xs = []
    ys = []
    metas = []

    for item in batch:
        if len(item) == 3:
            x, y, meta = item
        else:
            x, y = item
            meta = {}

        xs.append(x)
        ys.append(y)
        metas.append(meta)

    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys, metas

def load_rgba_image_pil(path):
    with Image.open(path) as img:
        try:
            img = img.convert('RGBA')  # Convert to RGBA
        except Exception as e:
            raise ValueError(f"Error converting image {path} to RGBA: {e}")
        return img



def load_rgba_image(image_path):
    import cv2
    rgba_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA) #Undo opencv crazyness.. <---------------FLIP RGBA to BGRA
    #rgba_image = rgba_image/255.0
    rgba_image = (rgba_image.astype('float32') / 255.0)
    #rgba_image = torch.tensor(rgba_image).float().to(device, dtype=torch.float32)
    #print(image_path," -> ",rgba_image.shape)
    return rgba_image


class RGBAImageFolderOLD(datasets.DatasetFolder):
    def __init__(self, root, transform=None):
        super(RGBAImageFolder, self).__init__(
                root,
                loader=load_rgba_image,  # Use custom loader for RGBA images
                extensions=('png', 'jpg', 'jpeg'),  # Add supported image extensions
                transform=transform
            )

class RGBAImageFolder(datasets.DatasetFolder):
    def __init__(self, root, transform=None, return_metadata=False):
        super(RGBAImageFolder, self).__init__(
            root,
            loader=load_rgba_image,
            extensions=('png', 'jpg', 'jpeg'),
            transform=transform
        )
        self.return_metadata = return_metadata

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_metadata:
            metadata = load_png_comment_metadata(path)
            return sample, target, metadata

        return sample, target


def _human_bytes(num_bytes: int) -> str:
    # Human-friendly binary units
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    n = float(num_bytes)
    for u in units:
        if n < 1024.0 or u == units[-1]:
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{n:.2f} B"


def _get_available_ram_bytes() -> int:
    """Returns available system RAM in bytes."""
    # Prefer psutil if installed
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    # Fallback: Linux /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    # kB -> bytes
                    return int(parts[1]) * 1024
    except Exception:
        pass

    return 0


def _get_path_size_bytes(path: str) -> int:
    """Returns the total size of files under path (or the file itself) in bytes."""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, _, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total
    except Exception:
        return 0


def _estimate_dataset_ram_bytes(dataset: Dataset, sample_count: int = 32) -> int:
    """Estimate RAM footprint (bytes) of caching dataset[i] objects.

    We sample a few items, compute their tensor payload sizes, and extrapolate.
    This is only a heuristic but is much better than relying on on-disk size.
    """
    n = len(dataset)
    if n == 0:
        return 0

    k = max(1, min(sample_count, n))
    # Spread samples across the dataset to avoid bias
    indices = np.linspace(0, n - 1, num=k, dtype=int)

    total_bytes = 0
    for idx in indices:
        x, y = dataset[int(idx)]
        item_bytes = 0
        # x may be a Tensor, numpy array, PIL image, or a tuple/list thereof
        def _payload_bytes(obj):
            if torch.is_tensor(obj):
                return int(obj.element_size() * obj.numel())
            if isinstance(obj, np.ndarray):
                return int(obj.nbytes)
            if isinstance(obj, Image.Image):
                # Approximate: width*height*channels*1 byte (before ToTensor). Conservative.
                bands = len(obj.getbands()) if hasattr(obj, "getbands") else 3
                return int(obj.size[0] * obj.size[1] * bands)
            if isinstance(obj, (list, tuple)):
                return sum(_payload_bytes(o) for o in obj)
            if isinstance(obj, dict):
                return sum(_payload_bytes(v) for v in obj.values())
            return 0

        item_bytes += _payload_bytes(x)
        # label bytes are negligible, but include a small constant overhead
        item_bytes += 64
        total_bytes += item_bytes

    avg = total_bytes / float(k)
    # Add overhead multiplier for Python objects / list storage
    overhead_multiplier = 1.25
    return int(avg * n * overhead_multiplier)

class RAMPreloadedDataset(Dataset):
    """
    Preloads an entire dataset into RAM (samples are cached as returned by the wrapped dataset).

    Notes:
    - This will increase startup time and RAM usage.
    - If you use DataLoader(num_workers>0), each worker may end up with its own copy depending on
      multiprocessing start method. For true single-copy behavior, prefer num_workers=0.
    """
    def __init__(self, dataset, show_progress=True):
        super().__init__()
        self._dataset = dataset
        self.classes = getattr(dataset, 'classes', None)
        self.targets = getattr(dataset, 'targets', None)

        n = len(dataset)
        print("\n[WARNING] cacheAllDataToRAM=True -> Preloading ALL dataset samples into RAM...")
        print("          This will take some time up front, but training batches will not hit the HDD.")
        print(f"          Samples to cache: {n}\n")

        self._cached = []

        iterator = range(n)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Caching dataset to RAM", unit="sample")
            except Exception:
                pass

        for i in iterator:
            self._cached.append(dataset[i])

        # If wrapped dataset doesn't expose targets, infer them from cached samples
        if self.targets is None:
            self.targets = [y for (_, y) in self._cached]

        print("[OK] Dataset cached to RAM. Starting training...\n")

    def __len__(self):
        return len(self._cached)

    def __getitem__(self, idx):
        return self._cached[idx]


class CombinedDataset(Dataset):
    """
    Concatenates multiple datasets and exposes:
      - classes
      - class_to_idx
      - targets  (concatenated)
    Assumes all sub-datasets share the same class list & mapping.
    """
    def __init__(self, datasets):
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("CombinedDataset received 0 datasets")

        # Verify consistent class mapping
        base_classes = getattr(datasets[0], "classes", None)
        base_cti     = getattr(datasets[0], "class_to_idx", None)
        if base_classes is None or base_cti is None:
            raise ValueError("Sub-dataset does not expose classes/class_to_idx")

        for i, ds in enumerate(datasets[1:], start=1):
            if getattr(ds, "classes", None) != base_classes:
                raise ValueError(f"Dataset #{i} has different classes. "
                                 f"Expected {base_classes}, got {getattr(ds,'classes',None)}")
            if getattr(ds, "class_to_idx", None) != base_cti:
                raise ValueError(f"Dataset #{i} has different class_to_idx mapping.")

        self.datasets = datasets
        self.classes = base_classes
        self.class_to_idx = base_cti

        # Build cumulative sizes for fast indexing
        self._lengths = [len(ds) for ds in datasets]
        self._offsets = []
        s = 0
        for L in self._lengths:
            self._offsets.append(s)
            s += L
        self._total_len = s

        # Concatenate targets (so your class distribution + weights still work)
        self.targets = []
        for ds in datasets:
            t = getattr(ds, "targets", None)
            if t is None:
                # Fallback if dataset doesn't expose .targets
                self.targets.extend([ds[i][1] for i in range(len(ds))])
            else:
                self.targets.extend(list(t))

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)

        # Find which dataset this index belongs to
        # (linear scan is fine for small number of datasets; can binary search if needed)
        for ds_i in range(len(self.datasets)-1, -1, -1):
            if idx >= self._offsets[ds_i]:
                local_idx = idx - self._offsets[ds_i]
                return self.datasets[ds_i][local_idx]

        raise RuntimeError("CombinedDataset indexing error")

def print_class_distribution(dataset, title="Dataset"):
    """
    Prints number of samples per class.

    Works with:
    - RGBAImageFolder
    - HDF5Dataset (if it exposes .targets)
    - random_split subsets
    """
    print(f"\n--- {title} Class Distribution ---")

    # Handle Subset (from random_split)
    if isinstance(dataset, torch.utils.data.Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
        classes = dataset.dataset.classes
    else:
        targets = dataset.targets
        classes = dataset.classes

    counter = Counter(targets)

    total = 0
    for class_idx in sorted(counter.keys()):
        class_name = classes[class_idx]
        count = counter[class_idx]
        total += count
        print(f"Class {class_idx} ({class_name}): {count} samples")

    print(f"Total samples: {total}")
    print("----------------------------------\n")

#Main
if __name__ == "__main__":
    configuration_file = "config.json"
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]

    print("Using ",configuration_file," configuration for training")
    config_json       = load_hyperparameters(os.path.dirname(os.path.abspath(__file__))+'/'+ configuration_file)
    print(config_json)

    if len(sys.argv) > 2:
        overwrite_model = sys.argv[2]
        config_json['model'] = overwrite_model
        print("Using model configuration provided by commandline parameter (",overwrite_model,")") 

    #Parse JSON Values
    #-----------------------------------------------------------------
    batch_size        = config_json['hparams']['batch_size']
    seed              = config_json['hparams']['seed']
    dropout_rate      = config_json['hparams']['dropout_rate']
    epochs            = config_json['hparams']['training_epochs']
    tile_size         = config_json['hparams']['tile_size']
    val_split         = config_json['dataloader']['validation_split']
    num_workers       = config_json['dataloader']['num_workers']
    gradient_clip_val = config_json['hparams']['gradient_clip_value']
    class_weight      = config_json['class_weight']
    lr                = config_json['optimizer']['learning_rate']
    use_wandb         = config_json['wandb']['use_wandb']
    loss              = config_json['loss']
    penalize_false_clean = float(config_json['penalize_false_clean'])
    #-----------------------------------------------------------------
    base_channels     = 32
    if  'base_channels' in config_json['hparams']:
          base_channels = config_json['hparams']['base_channels']
    print("Base channels ", base_channels)
    #-----------------------------------------------------------------
    final_dense_layer=512
    if  'final_dense_layer' in config_json['hparams']:
          final_dense_layer = config_json['hparams']['final_dense_layer']
    print("Final Dense Layer ", final_dense_layer)
    #-----------------------------------------------------------------
    model_name = config_json['model']
    if "name" in config_json:
             model_name = "%s_%s" % (config_json['name'],config_json['model'])
    #-----------------------------------------------------------------
    noise_std = 0.0        
    noise_clip = None
    if  'noise_std' in config_json['hparams']:
          noise_std = config_json['hparams']['noise_std']
    if  'noise_clip' in config_json['hparams']:
          noise_clip = config_json['hparams']['noise_clip']
    print("Simulated Noise STD ",noise_std,"/ CLIP ",noise_clip)
    #-----------------------------------------------------------------



    #Let's Go..
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    directory =  config_json['training_dataset']

    # Define the transform
    transform = transforms.Compose([
                                    transforms.ToTensor(),  # Convert images to PyTorch tensors
                                   #transforms.Normalize(mean=[0.2164, 0.2316, 0.2423, 0.2299], std=[0.0188, 0.0199, 0.0206, 0.0198]),
                                  ])

    H5PYFilename = '%s/dataset.h5' % directory 
    if (checkIfFileExists(H5PYFilename)):
          #If there is a .h5 file read directly from it to not spam I/O (Use DatasetConverter.py)
          from DatasetConverter import HDF5Dataset
          print("Using H5 dataset loader ",H5PYFilename)
          dataset = HDF5Dataset(H5PYFilename)
    else:
          #Normal .PNG decoding and loading
          print("Using Normal PNG dataset loader ",directory)
          dataset = RGBAImageFolder(root=directory,transform=transform,)


    #Just keep some of the classes     
    if ('selected_classes' in  config_json) and (config_json['selected_classes'] is not None) and (len(config_json['selected_classes'])>1) :
       print("Only selecting the classes ",config_json['selected_classes'], "for training")
       filter_dataset_classes(dataset, config_json['selected_classes'])

    # Optionally preload all samples to RAM (slow startup, fast epoch iteration)
    if ('cacheAllDataToRAM' in config_json['dataloader']) and (config_json['dataloader']['cacheAllDataToRAM']):
      # --- Sanity check: ensure there is enough available RAM ---
      available_ram = _get_available_ram_bytes()
      # On-disk footprint (quick lower bound)
      disk_bytes = _get_path_size_bytes(directory)
      # Heuristic RAM estimate based on sampling decoded items
      est_ram_bytes = _estimate_dataset_ram_bytes(dataset, sample_count=32)
    
      print("\n[RAM CHECK] Dataset on disk: ", _human_bytes(disk_bytes))
      if available_ram > 0:
            print("[RAM CHECK] System available RAM: ", _human_bytes(available_ram))
      else:
            print("[RAM CHECK] Could not determine available RAM (continuing without hard check).")
    
      # If we can measure available RAM, enforce a safety margin.
      if available_ram > 0:
            # Use the larger of the decoded estimate and 2x on-disk as a conservative requirement
            conservative_required = max(est_ram_bytes, int(disk_bytes * 2.0))
            safety_margin = 0.90  # do not consume more than 90% of available RAM
            limit = int(available_ram * safety_margin)
    
            print("[RAM CHECK] Estimated RAM needed to cache: ", _human_bytes(conservative_required))
            if conservative_required > limit:
                  print("\n[ERROR] Not enough available RAM to safely cache the entire dataset.")
                  print("        Required (est.): ", _human_bytes(conservative_required))
                  print("        Available:       ", _human_bytes(available_ram))
                  print("        Safety limit:    ", _human_bytes(limit), f" ({int(safety_margin*100)}% of available)")
                  print("\n        Tip: disable cacheAllDataToRAM or reduce dataset / tile size / dtype, or run on a machine with more RAM.\n")
                  sys.exit(1)
    
      dataset = RAMPreloadedDataset(dataset)
      num_workers = 1 #Set workers to 1 to avoid RAM duplication 


    # Calculate the sizes for training and validation sets
    dataset_size    = len(dataset)
    validation_size = int(val_split * dataset_size)
    train_size      = dataset_size - validation_size
    
    #Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    

    # Print class distribution BEFORE splitting
    print_class_distribution(dataset, title="Full Dataset")


    """
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(
                                              dataset,
                                              [train_size, validation_size],
                                              generator=torch.Generator().manual_seed(seed),
                                             )

    print_class_distribution(train_dataset, title="Training Set")
    print_class_distribution(val_dataset, title="Validation Set")
    """




    # -----------------------------------------------------------
    # Optional external validation dataset
    # -----------------------------------------------------------
    val_directory = None
    if "validation_dataset" in config_json and config_json["validation_dataset"]:
        val_directory = config_json["validation_dataset"]

    if val_directory is not None:
        print("Using explicit validation dataset:", val_directory)

        H5PYValFilename = f"{val_directory}/dataset.h5"

        if checkIfFileExists(H5PYValFilename):
            from DatasetConverter import HDF5Dataset
            val_dataset = HDF5Dataset(H5PYValFilename)
        else:
            val_dataset = RGBAImageFolder(root=val_directory, transform=transform)

        if ('selected_classes' in config_json) and config_json['selected_classes']:
            filter_dataset_classes(val_dataset, config_json['selected_classes'])

        if dataset.classes != val_dataset.classes:
            raise ValueError(f"Training/validation class mismatch: {dataset.classes} vs {val_dataset.classes}")

        if dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError("Training/validation class_to_idx mismatch")

        train_dataset = dataset
    else:
        print("No validation_dataset provided → using validation_split")

        dataset_size    = len(dataset)
        validation_size = int(val_split * dataset_size)
        train_size      = dataset_size - validation_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(seed),
        )

    # -----------------------------------------------------------

    print_class_distribution(train_dataset, title="Training Set")
    print_class_distribution(val_dataset, title="Validation Set")





    # Create DataLoaders
    train_loader = DataLoader(
                              train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True,
                              #collate_fn=metadata_collate_fn,
                             )

    val_loader  = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,  # Typically, we don't shuffle the validation set
                            num_workers=num_workers,
                            drop_last=True,
                            #collate_fn=metadata_collate_fn,
                           )

    #Print class names as a sanity check
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    #Get clean class which could be needed for extra penalization
    cleanClassID = 0
    for i in range(len(dataset.classes)):
           if (dataset.classes[i] == "class_clean") or (dataset.classes[i] == "Clean"):
              cleanClassID = i
    print(f"Clean class ID is : {cleanClassID}")

    """
    if class_weight:
        class_counts = Counter(train_dataset.dataset.targets)
        alpha = torch.tensor([1 / class_counts[i] for i in range(len(class_counts))])
        alpha = alpha / alpha.sum() 
        alpha = alpha.to(device)
    else:
        alpha = None
    """
    if class_weight:
        if isinstance(train_dataset, torch.utils.data.Subset):
            train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
        else:
            train_targets = train_dataset.targets

        class_counts = Counter(train_targets)
        alpha = torch.tensor(
            [1 / class_counts[i] for i in range(len(class_names))],
            dtype=torch.float32
        )
        alpha = alpha / alpha.sum()
        alpha = alpha.to(device)
    else:
        alpha = None


    # Initialize the classifier
    classifier = Classifier(
                            model=config_json['model'],
                            lr=lr,
                            num_classes=len(class_names),
                            tile_size=config_json['hparams']['tile_size'],
                            dropout_rate=dropout_rate,
                            penalize_false_clean=penalize_false_clean,
                            base_channels=base_channels,
                            final_dense_layer=final_dense_layer,
                            clean_class=cleanClassID,
                            noise_std=noise_std,
                            noise_clip=noise_clip)
    print(f"Learning rate: {lr}")
    
    model_type = classifier.type

    # Initialize a PyTorch Lightning trainer
    if use_wandb:
        loggers = [TensorBoardLogger("lightning_logs", name="classifier"),  WandbLogger(project=config_json['wandb']['project'], log_model=True,name=config_json['wandb']['name']+model_type+loss)]
    else:
        if (checkIfPathExists("tensorboard/")):
           os.system("echo \"Removing previous tensorboard logs..\" && rm -rf tensorboard/")
        loggers = [TensorBoardLogger("tensorboard", name=model_name)]

    trainer = pl.Trainer(
                         max_epochs=epochs,
                         logger=loggers,
                         #callbacks=[EarlyStopping(monitor='val_loss')],
                         accelerator=config_json['accelerator'],
                         devices=config_json['devices'],
                         gradient_clip_val=gradient_clip_val,
                         deterministic= (noise_std==0.0), #<- When there is noise switch to non deterministic to speed up training 
                       )

    #Train and log to console
    #------------------------------------------------------------------
    trainer.fit(classifier, train_loader, val_loader)


    #Save the model
    #------------------------------------------------------------------
    print("Saving the model")
    trainer.save_checkpoint('%s.pth' % model_name)

    # Evaluate dumped tiles (if directory exists)
    #------------------------------------------------------------------
    #tiles_dir = "tiles_dump_1760036129"  # replace or detect automatically
    #if os.path.isdir(tiles_dir):
    #    metrics = evaluate_dumped_tiles(classifier.model, tiles_dir, class_names, device=device)
    #    if metrics:
    #        with open("tile_evaluation.json", "w") as f:
    #            json.dump(metrics, f, indent=2)


    #Predictions
    #------------------------------------------------------------------
    print("Final model validation")
    classifier.eval()
    trainer.validate(classifier, val_loader)
    #------------------------------------------------------------------

    try:
      print("Removing previous confusion matrix data..")
      os.system("rm %s_confusion.json %s*.png" % (model_name,model_name) ) #Create zip of models

      print("Generating new confusion matrix data..")
      #Print confusion matrix numpy
      #------------------------------------------------------------------
      y_true = []
      y_pred = []
      """
      for x, y in val_loader:
        y_true.extend(y.numpy())
        y_pred.extend(classifier(x).argmax(dim=1).numpy())
      """
      for x, y in val_loader:
          x = x.to(classifier.device)
          y_true.extend(y.cpu().numpy())
          y_pred.extend(classifier(x).argmax(dim=1).detach().cpu().numpy())

      #num_classes = len(set(y_true))+1  # or classifier(x).shape[1]
      num_classes = len(dataset.classes)
      confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

      for true, pred in zip(y_true, y_pred):
        confusion_matrix[true, pred] += 1
      print(confusion_matrix)
      #------------------------------------------------------------------

      # Convert confusion matrix and classes to JSON-serializable form
      #------------------------------------------------------------------
      config_json["confusion_matrix"] = confusion_matrix.tolist()
      config_json["classes_int"] = [int(c) for c in set(y_true)]  # force Python ints
      config_json["classes"] = dataset.classes


      # Create extra summary JSON 
      #------------------------------------------------------------------
      print("Generating confusion matrix plot")
      confusion_json = {
                        "title": "%s / Tile Size = %u / Epochs = %u " % (model_name,tile_size,epochs),
                        "labels": dataset.classes, 
                        "matrix": confusion_matrix.tolist() 
                       }
      with open('%s_confusion.json' % model_name, 'w') as f:
       json.dump(confusion_json, f, indent=2)
      os.system("python3 plotTool.py %s_confusion.json" % model_name)
    except Exception as e:
      print("Failed generating a confusion matrix : ",e)
      os.system("echo \"Failed\" > %s_confusion.json" % model_name)


    #Save the JSON 
    #------------------------------------------------------------------
    print("Saving the JSON")
    with open('%s.json' % model_name, 'w') as f:
       json.dump(config_json, f, indent=2)


    #Update symbolic links 
    #------------------------------------------------------------------
    #os.system("rm last.pth last.json && ln -s %s.pth last.pth && ln -s %s.json last.json" % (model_name,model_name) )
    #No longer needed

    # Build zip filename with timestamp
    #------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{model_name}_{timestamp}.zip"

    print("Saving everything as %s archive" % zip_name)
    os.system("mkdir models/")
    os.system("zip -r models/%s %s.json %s_confusion.json %s*.png %s.pth tensorboard/*/*/*" % (zip_name,model_name,model_name,model_name,model_name) ) #Create zip of models


    print('To upload ALL models copy/paste:') 
    print("scp -P 2222 models/*.zip ammar@ammar.gr:/home/ammar/public_html/magician/ckpts2")  

    print('To upload last training results copy/paste:') 
    print("scp -P 2222 models/%s ammar@ammar.gr:/home/ammar/public_html/magician/ckpts2" % zip_name)  
    
