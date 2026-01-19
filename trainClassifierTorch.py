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
from torchvision.models import resnet18,vit_b_16, convnext_tiny,resnext50_32x4d,efficientnet_v2_s,ResNet18_Weights,ResNeXt50_32X4D_Weights,ViT_B_16_Weights,ConvNeXt_Tiny_Weights,EfficientNet_V2_S_Weights,Swin_V2_T_Weights,RegNet_Y_800MF_Weights
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
        self.AoLP=AoLP
        self.DoLP=DoLP
        self.Unpolarized = Unpolarized
        self.clean_class  = clean_class  
        self.penalize_false_clean = penalize_false_clean
        #-----------------------------------------
        self.noise_std  = noise_std
        self.noise_clip = noise_clip

        #RESNEXT
        if self.type == 'resnext50':
            self.model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
            #self.model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            self.model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(2048, num_classes)
        elif self.type == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512, num_classes)
        elif self.type == 'convnext_tiny':
            self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4))
            self.model.classifier[2]  = nn.Linear(768, num_classes)
        elif self.type == 'efficientnet_v2_s':
            self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier[1]  = nn.Linear(1280, num_classes,bias = True)
        elif self.type == 'swin_v2_t':
            self.model = torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4))
            self.model.head = nn.Linear(768, num_classes)
        elif self.type == 'regnet_y_800mf':
            self.model = torchvision.models.regnet_y_800mf(weights=RegNet_Y_800MF_Weights.DEFAULT)
            self.model.stem[0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #import ipdb; ipdb.set_trace()
            self.model.fc = nn.Linear(784, num_classes)
        elif ('custom' in self.type) or ('cnn' in self.type):
            self.model = CustomCNN(in_channels=4, intended_tile_size=tile_size, num_classes=num_classes, dropout_rate=dropout_rate, base_channels=self.base_channels, final_dense_layer=self.final_dense_layer)
        else:
            raise ValueError(f"Unsupported model type: {model}. Supported types are 'resnext50', 'resnet18', 'convnext_tiny', 'efficientnet_v2_s', 'swin_v2_t', 'regnet_y_800mf'.")

        if (load_checkpoint is not None):
           self.model = Classifier.load_from_checkpoint(load_checkpoint)

        #Resnet18
        #self.criterion = CategoricalFocalLoss(gamma=gamma, alpha=None)
        if loss == 'focal':
            self.criterion = CategoricalFocalLoss(gamma=2.0, alpha=None)
        elif loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss}. Supported types are 'focal' and 'cross_entropy'.")
        #Vitb16
        #self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        #self.model.conv_proj = nn.Conv2d(4, 768, kernel_size=(16, 16), stride=(16, 16))
        #self.model.heads[0] = nn.Linear(768, num_classes)
        self.accuracy  = Accuracy(task='MULTICLASS',  num_classes=num_classes)
        self.recall    = Recall(task='MULTICLASS',    num_classes=num_classes)
        self.precision = Precision(task='MULTICLASS', num_classes=num_classes)
        self.auroc     = AUROC(task='MULTICLASS',     num_classes=num_classes)
        #self.confusion_matrix = ConfusionMatrix(task="multiclass",num_classes=num_classes)

    def add_input_noise(self, x):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(x) * self.noise_std
            if self.noise_clip is not None:
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            x = x + noise
        return x


    def forward(self, x):
        #x = self.val_transformations(image=x)['image']
        #Normalize with mean and std
        #x = (x - torch.tensor(self.mean).reshape(1, 4, 1, 1)) / torch.tensor(self.std).reshape(1, 4, 1, 1)

        if (self.AoLP or self.DoLP):
          stokes = self.calculate_stokes(x)
        # Calculate DoLP and AoLP
          if (self.DoLP):
            DoLP = self.calculate_DoLP(stokes).reshape(x.shape[0], 1, x.shape[2], x.shape[3])
            x    = torch.cat((x, DoLP), dim=1)
          if (self.AoLP):
            AoLP = self.calculate_AoLP(stokes).reshape(x.shape[0], 1, x.shape[2], x.shape[3])
            x    = torch.cat((x, AoLP), dim=1)
        
        if (self.Unpolarized):
           mon = x.mean(dim=1, keepdim=True)  # Average the input tensor across the channel dimension
           x   = torch.cat((x, mon),dim=1)  # Concatenate the average tensor to the input tensor

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = self.add_input_noise(x)  

        if (self.AoLP or self.DoLP):
           stokes = self.calculate_stokes(x)
           # Calculate DoLP and AoLP
           if (self.DoLP):
            DoLP = self.calculate_DoLP(stokes).reshape(x.shape[0], 1, x.shape[2], x.shape[3])  # Reshape to match input dimensions
            x    = torch.cat((x, DoLP), dim=1)
           if (self.AoLP):
            AoLP = self.calculate_AoLP(stokes).reshape(x.shape[0], 1, x.shape[2], x.shape[3])  # Reshape to match input dimensions
            x    = torch.cat((x, AoLP), dim=1)

        if (self.Unpolarized):
           mon = x.mean(dim=1, keepdim=True)  # Average the input tensor across the channel dimension
           x   = torch.cat((x, mon),dim=1)  # Concatenate the average tensor to the input tensor

        y_hat = self.model(x)
        base_loss  = self.criterion(y_hat, y)

        if self.penalize_false_clean > 0.0:
            # Compute extra penalty for “false clean”
            pred_probs = F.softmax(y_hat, dim=1)
            # Consider all samples whose true class is NOT the clean class
            non_clean_mask = (y != self.clean_class)
        
            penalty_strength = float(self.penalize_false_clean)
        
            if non_clean_mask.any():
                p_clean = pred_probs[non_clean_mask, self.clean_class]
                # strong penalty that grows as p_clean -> 1
                false_clean_loss = -torch.log(1.0 - p_clean + 1e-8).mean()
                loss = base_loss + penalty_strength * false_clean_loss
            else:
                loss = base_loss
        else:
            loss = base_loss

        self.log('train_loss', loss,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if (self.AoLP or self.DoLP):
          stokes = self.calculate_stokes(x)
          # Calculate DoLP and AoLP
          if (self.DoLP):
            DoLP = self.calculate_DoLP(stokes).reshape(x.shape[0], 1, x.shape[2], x.shape[3])
            x    = torch.cat((x, DoLP), dim=1)
          if (self.AoLP):
            AoLP = self.calculate_AoLP(stokes).reshape(x.shape[0], 1, x.shape[2], x.shape[3])
            x    = torch.cat((x, AoLP), dim=1)

        if (self.Unpolarized):
          mon = x.mean(dim=1, keepdim=True)  # Average the input tensor across the channel dimension
          x   = torch.cat((x, mon),dim=1)  # Concatenate the average tensor to the input tensor
        
        y_hat = self.model(x)
        loss  = self.criterion(y_hat, y)
        self.log('val_loss', loss,sync_dist=True)

        self.accuracy.update(y_hat, y) 
        self.log('val_accuracy',  self.accuracy.compute(),  prog_bar=True, sync_dist=True)

        self.recall.update(y_hat, y)   
        self.log('val_recall',    self.recall.compute(),    prog_bar=True, sync_dist=True) 

        self.precision.update(y_hat, y)
        self.log('val_precision', self.precision.compute(), prog_bar=True, sync_dist=True)

        self.auroc.update(y_hat, y)
        self.log('val_auroc', self.auroc.compute(), prog_bar=True, sync_dist=True)

        """
        self.log('val_recall',    self.recall(y_hat, y),    prog_bar=True, sync_dist=True) #Why is recall /precision always the same
        self.log('val_precision', self.precision(y_hat, y), prog_bar=True, sync_dist=True)
        self.log('val_auroc',     self.auroc(y_hat, y), prog_bar=True, sync_dist=True)
        """
        
        return loss

    def calculate_stokes(self, x):
        """
        Calculate the Stokes parameters from the input tensor.
        Assuming x is a tensor of shape (batch_size, 4, height, width).
        """
        # Stokes parameters
        S0 = x[:, 0, :, :]
        S1 = x[:, 1, :, :] - x[:, 2, :, :]  # Difference between two channels
        S2 = x[:, 1, :, :] + x[:, 2, :, :]  # Sum of two channels
        S3 = x[:, 3, :, :]  # Assuming the fourth channel is the third Stokes parameter
        return torch.stack((S0, S1, S2, S3), dim=1)  # Shape: (batch_size, 4, height, width)
    
    def calculate_DoLP(self, x):
        """
        Calculate the Degree of Linear Polarization (DoLP) from the Stokes parameters.
        Assuming x is a tensor of shape (batch_size, 4, height, width).
        """
        S0 = x[:, 0, :, :]
        S1 = x[:, 1, :, :]
        S2 = x[:, 2, :, :]
        S3 = x[:, 3, :, :]
        
        # Calculate DoLP
        DoLP = torch.sqrt(S1**2 + S2**2) / S0
        return DoLP
    
    def calculate_AoLP(self, x):
        """
        Calculate the Angle of Linear Polarization (AoLP) from the Stokes parameters.
        Assuming x is a tensor of shape (batch_size, 4, height, width).
        """
        S1 = x[:, 1, :, :]
        S2 = x[:, 2, :, :]
        
        # Calculate AoLP
        AoLP = 0.5 * torch.atan2(S2, S1)
        return AoLP
    
    def on_validation_epoch_end(self):
        #self.log('val_accuracy_epoch', self.accuracy.compute(), prog_bar=True)
        self.accuracy.reset()  # Reset the metric for the next epoch
        
    
    def on_train_epoch_end(self):
        #self.log('train_accuracy_epoch', self.accuracy.compute(), prog_bar=True)
        self.accuracy.reset()  # Reset the metric for the next epoch
    
    #def on_train_epoch_start(self):
        #Unfreeze the model
        #if self.current_epoch == 2:
            #print("Unfreezing the model")
        #    for param in self.model.parameters():
        #        param.requires_grad = True
        
    def configure_optimizers(self):
        self.hparams.lr = self.lr
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return optimizer#{"optimizer": optimizer, "lr_scheduler": scheduler}
    



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


class RGBAImageFolder(datasets.DatasetFolder):
    def __init__(self, root, transform=None):
        super(RGBAImageFolder, self).__init__(
                root,
                loader=load_rgba_image,  # Use custom loader for RGBA images
                extensions=('png', 'jpg', 'jpeg'),  # Add supported image extensions
                transform=transform
            )

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
    directory =  config_json['directory']

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


    # Calculate the sizes for training and validation sets
    dataset_size    = len(dataset)
    validation_size = int(val_split * dataset_size)
    train_size      = dataset_size - validation_size
    
    #Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(
                                              dataset,
                                              [train_size, validation_size],
                                              generator=torch.Generator().manual_seed(seed),
                                             )

    # Create DataLoaders
    train_loader = DataLoader(
                              train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True
                             )

    val_loader  = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,  # Typically, we don't shuffle the validation set
                            num_workers=num_workers,
                            drop_last=True
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

    if class_weight:
        class_counts = Counter(train_dataset.dataset.targets)
        alpha = torch.tensor([1 / class_counts[i] for i in range(len(class_counts))])
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
      for x, y in val_loader:
        y_true.extend(y.numpy())
        y_pred.extend(classifier(x).argmax(dim=1).numpy())

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

    print('To upload results copy/paste:') 
    print("scp -P 2222 models/%s ammar@ammar.gr:/home/ammar/public_html/magician/ckpts" % zip_name)  
    
