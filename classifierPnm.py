#!/usr/bin/python3

""" 
Author : "Nikos Vasilikopoulos, Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 

ClassifierPnm and its inference helpers (tiling, heatmaps, majority voting,
erosion, model scanning). Split out of liveClassifierTorch.py so live/CLI
streaming, the annotator, ROS and ModelDownload share one maintainable core.
liveClassifierTorch.py re-exports everything for backward compatibility.
"""


import cv2
import numpy as np
# -> 
#python3 evaluate.py tile_classifier.keras /home/ammar/Documents/Programming/Magician/src/python/classifier/average100/sample01.pnm
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import json
import time
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from trainMagicianVisionClassifierTorch import Classifier
#from numba import njit #Test
from readData import readPolarPNMToRGBALive#,readPolarPNMToRGBAResized
from SharedMemoryManager import SharedMemoryManager
from torch.nn import functional as F
#--------------------------------------------------------------------------
def dumpListAsCSV(theList,fields,theFilename):
   import csv
   with open(theFilename, 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(theList)
#--------------------------------------------------------------------------
def log_performance(filename, model_name, step, tile_size,
                    majorityVote, maxProbabilityThreshold,
                    num_predictions, hz):
    """Append a performance log line to a file. Creates header if missing."""
    file_exists = os.path.exists(filename)

    with open(filename, "a") as f:
        # Write header first time
        if not file_exists:
            f.write("model_name, step, tile_size, majorityVote, "
                    "maxProbabilityThreshold, num_predictions, framerate_hz\n")

        # Write values
        f.write(f"{model_name}, {step}, {tile_size}, {int(majorityVote)}, "
                f"{maxProbabilityThreshold}, {num_predictions}, {hz:.4f}\n")
#--------------------------------------------------------------------------
def load_classes_json(filename):
    with open(filename, 'r') as f:
        classes = json.load(f)
    return classes
#--------------------------------------------------------------------------
def checkIfPathIsDirectory(filename):
    return os.path.isdir(filename) 
#--------------------------------------------------------------------------
def checkIfFileExists(filename):
    return os.path.isfile(filename)
#--------------------------------------------------------------------------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Drawing routines to decorate output to make classification understandable
#--------------------------------------------------------------------------
def getNDifferentColors(n):
    """
    Generate n visually distinct RGB colors as (R, G, B) tuples.
    No external libraries are used.
    """
    if n <= 0:
        return []

    class_colors = []

    # Predefined base colors (optional)
    base_colors = [
        (0, 0, 255),     # Blue
        (255, 0, 0),     # Red
        (255, 255, 0),   # Yellow
        (0, 255, 0),     # Green
    ]

    # If n fits in predefined colors, return the subset
    if n <= len(base_colors):
        return base_colors[:n]

    # Otherwise, add base colors and generate more
    class_colors.extend(base_colors)

    # Generate additional distinct colors
    step = int(255 / ((n - len(base_colors)) ** (1/3))) or 1
    for r in range(0, 256, step):
        for g in range(0, 256, step):
            for b in range(0, 256, step):
                if len(class_colors) >= n:
                    return class_colors[:n]
                color = (r, g, b)
                if color not in class_colors:
                    class_colors.append(color)

    return class_colors[:n]
#------------------------------------------------------------------------
def draw_cross(image, center, half_size, color):
    y, x = center
    # Draw horizontal line of the cross
    image[y, x - half_size:x + half_size + 1] = color
    # Draw vertical line of the cross
    image[y - half_size:y + half_size + 1, x] = color
    return image
#------------------------------------------------------------------------
def draw_up(image, center, half_size, color):
    y, x = center
    for i in range(half_size + 1):
        image[y - i, x - i:x + i + 1] = color
    return image
#------------------------------------------------------------------------
def draw_down(image, center, half_size, color):
    y, x = center
    for i in range(half_size + 1):
        image[y + i, x - i:x + i + 1] = color
    return image
#------------------------------------------------------------------------
def draw_left(image, center, half_size, color):
    y, x = center
    for i in range(half_size + 1):
        if 0 <= y - i < image.shape[0]:
            image[y - i, x - i] = color
        if 0 <= y + i < image.shape[0]:
            image[y + i, x - i] = color
    return image
#------------------------------------------------------------------------
def draw_right(image, center, half_size, color):
    y, x = center
    for i in range(half_size + 1):
        if 0 <= y - i < image.shape[0]:
            image[y - i, x + i] = color
        if 0 <= y + i < image.shape[0]:
            image[y + i, x + i] = color
    return image
#------------------------------------------------------------------------
def draw_X(image, center, half_size, color):
    y, x = center
    for i in range(-half_size, half_size + 1):
        if 0 <= y + i < image.shape[0] and 0 <= x + i < image.shape[1]:
            image[y + i, x + i] = color  # main diagonal
        if 0 <= y + i < image.shape[0] and 0 <= x - i < image.shape[1]:
            image[y + i, x - i] = color  # anti-diagonal
    return image
#------------------------------------------------------------------------
def printLabels(heatmap,classes,class_colors):
    x = 30
    y = 30 
    for classID in range(len(classes)):
       cv2.putText(heatmap, classes[classID], (x-2,y-2) , cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0) , 5)
       cv2.putText(heatmap, classes[classID], (x-1,y-1) , cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255) , 5)
       cv2.putText(heatmap, classes[classID], (x,y) , cv2.FONT_HERSHEY_SIMPLEX, 1.5, class_colors[classID], 5)
       y += 40
#------------------------------------------------------------------------


def verifyTileNumber(numberOfTiles , original_image, tile_size, step):
    height, width, _ = original_image.shape
    tilesH = (height - tile_size) // step + 1
    tilesW = (width  - tile_size) // step + 1
    expected_tiles = (tilesH) * (tilesW)

    if numberOfTiles != expected_tiles:
        return False
    return True

#------------------------------------------------------------------------


@torch.no_grad()
def generate_predictionStatistics(predictions,num_classes):
    num_samples = predictions.shape[0]
    
    minimums = list()
    maximums = list()
    stds     = list()
    means    = list()
    
    for cID in range(num_classes):
      minimums.append(np.min(predictions[:,cID]))
      maximums.append(np.max(predictions[:,cID]))
      stds.append(np.std(predictions[:,cID]))
      means.append(np.mean(predictions[:,cID]))

    return minimums,maximums,stds,means

@torch.no_grad()
def sliding_window_majority_vote(predictions, window_size=4):
    """
    Apply majority voting to enforce consistency in a 1D array of predictions.
    
    Args:
        predictions (np.array): 1D array of predicted class labels.
        window_size (int): Size of the sliding window (must be odd).
    
    Returns:
        np.array: Smoothed 1D array of predictions.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    
    # Pad the predictions to handle edges
    pad_size = window_size // 2
    padded_predictions = np.pad(predictions, (pad_size,), mode='edge')
    
    # Apply sliding window majority voting
    smoothed_predictions = np.zeros_like(predictions)
    for i in range(len(predictions)):
        window = padded_predictions[i:i + window_size]
        unique, counts = np.unique(window, return_counts=True)
        smoothed_predictions[i] = unique[np.argmax(counts)]
    
    return smoothed_predictions

@torch.no_grad()
def majority_vote_2d_pytorch(predictions_2d, tilesHorizontally, tilesVertically, window_size=3):
    """
    Efficient 2D majority voting using PyTorch unfold.
    Supports arbitrary tile grids and odd window sizes.
    """

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    # Convert to torch tensor
    if isinstance(predictions_2d, torch.Tensor):
        preds = predictions_2d.detach().cpu().view(-1).to(dtype=torch.int64)
    else:
        preds = torch.tensor(predictions_2d, dtype=torch.int64)

    expected_len = int(tilesHorizontally) * int(tilesVertically)
    n = preds.numel()

    # Auto-adjust mismatch (pad or truncate)
    if n < expected_len:
        pad_value = int(torch.mode(preds).values.item()) if n > 0 else -1
        preds = F.pad(preds, (0, expected_len - n), value=pad_value)
        print(f"[majority_vote_2d_pytorch] WARNING: padded predictions from {n} to {expected_len}")
    elif n > expected_len:
        preds = preds[:expected_len]
        print(f"[majority_vote_2d_pytorch] WARNING: truncated predictions from {n} to {expected_len}")

    # Reshape into 2D grid
    grid = preds.view(tilesVertically, tilesHorizontally)

    pad_size = window_size // 2

    # Add fake channel dimension so replicate padding works
    grid = grid.unsqueeze(0)  # [1, H, W]
    padded = F.pad(grid, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    padded = padded.squeeze(0)  # back to [H', W']

    # Unfold sliding windows
    unfolded = padded.unfold(0, window_size, 1).unfold(1, window_size, 1)  # [H, W, win, win]

    # Flatten window into 1D per location
    H, W = unfolded.shape[:2]
    unfolded = unfolded.contiguous().view(H, W, -1)

    # Compute majority vote (mode)
    mode_values, _ = torch.mode(unfolded, dim=2)
    return mode_values.cpu().numpy().astype('int64')

@torch.no_grad()
def remove_large_blobs(binary_image, max_size):
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an output binary image
    output_image = np.zeros_like(binary_image)  
    
    for i in range(1, num_labels):  # Skip the background label 0
        if stats[i, cv2.CC_STAT_AREA] <= max_size:
            output_image[labels == i] = 255
    
    return output_image

@torch.no_grad()
def flood_fill_with_threshold(image, seed_point, new_value, pixel_threshold):
    """
    Perform a flood fill on a monochrome image if the number of pixels affected exceeds a threshold.

    :param image: Input grayscale image (numpy array).
    :param seed_point: Tuple (x, y) representing the seed point for the flood fill.
    :param new_value: New color value to apply during flood fill.
    :param pixel_threshold: Minimum number of pixels affected to perform the flood fill.
    :return: Tuple (number_of_affected_pixels, modified_image) or (0, original_image) if threshold not met.
    """
    # Ensure the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image")

    # Create a copy of the image to avoid modifying the original image
    image_copy = image.copy()

    # Initialize the mask for flood fill
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Perform the flood fill
    num_affected_pixels, _, _, rect = cv2.floodFill(image_copy, mask, seed_point, new_value)

    # Check if the number of affected pixels exceeds the threshold
    if num_affected_pixels > pixel_threshold:
        return num_affected_pixels, image_copy
    else:
        return 0, image

@torch.no_grad()
def remove_byflood(occupancy):
    for y in range(occupancy.shape[0]):
        for x in range(occupancy.shape[0]):
              if (occupancy[y,x]>0):
                  num,occupancy = flood_fill_with_threshold(occupancy, (x,y) , 0, 32)
    return occupancy

@torch.no_grad()
def mask_outside_rectangleA(image, rect):
    # Create a copy of the image to avoid modifying the original
    masked_image = image.copy()
    
    # Get the rectangle parameters
    x, y, width, height = rect

    # Create a mask with the same dimensions as the image, initialized to 0 (black)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Set the pixels within the rectangle to 1 (white)
    mask[y:y+height, x:x+width] = 1

    # Apply the mask to the image
    masked_image = masked_image * mask

    return masked_image

@torch.no_grad()
def mask_outside_rectangle(image, rect): 
    # Create a copy of the image to avoid modifying the original 
    
    # Get the rectangle parameters
    x, y, width, height = rect

    # Set pixels above the rectangle to 0
    image[:y, :] = 0
    
    # Set pixels below the rectangle to 0
    image[y+height:, :] = 0
    
    # Set pixels to the left of the rectangle to 0
    image[:, :x] = 0
    
    # Set pixels to the right of the rectangle to 0
    image[:, x+width:] = 0

    return image

@torch.no_grad()
def detect_4x4_rectangle(heatmap, occupancy, thresholdMin=60.0, thresholdMax=68.00):
    #occupancy = remove_large_blobs(occupancy,45)
    occupancy = remove_byflood(occupancy)
    occupancy = mask_outside_rectangle(occupancy, (10,10,occupancy.shape[1]-20,occupancy.shape[0]-20) )
    final_occupancy = np.zeros_like(occupancy)  

    scaleX = heatmap.shape[0] / occupancy.shape[0] 
    scaleY = heatmap.shape[1] / occupancy.shape[1]

    data    = [
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 255,   255,   255,   255, 0],
               [0, 0, 0, 255,   255,   255,   255, 0],
               [0, 0, 0, 255,   255,   255,   255, 0],
               [0, 0, 0, 255,   255,   255,   255, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0], 
              ]
 

    # Create a NumPy array with dtype uint8
    pattern = np.array(data, dtype=np.uint8)
    pattern_height, pattern_width = pattern.shape
    pattern_pixels = pattern_width * pattern_height
 
    dataInv  = [
               [255, 255, 255, 255, 255, 255],
               [255, 0,   0,   0,     0, 255],
               [255, 0,   0,   0,     0, 255],
               [255, 0,   0,   0,     0, 255],
               [255, 0,   0,   0,     0, 255],
               [255, 255, 255, 255, 255, 255],
              ]

    patternInv = np.array(dataInv, dtype=np.uint8)
    patternInv_height, patternInv_width = patternInv.shape
    patternInv_pixels = patternInv_width * patternInv_height
    windowInv  = patternInv 

    # Scan the image with a sliding window
    img_height, img_width = occupancy.shape
    
    detected_coordinates = []
    
    for y in range(img_height - pattern_height + 1):
        for x in range(img_width - pattern_width + 1):

            window = occupancy[y:y + pattern_height, x:x + pattern_width]
            score = np.sum(np.abs(window - pattern)) / pattern_pixels
            if (thresholdMin <= score) and (score <= thresholdMax):  # Allow some tolerance 
              final_occupancy[y,x]=123
              draw_cross(heatmap, (int((y+4)*scaleY), int((x+3)*scaleX)) , 10, (0,255,255) )
              windowInv[:,:] = occupancy[y+1:y+1 + patternInv_height, x+2:x+2 + patternInv_width]#   window[2:,1:7]
              windowInv      = 255 - windowInv
              score2 = np.sum(np.abs(windowInv - patternInv)) / patternInv_pixels
              detected_coordinates.append((x, y))
              if score2<80:
              #if np.array_equal(windowInv, patternInv):
                #detected_coordinates.append((x, y))
                draw_cross(heatmap, (int((y+4)*scaleY), int((x+3)*scaleX)) , 15, (0,0,255) )
                final_occupancy[y,x]=255

    #occupancy[:,:] = final_occupancy[:,:]
    return detected_coordinates, occupancy



def dump_tiles_as_png(rgba_image, predictions, classes, tile_size, step, output_dir=None):
    """
    Dumps each tile from the RGBA image as a PNG file with its predicted class in the filename.

    Args:
        rgba_image (torch.Tensor or np.ndarray): The full RGBA image tensor (H x W x 4).
        predictions (np.ndarray or list): Flattened array of predicted class indices.
        classes (list): List of class names.
        tile_size (int): Size of each tile.
        step (int): Step size between tiles.
        output_dir (str): Optional output directory path. If None, creates a timestamped folder.
    """
    # Convert to numpy if needed
    if isinstance(rgba_image, torch.Tensor):
        rgba_np = rgba_image.detach().cpu().numpy()
    else:
        rgba_np = np.array(rgba_image)

    # Ensure correct shape
    if rgba_np.ndim == 3 and rgba_np.shape[2] == 4:
        pass
    elif rgba_np.ndim == 4 and rgba_np.shape[0] == 1:
        rgba_np = rgba_np[0]
    elif rgba_np.ndim == 3 and rgba_np.shape[0] == 4:
        rgba_np = np.transpose(rgba_np, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected image shape: {rgba_np.shape}")

    rgba_np = np.clip(rgba_np, 0, 255).astype(np.uint8)

    H, W, _ = rgba_np.shape
    y_indices = list(range(0, H - tile_size, step))
    x_indices = list(range(0, W - tile_size, step))

    predictions_np = np.array(predictions).astype(np.int32).flatten()

    # Output directory
    if output_dir is None:
        ts = int(time.time())
        output_dir = f"tiles_dump_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    idx = 0
    saved = 0

    for y in y_indices:
        for x in x_indices:
            try:
                if idx >= len(predictions_np):
                    pred_cls = -1
                else:
                    pred_cls = int(predictions_np[idx])

                tile = rgba_np[y:y+tile_size, x:x+tile_size, :]

                # Get class name
                if 0 <= pred_cls < len(classes):
                    cls_name = str(classes[pred_cls]).replace(" ", "_")
                else:
                    cls_name = f"cls{pred_cls}"

                filename = f"tile_{idx:06d}_y{y}_x{x}_cls{pred_cls}_{cls_name}.png"
                filepath = os.path.join(output_dir, filename)

                cv2.imwrite(filepath, tile)
                saved += 1
            except Exception as e:
                print(f"Failed saving tile {idx} at ({x},{y}): {repr(e)}")
            finally:
                idx += 1

    print(f"Saved {saved} tiles to '{output_dir}'")
    return output_dir


@torch.no_grad()
def tile_and_cast_data_torch(image, tile_size=24, step=2):
    # Convert image to tensor (if it's a NumPy array)
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    #print("tile_and_cast_data_torch input image:",image.shape)
    #image H , W , C | torch.Size([1024, 1224, 4])

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image  = image.to(device)

    # Get the dimensions
    height, width, channels = image.shape

    # Rearrange dimensions to (C, H, W) for unfold operation
    image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # Use unfold to extract patches efficiently
    tiles = image.unfold(1, tile_size, step).unfold(2, tile_size, step)  # Shape: (C, num_tiles_y, num_tiles_x, tile_size, tile_size)

    # Rearrange dimensions to match original output format (N, tile_size, tile_size, C)
    tiles = tiles.permute(1, 2, 3, 4, 0).contiguous()

    # Reshape into (N, tile_size, tile_size, channels)
    tiles = tiles.view(-1, tile_size, tile_size, channels)

    return tiles



@torch.no_grad()
def generate_heatmap(predictions, confidences, class_id_to_name, class_id_to_color, cleanClassID,
                     rgba_image, tile_size=24, step=2):
    """
    Generate color-coded heatmap using integer class IDs only.
    Matches tiling produced by tile_and_cast_data_torch.
    """
    original_image = torch.as_tensor(rgba_image, dtype=torch.uint8)
    height, width, _ = original_image.shape

    y_indices = torch.arange(0, height - tile_size + 1, step)
    x_indices = torch.arange(0, width  - tile_size + 1, step)
    tilesH = len(y_indices)
    tilesW = len(x_indices)
    expected_tiles = tilesH * tilesW

    #if len(predictions) != expected_tiles:
    if not (verifyTileNumber(len(predictions), original_image, tile_size, step)):
        print(f"⚠️ generate_heatmap warning: predictions={len(predictions)} tiles expected={expected_tiles}")

    occupancy = torch.full((tilesH, tilesW), 255, dtype=torch.uint8)
    responses = {"points": [], "classes": [], "classIDs": [],  "confidences": []}
    heatmap = original_image[:, :, :3].clone()
    activations = torch.zeros(len(class_id_to_name), dtype=torch.int32)

    predicted_classes = torch.as_tensor(predictions, dtype=torch.int32)
    totalActivations = 0
    idx = 0
    num_preds = len(predicted_classes)
    half_tile_size = tile_size // 2

    bg_prob_sum = 0.0
    bg_count    = 0

    for vTile, y in enumerate(y_indices):
        for hTile, x in enumerate(x_indices):
            if idx >= num_preds:
                break

            predicted_class = int(predicted_classes[idx])

            # Skip invalid IDs gracefully
            if predicted_class < 0 or predicted_class >= len(class_id_to_name):
                idx += 1
                continue

            if predicted_class != cleanClassID:
                totalActivations += 1
                color = class_id_to_color[predicted_class]

                activationCoordinateX = int(x + half_tile_size)
                activationCoordinateY = int(y + half_tile_size)

                confidence = float(confidences[idx])


                color = (color.float() * confidence).clamp(0, 255).to(torch.uint8)

                draw_cross(heatmap, (activationCoordinateY, activationCoordinateX), 10, color)


                activations[predicted_class] += 1
                responses["points"].append( (activationCoordinateX, activationCoordinateY) )
                responses["classes"].append(class_id_to_name[predicted_class])
                responses["classIDs"].append(int(predicted_class))
                responses["confidences"].append(confidence)
                try:
                  occupancy[vTile, hTile] = 0
                except Exception as e:
                  print("Failed setting occupancy:", repr(e))
            else:
                bg_prob_sum += float(confidences[idx])
                bg_count    += 1

            idx += 1

        if idx >= num_preds:
            break

    responses["background_avg_prob"] = bg_prob_sum / bg_count if bg_count > 0 else 0.0

    print(f"{totalActivations}/{num_preds} activations")
    print("Per-class activations:", activations.tolist())
    return heatmap.cpu().numpy(), occupancy.cpu().numpy(), responses


@torch.no_grad()
def process_predictions_erode(predictions, confidences, class_id_to_name, cleanClassID, rgba_image, tile_size=48, step=14, erosion_kernel=1, erosion_threshold=1):
    """
    erosion_kernel: radius in tiles for neighbor search
    erosion_threshold: minimum number of neighbors required to keep a prediction
    """

    original_image   = torch.as_tensor(rgba_image, dtype=torch.uint8)
    height, width, _ = original_image.shape

    y_indices = torch.arange(0, height - tile_size + 1, step)
    x_indices = torch.arange(0, width - tile_size + 1, step)
    tilesH           = len(y_indices)
    tilesW           = len(x_indices)
    expected_tiles   = tilesH * tilesW

    #if len(predictions) != expected_tiles:
    if not (verifyTileNumber(len(predictions), original_image, tile_size, step)):
        print(f"⚠️ Warning: predictions={len(predictions)} tiles expected={expected_tiles}")

    occupancy = torch.full((tilesH, tilesW), 255, dtype=torch.uint8)
    responses = {"points": [], "classes": [], "classIDs": [],  "confidences": []}
    heatmap = original_image[:, :, :3].clone()
    activations = torch.zeros(len(class_id_to_name), dtype=torch.int32)

    predicted_classes = torch.as_tensor(predictions, dtype=torch.int32)
    totalActivations = 0
    idx = 0
    num_preds = len(predicted_classes)
    half_tile_size = tile_size // 2

    # ---- FIRST PASS: collect raw detections ----
    coord_map = []  # stores (vTile, hTile, predicted_class, confidence)
    
    for vTile, y in enumerate(y_indices):
        for hTile, x in enumerate(x_indices):

            if idx >= num_preds:
                break

            predicted_class = int(predicted_classes[idx])

            # Skip invalid IDs gracefully
            if 0 <= predicted_class < len(class_id_to_name):
                if predicted_class != cleanClassID:
                    confidence = float(confidences[idx]) 
                    coord_map.append((vTile, hTile, predicted_class, confidence))
                    totalActivations += 1
                    activations[predicted_class] += 1
                    occupancy[vTile, hTile] = 0

            idx += 1
        if idx >= num_preds:
            break

    print(f"{totalActivations}/{num_preds} activations (before erosion)")

    # ---- EROSION MASK CONSTRUCTION ----
    eroded_mask = torch.zeros_like(occupancy, dtype=torch.uint8)  # 1 = keep

    for (v, h, clsID, confidence) in coord_map:
        # Count neighbors within erosion_kernel
        count = 0
        for dv in range(-erosion_kernel, erosion_kernel + 1):
            for dh in range(-erosion_kernel, erosion_kernel + 1):
                nv, nh = v + dv, h + dh
                if 0 <= nv < tilesH  and 0 <= nh < tilesW :
                    if occupancy[nv, nh] == 0:
                        count += 1
                        #print("Count ",count," ",nv,",",nh," erosion_kernel=",erosion_kernel," erosion_threshold=",erosion_threshold)

        # If enough neighbors, keep it
        if count >= erosion_threshold:
            eroded_mask[v, h] = 1

    # ---- SECOND PASS: rebuild filtered responses ----
    filtered_responses = {"points": [], "classes": [], "classIDs": [],  "confidences": []}
    filtered_activations = torch.zeros_like(activations)

    for (vTile, hTile, predicted_class, confidence) in coord_map:
        if eroded_mask[vTile, hTile] == 0:
            continue  # removed by erosion

        x = hTile * step
        y = vTile * step
        activationCoordinateX = int(x + half_tile_size)
        activationCoordinateY = int(y + half_tile_size)

        filtered_responses["points"].append((activationCoordinateX, activationCoordinateY))
        filtered_responses["classes"].append(class_id_to_name[predicted_class])
        filtered_responses["classIDs"].append(predicted_class)
        filtered_responses["confidences"].append(confidence)
        filtered_activations[predicted_class] += 1

    predicted_classes_flat = torch.as_tensor(predictions, dtype=torch.int32)
    bg_mask = (predicted_classes_flat == cleanClassID)
    bg_count = int(bg_mask.sum().item())
    if bg_count > 0:
        bg_prob_sum = float(torch.as_tensor(confidences, dtype=torch.float32)[bg_mask].sum().item())
        filtered_responses["background_avg_prob"] = bg_prob_sum / bg_count
    else:
        filtered_responses["background_avg_prob"] = 0.0

    print(f"{filtered_activations.sum().item()}/{num_preds} activations (after erosion)")
    print("Per-class activations:", filtered_activations.tolist())

    return occupancy.cpu().numpy(), filtered_responses






@torch.no_grad()
def process_predictions(predictions, confidences, class_id_to_name, cleanClassID, rgba_image, tile_size=24, step=2):
    original_image = torch.as_tensor(rgba_image, dtype=torch.uint8)
    height, width, _ = original_image.shape

    tilesH = (height - tile_size) // step
    tilesW = (width - tile_size) // step
    expected_tiles = (tilesH) * (tilesW)

    #if len(predictions) != expected_tiles:
    if not (verifyTileNumber(len(predictions), original_image, tile_size, step)):
        print(f"⚠️ Warning: predictions={len(predictions)} tiles expected={expected_tiles}")

    occupancy = torch.full((tilesH, tilesW), 255, dtype=torch.uint8)
    responses = {"points": [], "classes": [], "classIDs": [],  "confidences": []}
    heatmap = original_image[:, :, :3].clone()
    activations = torch.zeros(len(class_id_to_name), dtype=torch.int32)

    y_indices = torch.arange(0, height - tile_size + 1, step)
    x_indices = torch.arange(0, width - tile_size + 1, step)

    predicted_classes = torch.as_tensor(predictions, dtype=torch.int32)
    totalActivations = 0
    idx = 0
    num_preds = len(predicted_classes)
    half_tile_size = tile_size // 2

    bg_prob_sum = 0.0
    bg_count    = 0

    for vTile, y in enumerate(y_indices):
        for hTile, x in enumerate(x_indices):
            if idx >= num_preds:
                break

            predicted_class = int(predicted_classes[idx])

            # Skip invalid IDs gracefully
            if predicted_class < 0 or predicted_class >= len(class_id_to_name):
                idx += 1
                continue

            if predicted_class != cleanClassID:
                totalActivations += 1
                activationCoordinateX = int(x + half_tile_size)
                activationCoordinateY = int(y + half_tile_size)
                activations[predicted_class] += 1
                confidence = float(confidences[idx])

                responses["points"].append( (activationCoordinateX, activationCoordinateY) )
                responses["classes"].append(class_id_to_name[predicted_class])
                responses["classIDs"].append(int(predicted_class))
                responses["confidences"].append(confidence)

                occupancy[vTile, hTile] = 0
            else:
                bg_prob_sum += float(confidences[idx])
                bg_count    += 1

            idx += 1

        if idx >= num_preds:
            break

    responses["background_avg_prob"] = bg_prob_sum / bg_count if bg_count > 0 else 0.0

    print(f"{totalActivations}/{num_preds} activations")
    print("Per-class activations:", activations.tolist())
    return occupancy.cpu().numpy(), responses

#----------------------------------------------------------
#----------------------------------------------------------
def draw_heatmap(rgba_image, responses, class_id_to_color, size=10):
    """
    Draw crosses onto a heatmap using the responses returned by process_predictions().
    """

    # RGB only
    original_image = torch.as_tensor(rgba_image, dtype=torch.uint8)
    heatmap = original_image[:, :, :3].clone().cpu().numpy()

    for (x, y), class_id in zip(responses["points"], responses["classIDs"]):

        color = class_id_to_color[class_id]

        # points are already in the demosaiced (half-res) space this heatmap
        # lives in — draw_cross takes (Y, X); halving them again squeezed every
        # cross toward the top-left (bug visible once erosion voting was enabled)
        draw_cross(heatmap, (y, x), size, color)

    return heatmap
#----------------------------------------------------------
#----------------------------------------------------------

@torch.no_grad()
def classify_tiles(model, rgba_image, tile_size=64, step=0,
                   chunks=0, majorityVote=True,
                   thresholdMaxProbability=0.50,
                   forceLowMaxProbToThisClass=None,
                   return_torch=False,
                   return_tiles=False):
    """
    Classify tiles efficiently, returning integer class IDs.

    return_torch : if True, return GPU tensors instead of numpy arrays (avoids
                  GPU→CPU copy when the caller immediately wraps back to tensor).
    return_tiles : if True, return the tile tensor (N, C, tile_size, tile_size) as
                  a third return value so the caller can reuse it for stage-2 without
                  re-running unfold.
    """
    start = time.time()

    # Extract tiles and prepare tensor (N, C, H, W)
    npTiles = tile_and_cast_data_torch(rgba_image, tile_size=tile_size, step=step)
    npTiles = npTiles.to(dtype=torch.float32).permute(0, 3, 1, 2).contiguous().to('cuda')

    channels = 4
    if npTiles.shape[1:] != (channels, tile_size, tile_size):
        raise ValueError(f"Expected {channels}x{tile_size}x{tile_size}, got {npTiles.shape[1:]}")

    low_activations = 0

    # Predict in one pass or in chunks
    if chunks == 0:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            preds = model(npTiles)
        probs = torch.nn.functional.softmax(preds.float(), dim=1)
        max_probs, predictions = torch.max(probs, dim=1)
        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            mask = max_probs < thresholdMaxProbability
            low_activations += mask.sum().item()
            predictions[mask] = forceLowMaxProbToThisClass
    else:
        preds_list = []
        for chunk in npTiles.chunk(chunks):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                preds_list.append(model(chunk))
        preds = torch.cat(preds_list)
        probs = torch.nn.functional.softmax(preds.float(), dim=1)
        max_probs, predictions = torch.max(probs, dim=1)
        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            mask = max_probs < thresholdMaxProbability
            low_activations += mask.sum().item()
            predictions[mask] = forceLowMaxProbToThisClass

    print(f"Low-confidence tiles reassigned: {low_activations}")

    # Spatial smoothing (optional) — must happen before optional CPU conversion
    if majorityVote:
        h, w, _ = rgba_image.shape
        tilesHorizontally = (w - tile_size) // step + 1
        tilesVertically   = (h - tile_size) // step + 1
        predictions_np = majority_vote_2d_pytorch(
            predictions.cpu().numpy(), tilesHorizontally, tilesVertically, window_size=3)
        max_probs_np = max_probs.cpu().numpy().flatten()
        print(f"classify_tiles done in {time.time() - start:.2f}s, got {predictions_np.size} tiles")
        if return_tiles:
            return predictions_np.flatten(), max_probs_np, npTiles
        return predictions_np.flatten(), max_probs_np

    print(f"classify_tiles done in {time.time() - start:.2f}s, got {len(predictions)} tiles")

    if return_torch:
        if return_tiles:
            return predictions, max_probs, npTiles
        return predictions, max_probs

    out_preds = predictions.cpu().numpy().flatten()
    out_confs = max_probs.cpu().numpy().flatten()
    if return_tiles:
        return out_preds, out_confs, npTiles
    return out_preds, out_confs



@torch.no_grad()
def runSingle(image,
              model,
              device,
              classes,
              class_colors,
              tile_size,
              step,
              dumpTiles=False,
              majorityVote=True,
              maxProbabilityThreshold=0.50,
              erosion_kernel=0, 
              erosion_threshold=0,
              name="Model",
              log=True,
              draw=True):
    """
    Full pipeline: read image, classify tiles, and generate heatmap.
    Uses integer IDs internally for performance.
    """
    print(f"runSingle: image {image.shape}, tile={tile_size}, step={step}, classes={len(classes)} erosion_kernel={erosion_kernel} erosion_threshold={erosion_threshold}")

    # 1. Read & normalize image
    rgba_image = readPolarPNMToRGBALive(image)
    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
    rgba_image = (rgba_image.astype('float32') / 255.0)
    rgba_image = torch.as_tensor(rgba_image, device=device, dtype=torch.float32)

    # 2. Identify "clean" class once
    cleanClassID = next((i for i, c in enumerate(classes) if c.lower() in ("class_clean", "clean")), None)

    # 3. Perform inference
    start = time.time()

    predictions, confidences = classify_tiles(
                                              model,
                                              rgba_image,
                                              tile_size=tile_size,
                                              step=step,
                                              majorityVote=majorityVote,
                                              thresholdMaxProbability=maxProbabilityThreshold,
                                              forceLowMaxProbToThisClass=cleanClassID,
                                             )
    print(bcolors.OKGREEN)
    elapsed = time.time() - start + 1e-4
    hz = 1.0 / elapsed
    tiles_per_sec = len(predictions) / elapsed
    print("%s / step=%u / inference @ %0.2f Hz  (%d tiles, %.0f tiles/sec)" % (name, step, hz, len(predictions), tiles_per_sec))
    print(bcolors.ENDC)

    if (log):
      log_performance("perf.csv", name, step, tile_size, majorityVote, maxProbabilityThreshold, len(predictions), hz)
    
    # 4. Restore image intensity for heatmap overlay
    rgba_image *= 255.0

    if dumpTiles:
        dump_tiles_as_png(rgba_image, predictions, classes, tile_size, step)

    # 5. Precompute lookup tables
    class_id_to_name  = classes
    class_id_to_color = [torch.tensor(c, dtype=torch.uint8) for c in class_colors]


    if (erosion_kernel==0) or (erosion_threshold==0):
    # 6. Generate heatmap safely
       heatmapRGBImage, occupancy, responses = generate_heatmap(
                                                                predictions,
                                                                confidences,
                                                                class_id_to_name,
                                                                class_id_to_color,
                                                                cleanClassID,
                                                                rgba_image,
                                                                tile_size=tile_size,
                                                                step=step
                                                               )
    else:
    # 6. Process predictions using an erode filter
       occupancy, responses  = process_predictions_erode(
                                                         predictions,
                                                         confidences,
                                                         class_id_to_name,
                                                         cleanClassID,
                                                         rgba_image,
                                                         tile_size=tile_size,
                                                         step=step,
                                                         erosion_kernel=erosion_kernel, 
                                                         erosion_threshold=erosion_threshold
                                                        )

    # 7. Draw heatmap only if needed
       if (draw):
          heatmapRGBImage = draw_heatmap(
                                         rgba_image,
                                         responses,
                                         class_id_to_color
                                        )
       else:
          original_image = torch.as_tensor(rgba_image, dtype=torch.uint8)
          heatmap = original_image[:, :, :3].clone().cpu().numpy()

    return heatmapRGBImage, occupancy, responses

def ensure_file(path):
            import subprocess
            if os.path.exists(path):
                return True

            filename = os.path.basename(path)
            url = f"http://ammar.gr/magician/ckpts/{filename}"

            print(f"File {path} not found. Downloading from {url} ...")

            try:
                # -q quiet, -O output file, --show-progress gives progress bar
                result = subprocess.run(
                    ["wget", "-q", "--show-progress", "-O", path, url],
                    check=True
                )
                print(f"Downloaded {filename} → {path}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Failed downloading {url}")
                print("wget error:", e)
                return False



class ClassifierPnm:
    def __init__(self,
                 model_path='/app/src/python/classifier/resnet18.pth', 
                 cfg_path='/app/src/python/classifier/resnet18.json', 
                 tile_classes=['class_neg', 'class_pos', 'class_clean','class_unknown'],
                 tile_size=64,
                 step=16,
                 precache=True):

        # ------------------------------
        # Ensure model + config exist
        # ------------------------------
        if not ensure_file(model_path):
            print("Failed fetching ",model_path)
            sys.exit(1)

        if not ensure_file(cfg_path):
            print("Failed fetching ",cfg_path)
            sys.exit(1)

        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    print("Opening Classifier configuration file ",cfg_path)
                    self.cfg = json.load(f)
                    self.tile_classes = self.cfg["classes"]
                    self.classes      = self.cfg["classes"]
                    self.tile_size    = self.cfg["hparams"]["tile_size"]
            except Exception as e: 
                print("Failed reading ",cfg_path)
                print("Failed:", repr(e))
                sys.exit(1)
        else:
            print("Classifier configuration file ",cfg_path," does not exist")
            sys.exit(1)
        #--------------------------------------------------------------
        self.step = step
        self.name = os.path.basename(model_path)
        self.model_path = model_path
        self.maxProbabilityThreshold = 0.0
        self.hz = 0.0
        #--------------------------------------------------------------
        print("Classes : ",self.tile_classes)
        print("Tile Size : ",self.tile_size)
        #--------------------------------------------------------------
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        #--------------------------------------------------------------
        self.base_channels = 32
        if  'base_channels' in self.cfg['hparams']:
          self.base_channels = self.cfg['hparams']['base_channels']
        print("Base channels ", self.base_channels)
        #--------------------------------------------------------------
        self.final_dense_layer=512
        if  'final_dense_layer' in self.cfg['hparams']:
                  self.final_dense_layer = self.cfg['hparams']['final_dense_layer']
        print("Final Dense Layer ", self.final_dense_layer)
        #-----------------------------------------------------------------
        self.model = self.load_model()
        if (precache):
           #for i in range(5):
           #    self.test_model(i)
           self.test_model(0)
        self.class_colors = getNDifferentColors(len(self.tile_classes))


    def test_model(self,iteration=0):
        tile_size = self.cfg['hparams']['tile_size']

        # Random image: (batch=1, channels=4, H, W)
        x = torch.rand(1, 4, tile_size, tile_size, device=self.device )

        # Make sure CUDA is synchronized before timing
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            y = self.model(x)

        # Synchronize again for accurate GPU timing
        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        print(f"test_model {self.name} forward pass #{iteration} took {elapsed_ms:.3f} ms")

        if (iteration==0):
         print("Test input shape:", x.shape)
         if isinstance(y, (list, tuple)):
            print("Model output (list):")
            for i, out in enumerate(y):
                print(f"  Output {i} shape:", out.shape)
         elif isinstance(y, dict):
            print("Model output (dict):")
            for k, v in y.items():
                print(f"  {k}: {v.shape}")
         else:
            print("Model output shape:", y.shape)

        return elapsed_ms

    def load_model(self):
        # Derived polarization channels must match training or conv1 shapes differ
        hp = self.cfg.get('hparams', {})
        derived = dict(
                       AoLP=bool(hp.get('AoLP', False)),
                       DoLP=bool(hp.get('DoLP', False)),
                       Unpolarized=bool(hp.get('Unpolarized', hp.get('unpolarized', False))),
                       MaxPolarization=bool(hp.get('MaxPolarization', False)),
                       MinPolarization=bool(hp.get('MinPolarization', False)),
                       RangePolarization=bool(hp.get('RangePolarization', False)),
                       monochrome=bool(hp.get('monochrome', False)),
                      )
        model = Classifier(
                           model=self.cfg['model'],
                           lr=0.1,
                           num_classes=len(self.classes),
                           tile_size=self.cfg['hparams']['tile_size'],
                           base_channels = self.base_channels,
                           final_dense_layer = self.final_dense_layer,
                           **derived
                          )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"Optimizing {self.model_path}")
        model.compile()
        print(f"Loaded {self.model_path}")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        model = model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _is_valid_pth(path):
        """Return True if the file is a readable zip/pickle (PyTorch checkpoint)."""
        try:
            import zipfile
            if zipfile.is_zipfile(path):
                return True
            # Older pickle-based checkpoints start with the magic bytes \x80\x02
            with open(path, 'rb') as f:
                header = f.read(2)
            return header == b'\x80\x02'
        except Exception:
            return False

    @staticmethod
    def model_scan(directoryPath):
        """Return list of base names for matching, valid .pth and .json pairs in directory."""
        if not os.path.isdir(directoryPath):
            print(f"Directory not found: {directoryPath}")
            return []

        files = os.listdir(directoryPath)
        pth_files  = {os.path.splitext(f)[0] for f in files if f.endswith('.pth')}
        json_files = {os.path.splitext(f)[0] for f in files if f.endswith('.json')}
        matches = sorted(pth_files.intersection(json_files))

        valid = []
        for name in matches:
            pth_path = os.path.join(directoryPath, f"{name}.pth")
            if ClassifierPnm._is_valid_pth(pth_path):
                valid.append(name)
            else:
                print(f"[WARN] Skipping corrupted/incomplete checkpoint: {pth_path}")
        return valid
 
    def reload_model(self, directoryPath, name):
        """Unload previous model and reload a new model + config from given name."""
        model_path = os.path.join(directoryPath, f"{name}.pth")
        cfg_path   = os.path.join(directoryPath, f"{name}.json")
        self.name  = os.path.basename(model_path)

        if not (os.path.exists(model_path) and os.path.exists(cfg_path)):
            print(f"Missing model or config for '{name}' in {directoryPath}")
            return False

        try:
            with open(cfg_path, "r") as f:
                self.cfg = json.load(f)
                self.tile_classes = self.cfg["classes"]
                self.classes      = self.cfg["classes"]
                self.tile_size    = self.cfg["hparams"]["tile_size"]
        except Exception as e:
            print("Failed reading config:", repr(e))
            return False
        #--------------------------------------------------------------
        self.base_channels = 32
        if  'base_channels' in self.cfg['hparams']:
          self.base_channels = self.cfg['hparams']['base_channels']
        print("Base channels ", self.base_channels)
        #--------------------------------------------------------------
        self.final_dense_layer=512
        if  'final_dense_layer' in self.cfg['hparams']:
                  self.final_dense_layer = self.cfg['hparams']['final_dense_layer']
        print("Final Dense Layer ", self.final_dense_layer)
        #-----------------------------------------------------------------
        self.model_path = model_path
        print(f"Reloading model '{name}' from {directoryPath} ...")
        try:
            self.model = self.load_model()
        except (RuntimeError, EOFError, Exception) as e:
            print(f"Failed to load model '{name}': {e}")
            print(f"The file '{model_path}' may be corrupted or incomplete.")
            return False
        self.class_colors = getNDifferentColors(len(self.tile_classes))
        print(f"Reload complete: {name}")
        return True
    
    @torch.no_grad()
    def forward(self, image, majorityVote = False, legend=True, erosion_kernel=0, erosion_threshold=0):
        start      = time.time()    
   
        heatmap, occupancy, responses = runSingle(image, 
                                                  self.model,
                                                  self.device,
                                                  self.classes,
                                                  self.class_colors,
                                                  self.tile_size,
                                                  self.step,
                                                  majorityVote=majorityVote,
                                                  maxProbabilityThreshold=self.maxProbabilityThreshold,
                                                  erosion_kernel=erosion_kernel, 
                                                  erosion_threshold=erosion_threshold,
                                                  name=self.name)

        if legend:
            heatmap = self.add_legend(heatmap)

        seconds    = time.time() - start
        self.hz    = 1 / (seconds+0.0001)

        return heatmap, occupancy, responses

    def add_legend(self, heatmap):
        """Overlay a class legend onto the heatmap using OpenCV."""
        print("Overlaying legend")
        overlay = heatmap.copy()
        alpha = 0.7  # transparency

        # Legend dimensions and styling
        padding = 10
        rect_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1

        # Starting position
        x0, y0 = padding, padding

        for i, (cls, color) in enumerate(zip(self.classes, self.class_colors)):
            # Rectangle coordinates
            top_left = (x0, y0 + i * (rect_height + padding))
            bottom_right = (x0 + rect_height, y0 + i * (rect_height + padding) + rect_height)

            # Draw color box
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)

            # Draw text label
            text_pos = (bottom_right[0] + 8, bottom_right[1] - 4)
            cv2.putText(overlay, cls, text_pos, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA )
            text_pos = (bottom_right[0] + 12, bottom_right[1] - 8)
            cv2.putText(overlay, cls, text_pos, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA )
            text_pos = (bottom_right[0] + 10, bottom_right[1] - 6)
            cv2.putText(overlay, cls, text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA )

        # Blend the overlay with the original heatmap
        cv2.addWeighted(overlay, alpha, heatmap, 1 - alpha, 0, heatmap)

        return heatmap


