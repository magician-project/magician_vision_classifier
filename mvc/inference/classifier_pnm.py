#!/usr/bin/python3

""" 
Author : "Nikos Vasilikopoulos, Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 

ClassifierPnm and its inference helpers (tiling, heatmaps, majority voting,
erosion, model scanning). Split out of liveClassifierTorch.py so live/CLI
streaming, the annotator, ROS and model_download share one maintainable core.
liveClassifierTorch.py re-exports everything for backward compatibility.
"""


import cv2
import numpy as np
# -> 
#python3 evaluate.py tile_classifier.keras /home/ammar/Documents/Programming/Magician/src/python/classifier/average100/sample01.pnm
import os
import glob
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import json
import time
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from mvc.core.lit_classifier import Classifier   # its real home; the trainer only re-exports it
#from numba import njit #Test
from mvc.core.read_data import readPolarPNMToRGBALive#,readPolarPNMToRGBAResized
from mvc.paths import repo_root
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
# Hand-picked cross colours, ordered so that consecutive classes land on hues that are
# far apart. Every entry is bright and saturated: no black and no near-black, which the
# old RGB-cube generator produced (it started the grid at (0,0,0) and then emitted dark
# duplicates of the base colours), leaving crosses invisible on dark frames.
DISTINCT_CLASS_COLORS = [
    (255,  40,  40),   # red
    (  0, 160, 255),   # azure
    (255, 220,   0),   # yellow
    (  0, 220,  60),   # green
    (255,   0, 200),   # magenta
    (  0, 235, 235),   # cyan
    (255, 140,   0),   # orange
    (150, 110, 255),   # violet
    (255, 255, 255),   # white
    (170, 255,  40),   # lime
    (255, 130, 180),   # pink
    ( 90, 255, 180),   # mint
    (200,  80, 255),   # purple
    (255, 190, 100),   # apricot
    (120, 180, 255),   # sky
    (220, 255, 255),   # ice
]


def getNDifferentColors(n):
    """
    Generate n visually distinct RGB colors as (R, G, B) tuples.
    Only the standard library is used.

    The first colours come from DISTINCT_CLASS_COLORS; beyond that hues are walked by
    the golden angle so new entries fall in the largest remaining gap. Value/saturation
    stay high on purpose so a cross is always readable over the underlying image.
    """
    if n <= 0:
        return []

    if n <= len(DISTINCT_CLASS_COLORS):
        return list(DISTINCT_CLASS_COLORS[:n])

    import colorsys
    class_colors = list(DISTINCT_CLASS_COLORS)

    extra = 0
    while len(class_colors) < n and extra < 4096:
        hue = (extra * 0.6180339887) % 1.0            # golden-angle hue walk
        sat = 1.0 if (extra % 2 == 0) else 0.55       # alternate vivid / pastel
        val = 1.0 if (extra % 3 != 2) else 0.80       # never dark enough to read as black
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        color = (int(r * 255), int(g * 255), int(b * 255))
        extra += 1
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


# ---------------------------------------------------------------------------
# DEAD CODE (D2) -- COMMENTED OUT 2026-07-28, NOT YET DELETED.
# Unreachable: called only by detect_4x4_rectangle, which is itself dead (D3).
# Also BROKEN -- the inner loop iterates x over range(occupancy.shape[0]) instead
# of shape[1], so on a typical wide occupancy grid (e.g. 5x40) 35 columns are
# never visited.
# Retained verbatim below so the logic is recoverable; delete once you are sure
# nothing external depends on it. See ISSUES.md D2.
# ---------------------------------------------------------------------------
# @torch.no_grad()
# def remove_byflood(occupancy):
#     for y in range(occupancy.shape[0]):
#         for x in range(occupancy.shape[0]):
#               if (occupancy[y,x]>0):
#                   num,occupancy = flood_fill_with_threshold(occupancy, (x,y) , 0, 32)
#     return occupancy

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


# ---------------------------------------------------------------------------
# DEAD CODE (D3) -- COMMENTED OUT 2026-07-28, NOT YET DELETED.
# Unreachable: its only call site is commented out at liveClassifierTorch.py:79.
# Also BROKEN -- np.abs(window - pattern) on uint8 arrays wraps around, so
# |0 - 255| evaluates to 1 instead of 255 and the match score is meaningless.
# Cast to int16 before subtracting if this is ever revived.
# Retained verbatim below so the logic is recoverable; delete once you are sure
# nothing external depends on it. See ISSUES.md D3.
# ---------------------------------------------------------------------------
# @torch.no_grad()
# def detect_4x4_rectangle(heatmap, occupancy, thresholdMin=60.0, thresholdMax=68.00):
#     #occupancy = remove_large_blobs(occupancy,45)
#     occupancy = remove_byflood(occupancy)
#     occupancy = mask_outside_rectangle(occupancy, (10,10,occupancy.shape[1]-20,occupancy.shape[0]-20) )
#     final_occupancy = np.zeros_like(occupancy)  
#
#     scaleX = heatmap.shape[0] / occupancy.shape[0] 
#     scaleY = heatmap.shape[1] / occupancy.shape[1]
#
#     data    = [
#                [0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 255,   255,   255,   255, 0],
#                [0, 0, 0, 255,   255,   255,   255, 0],
#                [0, 0, 0, 255,   255,   255,   255, 0],
#                [0, 0, 0, 255,   255,   255,   255, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0], 
#               ]
#
#
#     # Create a NumPy array with dtype uint8
#     pattern = np.array(data, dtype=np.uint8)
#     pattern_height, pattern_width = pattern.shape
#     pattern_pixels = pattern_width * pattern_height
#
#     dataInv  = [
#                [255, 255, 255, 255, 255, 255],
#                [255, 0,   0,   0,     0, 255],
#                [255, 0,   0,   0,     0, 255],
#                [255, 0,   0,   0,     0, 255],
#                [255, 0,   0,   0,     0, 255],
#                [255, 255, 255, 255, 255, 255],
#               ]
#
#     patternInv = np.array(dataInv, dtype=np.uint8)
#     patternInv_height, patternInv_width = patternInv.shape
#     patternInv_pixels = patternInv_width * patternInv_height
#     windowInv  = patternInv 
#
#     # Scan the image with a sliding window
#     img_height, img_width = occupancy.shape
#
#     detected_coordinates = []
#
#     for y in range(img_height - pattern_height + 1):
#         for x in range(img_width - pattern_width + 1):
#
#             window = occupancy[y:y + pattern_height, x:x + pattern_width]
#             score = np.sum(np.abs(window - pattern)) / pattern_pixels
#             if (thresholdMin <= score) and (score <= thresholdMax):  # Allow some tolerance 
#               final_occupancy[y,x]=123
#               draw_cross(heatmap, (int((y+4)*scaleY), int((x+3)*scaleX)) , 10, (0,255,255) )
#               windowInv[:,:] = occupancy[y+1:y+1 + patternInv_height, x+2:x+2 + patternInv_width]#   window[2:,1:7]
#               windowInv      = 255 - windowInv
#               score2 = np.sum(np.abs(windowInv - patternInv)) / patternInv_pixels
#               detected_coordinates.append((x, y))
#               if score2<80:
#               #if np.array_equal(windowInv, patternInv):
#                 #detected_coordinates.append((x, y))
#                 draw_cross(heatmap, (int((y+4)*scaleY), int((x+3)*scaleX)) , 15, (0,0,255) )
#                 final_occupancy[y,x]=255
#
#     #occupancy[:,:] = final_occupancy[:,:]
#     return detected_coordinates, occupancy



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
    # Convert image to tensor (if it's a NumPy array).
    # Preserve the incoming dtype: uint8 frames must stay uint8 all the way to
    # Classifier.build_input_features(), which is what applies the /255. Casting
    # to float here would skip that and feed the model 0-255 values.
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

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

    VECTORIZED (2026-09-09). The original was a plain Python loop over every grid cell
    (thousands of iterations), calling draw_cross() + a color-scale/clamp/cast per
    ACTIVATED tile -- each of those is a separate small GPU op when `rgba_image` is
    CUDA-resident, so an N-activation frame issued O(N) tiny CUDA kernel launches from
    Python. A live cascade smoke test measured this at ~140-160ms/frame -- 60-70% of
    total per-frame latency, more than either model's forward pass -- see
    knowledge/7-9-report.md §8.6, which is what prompted this rewrite. This version does
    the masking/counting/list-building on CPU (cheap, and predictions/confidences already
    arrive as numpy arrays from every call site) and draws every activated tile's cross in
    exactly TWO batched GPU scatter-writes total, regardless of how many tiles activate.

    Verified BYTE-IDENTICAL (heatmap pixels, occupancy, responses) against the original
    element-by-element implementation, including under a synthetic all-4588-tiles-activated
    stress test -- see the smoke test in knowledge/7-9-report.md §8.6. That stress test is
    what caught a real hazard worth recording: a naive "two batched scatter writes" version
    of this rewrite (draw all horizontal bars in one write, all vertical bars in another)
    passed on realistic sparse activation but produced TORN pixels (channel 0 from one
    tile's colour, channels 1-2 from another's) under dense activation, because CUDA's
    fancy-index assignment has no atomicity guarantee across duplicate (row, col) targets
    within one write. The fix below resolves every pixel's winning tile ON THE CPU first
    (exactly reproducing the original's "later tile in raster order wins" rule, since a
    later tile's draw_cross() call always overwrites an earlier one's at any shared pixel,
    whether the collision is between two horizontal bars, two vertical bars, or one of
    each) and only then issues a SINGLE duplicate-free scatter write -- so no matter how
    dense the activation, there is no racy write left to produce a torn colour.
    """
    original_image = torch.as_tensor(rgba_image, dtype=torch.uint8)
    device = original_image.device
    height, width, _ = original_image.shape

    y_indices = torch.arange(0, height - tile_size + 1, step)
    x_indices = torch.arange(0, width  - tile_size + 1, step)
    tilesH = len(y_indices)
    tilesW = len(x_indices)
    expected_tiles = tilesH * tilesW

    if not (verifyTileNumber(len(predictions), original_image, tile_size, step)):
        print(f"⚠️ generate_heatmap warning: predictions={len(predictions)} tiles expected={expected_tiles}")

    num_classes = len(class_id_to_name)
    num_preds = len(predictions)
    half_tile_size = tile_size // 2

    # All of this stays on CPU: predictions/confidences already arrive as numpy arrays at
    # every call site, and this bookkeeping is tiny (thousands of scalars) -- no benefit to
    # moving it to the GPU, and it keeps index tensors off the device that only the actual
    # pixel writes below need to touch.
    predicted_classes_all = torch.as_tensor(predictions, dtype=torch.int64)
    confidences_all = torch.as_tensor(confidences, dtype=torch.float32)

    # Matches the original's early "idx >= num_preds: break" -- process exactly the
    # min(predictions, grid cells) flat positions, in the SAME raster order (vTile outer,
    # hTile inner) the nested loop visited them in.
    n = min(num_preds, expected_tiles)
    predicted_classes = predicted_classes_all[:n]
    confs = confidences_all[:n]
    y_centers_grid = (y_indices + half_tile_size).unsqueeze(1).expand(tilesH, tilesW).reshape(-1)[:n]
    x_centers_grid = (x_indices + half_tile_size).unsqueeze(0).expand(tilesH, tilesW).reshape(-1)[:n]

    valid_mask = (predicted_classes >= 0) & (predicted_classes < num_classes)
    clean_mask = valid_mask & (predicted_classes == cleanClassID)
    activated_mask = valid_mask & (predicted_classes != cleanClassID)

    bg_count = int(clean_mask.sum())
    bg_prob_sum = float(confs[clean_mask].sum()) if bg_count > 0 else 0.0

    activated_idx = torch.nonzero(activated_mask, as_tuple=True)[0]   # ascending -> raster order
    totalActivations = int(activated_idx.numel())

    heatmap = original_image[:, :, :3].clone()
    occupancy = torch.full((tilesH, tilesW), 255, dtype=torch.uint8)   # CPU, matches original
    activations = torch.zeros(num_classes, dtype=torch.int32)
    responses = {"points": [], "classes": [], "classIDs": [], "confidences": []}

    if totalActivations > 0:
        act_classes = predicted_classes[activated_idx]
        act_confs   = confs[activated_idx]
        act_y = y_centers_grid[activated_idx]
        act_x = x_centers_grid[activated_idx]

        activations += torch.bincount(act_classes, minlength=num_classes).to(activations.dtype)
        occupancy.view(-1)[activated_idx] = 0

        # Per-tile colour, confidence-modulated exactly as the original per-tile scaling.
        colors_table = torch.stack(list(class_id_to_color)).float()   # (C, 3)
        base_colors = colors_table[act_classes]                        # (N, 3)
        mod = 0.60 + 0.40 * act_confs                                  # (N,)
        act_colors = (base_colors * mod.unsqueeze(1)).clamp(0, 255).to(torch.uint8)

        # Every activated tile's cross ("+" shape, half-size 10, matching the original
        # draw_cross call) touches 2K-1 pixels (K = 2*half+1, the centre shared by both
        # bars). Build ALL of them -- horizontal bar then vertical bar, for every tile --
        # as one flat (row, col, tile_rank, colour) list, tile_rank = the tile's position
        # in raster/activation order (0..N-1), matching the original's draw sequence.
        half = 10
        offs = torch.arange(-half, half + 1)
        K = offs.numel()
        ranks = torch.arange(totalActivations)

        h_rows = act_y.unsqueeze(1).expand(-1, K).clamp(0, height - 1).reshape(-1)
        h_cols = (act_x.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, width - 1).reshape(-1)
        v_rows = (act_y.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, height - 1).reshape(-1)
        v_cols = act_x.unsqueeze(1).expand(-1, K).clamp(0, width - 1).reshape(-1)

        all_rows = torch.cat([h_rows, v_rows])
        all_cols = torch.cat([h_cols, v_cols])
        all_ranks = ranks.unsqueeze(1).expand(-1, K).reshape(-1).repeat(2)
        all_colors = act_colors.unsqueeze(1).expand(-1, K, 3).reshape(-1, 3).repeat(2, 1)

        # Resolve every pixel to its single winning (highest-rank) tile BEFORE touching the
        # GPU tensor at all -- sort by (pixel, rank) so each pixel's occurrences are
        # contiguous and rank-ascending, then keep only the last (highest-rank) occurrence
        # per pixel. This reproduces the original's "later tile always overwrites" rule
        # exactly, and leaves no duplicate index for the final scatter write to race on.
        linear = all_rows.to(torch.int64) * width + all_cols.to(torch.int64)
        combined_key = linear * (totalActivations + 1) + all_ranks.to(torch.int64)
        order = torch.argsort(combined_key)
        linear_sorted = linear[order]
        colors_sorted = all_colors[order]
        uniq_linear, counts = torch.unique_consecutive(linear_sorted, return_counts=True)
        last_of_group = torch.cumsum(counts, 0) - 1
        win_colors = colors_sorted[last_of_group]
        win_rows = torch.div(uniq_linear, width, rounding_mode='floor')
        win_cols = uniq_linear % width
        heatmap[win_rows.to(device), win_cols.to(device)] = win_colors.to(device)

        # Python-side lists for `responses` are unavoidably per-activation (a ROS message
        # needs real python ints/strs), but this is now ONE batched .tolist() per field
        # instead of N separate int()/float() conversions.
        xs, ys = act_x.tolist(), act_y.tolist()
        classIDs, confs_list = act_classes.tolist(), act_confs.tolist()
        responses["points"]      = list(zip(xs, ys))
        responses["classes"]     = [class_id_to_name[c] for c in classIDs]
        responses["classIDs"]    = classIDs
        responses["confidences"] = confs_list

    responses["background_avg_prob"] = bg_prob_sum / bg_count if bg_count > 0 else 0.0

    #print(f"{totalActivations}/{num_preds} activations")
    #print("Per-class activations:", activations.tolist())
    return heatmap.cpu().numpy(), occupancy.cpu().numpy(), responses


def _neighbor_counts_torch(occupied2d, kernel):
    """(tilesH, tilesW) int64: count of OCCUPIED cells (including itself) in every cell's
    (2*kernel+1)^2 neighbourhood, via a 2-D integral image -- O(cells), not the
    O(activations * kernel^2) nested Python loop this replaces. Same algorithm as
    analysis/eval/eval_vote_curve.py's neighbor_counts (numpy there; kept in torch here to
    match this module's tensors -- the maths is identical, and both were cross-checked
    against each other before this function replaced the loop it's named after."""
    h, w = occupied2d.shape
    m = occupied2d.to(torch.int64)
    padded = torch.zeros((h + 2 * kernel, w + 2 * kernel), dtype=torch.int64)
    padded[kernel:kernel + h, kernel:kernel + w] = m
    ii = torch.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=torch.int64)
    ii[1:, 1:] = torch.cumsum(torch.cumsum(padded, dim=0), dim=1)
    k = 2 * kernel + 1
    return ii[k:k + h, k:k + w] - ii[0:h, k:k + w] - ii[k:k + h, 0:w] + ii[0:h, 0:w]


@torch.no_grad()
def process_predictions_erode(predictions, confidences, class_id_to_name, cleanClassID, rgba_image, tile_size=48, step=14, erosion_kernel=1, erosion_threshold=1):
    """
    erosion_kernel: radius in tiles for neighbor search
    erosion_threshold: minimum number of neighbors required to keep a prediction

    VECTORIZED (2026-09-09) -- same motivation and pattern as generate_heatmap's rewrite
    (knowledge/7-9-report.md SS8.7), found by profiling this function specifically because
    it is the one the SHIPPED "default" preset actually calls every frame (erosion_kernel=1,
    min_votes=2 are both non-zero there, so runSingle() takes this branch, not
    generate_heatmap's). The original was two Python loops that scale with activation
    count -- one that iterates every activated tile AND checks all (2k+1)^2 neighbours one
    at a time. Measured: 26ms at 108 activations (a real frame), climbing linearly to
    487ms at 4,588 (every tile activated) -- i.e. this was already the dominant per-frame
    cost on any reasonably busy frame, worse than generate_heatmap ever was. The erosion
    decision itself has no order dependence to preserve (every neighbour-count reads a
    fixed occupancy snapshot taken before any counting starts), so this is not a
    "verified equivalent, one accepted ambiguity" situation like generate_heatmap's
    rewrite -- it is byte-for-byte the same computation, done with one integral image
    (_neighbor_counts_torch) instead of one Python iteration per (activated tile, neighbour)
    pair. Verified identical to the original element-by-element implementation, including
    under the same activation-fraction stress test that measured the numbers above.
    """
    original_image   = torch.as_tensor(rgba_image, dtype=torch.uint8)
    height, width, _ = original_image.shape

    y_indices = torch.arange(0, height - tile_size + 1, step)
    x_indices = torch.arange(0, width - tile_size + 1, step)
    tilesH           = len(y_indices)
    tilesW           = len(x_indices)
    expected_tiles   = tilesH * tilesW

    if not (verifyTileNumber(len(predictions), original_image, tile_size, step)):
        print(f"⚠️ Warning: predictions={len(predictions)} tiles expected={expected_tiles}")

    num_classes = len(class_id_to_name)
    num_preds = len(predictions)
    half_tile_size = tile_size // 2

    predicted_classes_all = torch.as_tensor(predictions, dtype=torch.int64)
    confidences_all = torch.as_tensor(confidences, dtype=torch.float32)

    # Matches the original's early "idx >= num_preds: break" -- only the first
    # min(predictions, grid cells) flat positions (raster order) are ever visited.
    n = min(num_preds, expected_tiles)
    predicted_classes = predicted_classes_all[:n]

    valid_mask = (predicted_classes >= 0) & (predicted_classes < num_classes)
    activated_flat = valid_mask & (predicted_classes != cleanClassID)
    totalActivations = int(activated_flat.sum())
    #print(f"{totalActivations}/{num_preds} activations (before erosion)")

    occupancy = torch.full((tilesH, tilesW), 255, dtype=torch.uint8)
    occupied2d = torch.zeros(expected_tiles, dtype=torch.bool)
    occupied2d[:n] = activated_flat
    occupancy.view(-1)[:n][activated_flat] = 0
    occupied2d = occupied2d.view(tilesH, tilesW)

    # ---- EROSION: keep an activated tile iff its (2k+1)^2 neighbourhood (including
    # itself) has >= erosion_threshold activated tiles -- identical rule to the original,
    # computed for every cell at once instead of once per activated tile. ----
    counts = _neighbor_counts_torch(occupied2d, erosion_kernel)
    eroded_mask2d = occupied2d & (counts >= erosion_threshold)

    kept_idx = torch.nonzero(eroded_mask2d.view(-1), as_tuple=True)[0]   # ascending -> raster order, same as the original's coord_map iteration order
    kept_classes = predicted_classes_all[kept_idx]
    kept_confs = confidences_all[kept_idx]
    vTiles = torch.div(kept_idx, tilesW, rounding_mode='floor')
    hTiles = kept_idx % tilesW
    xs = (hTiles * step + half_tile_size).tolist()
    ys = (vTiles * step + half_tile_size).tolist()
    classIDs = kept_classes.tolist()
    confs_list = kept_confs.tolist()

    filtered_responses = {
        "points": list(zip(xs, ys)),
        "classes": [class_id_to_name[c] for c in classIDs],
        "classIDs": classIDs,
        "confidences": confs_list,
    }
    filtered_activations = torch.bincount(kept_classes, minlength=num_classes) \
        if kept_idx.numel() > 0 else torch.zeros(num_classes, dtype=torch.int64)

    # background_avg_prob: over the FULL (untruncated) predictions array, matching the
    # original -- it read predictions/confidences fresh here rather than the grid-truncated
    # view used above.
    bg_mask = (predicted_classes_all == cleanClassID)
    bg_count = int(bg_mask.sum())
    if bg_count > 0:
        bg_prob_sum = float(confidences_all[bg_mask].sum())
        filtered_responses["background_avg_prob"] = bg_prob_sum / bg_count
    else:
        filtered_responses["background_avg_prob"] = 0.0

    #print(f"{int(filtered_activations.sum())}/{num_preds} activations (after erosion)")
    #print("Per-class activations:", filtered_activations.tolist())

    return occupancy.cpu().numpy(), filtered_responses







# ---------------------------------------------------------------------------
# DEAD CODE (D1) -- COMMENTED OUT 2026-07-28, NOT YET DELETED.
# Unreachable: the only mention is a docstring reference in draw_heatmap. It is also
# BROKEN -- occupancy is allocated as (H-tile)//step, missing the +1 that the
# y_indices/x_indices aranges use, so the last row/column raises
# 'IndexError: index 10 is out of bounds for dimension 1 with size 10'.
# The live path uses generate_heatmap or process_predictions_erode instead.
# Retained verbatim below so the logic is recoverable; delete once you are sure
# nothing external depends on it. See ISSUES.md D1.
# ---------------------------------------------------------------------------
# @torch.no_grad()
# def process_predictions(predictions, confidences, class_id_to_name, cleanClassID, rgba_image, tile_size=24, step=2):
#     original_image = torch.as_tensor(rgba_image, dtype=torch.uint8)
#     height, width, _ = original_image.shape
#
#     tilesH = (height - tile_size) // step
#     tilesW = (width - tile_size) // step
#     expected_tiles = (tilesH) * (tilesW)
#
#     #if len(predictions) != expected_tiles:
#     if not (verifyTileNumber(len(predictions), original_image, tile_size, step)):
#         print(f"⚠️ Warning: predictions={len(predictions)} tiles expected={expected_tiles}")
#
#     occupancy = torch.full((tilesH, tilesW), 255, dtype=torch.uint8)
#     responses = {"points": [], "classes": [], "classIDs": [],  "confidences": []}
#     heatmap = original_image[:, :, :3].clone()
#     activations = torch.zeros(len(class_id_to_name), dtype=torch.int32)
#
#     y_indices = torch.arange(0, height - tile_size + 1, step)
#     x_indices = torch.arange(0, width - tile_size + 1, step)
#
#     predicted_classes = torch.as_tensor(predictions, dtype=torch.int32)
#     totalActivations = 0
#     idx = 0
#     num_preds = len(predicted_classes)
#     half_tile_size = tile_size // 2
#
#     bg_prob_sum = 0.0
#     bg_count    = 0
#
#     for vTile, y in enumerate(y_indices):
#         for hTile, x in enumerate(x_indices):
#             if idx >= num_preds:
#                 break
#
#             predicted_class = int(predicted_classes[idx])
#
#             # Skip invalid IDs gracefully
#             if predicted_class < 0 or predicted_class >= len(class_id_to_name):
#                 idx += 1
#                 continue
#
#             if predicted_class != cleanClassID:
#                 totalActivations += 1
#                 activationCoordinateX = int(x + half_tile_size)
#                 activationCoordinateY = int(y + half_tile_size)
#                 activations[predicted_class] += 1
#                 confidence = float(confidences[idx])
#
#                 responses["points"].append( (activationCoordinateX, activationCoordinateY) )
#                 responses["classes"].append(class_id_to_name[predicted_class])
#                 responses["classIDs"].append(int(predicted_class))
#                 responses["confidences"].append(confidence)
#
#                 occupancy[vTile, hTile] = 0
#             else:
#                 bg_prob_sum += float(confidences[idx])
#                 bg_count    += 1
#
#             idx += 1
#
#         if idx >= num_preds:
#             break
#
#     responses["background_avg_prob"] = bg_prob_sum / bg_count if bg_count > 0 else 0.0
#
#     print(f"{totalActivations}/{num_preds} activations")
#     print("Per-class activations:", activations.tolist())
#     return occupancy.cpu().numpy(), responses

#----------------------------------------------------------
#----------------------------------------------------------
def draw_heatmap(rgba_image, responses, class_id_to_color, size=10):
    """
    Draw crosses onto a heatmap using the responses returned by process_predictions().

    VECTORIZED (2026-09-09) -- same pattern as generate_heatmap's and
    process_predictions_erode's rewrites (knowledge/7-9-report.md SS8.7 and the
    profiling that followed it): the original called draw_cross() once per point in a
    Python loop. Measured: ~3.5ms at 60 points, climbing to ~65.8ms at 4,588 (every tile)
    -- linear in point count, the same anti-pattern class, now the largest remaining cost
    on any frame with many surviving (post-erosion) detections.

    Draws every point's cross via ONE deduplicated scatter write: resolve, per pixel,
    which point wins on overlap ("later point in responses["points"] order wins",
    matching the original's sequential draw_cross calls exactly) before ever writing to
    the array -- so there is no duplicate-index write left to race on, the same fix
    that made generate_heatmap's rewrite byte-identical rather than merely "close."
    Verified byte-identical to the original, including under the same point-count
    stress test that measured the numbers above -- WITHIN the domain any real point can
    ever fall in. One out-of-domain difference, found and characterized rather than
    ignored: the original's `image[y, x-10:x+11] = color` is a plain Python slice, so a
    cross centered within 10px of the frame edge gets a NEGATIVE slice start, which
    silently produces an EMPTY slice (draws nothing) rather than wrapping around or
    clamping. This rewrite clamps instead, drawing a partial cross there. The two
    disagree only for a point literally within 10px of x=0/x=width or y=0/y=height --
    which no real point ever is: every point here is a tile centre (tile_size=48,
    half=24), so it is always >=24px from any edge, comfortably outside the 10px band
    where the two implementations would differ.
    """
    original_image = torch.as_tensor(rgba_image, dtype=torch.uint8)
    heatmap = original_image[:, :, :3].clone().cpu().numpy()

    points = responses["points"]
    class_ids = responses["classIDs"]
    n = len(points)
    if n == 0:
        return heatmap

    height, width = heatmap.shape[:2]
    xs = torch.tensor([p[0] for p in points], dtype=torch.int64)
    ys = torch.tensor([p[1] for p in points], dtype=torch.int64)
    cls = torch.tensor(class_ids, dtype=torch.int64)
    colors_table = torch.stack([c.to(torch.uint8) for c in class_id_to_color])   # (C, 3)
    colors = colors_table[cls]                                                   # (N, 3)
    # NOTE: unlike generate_heatmap, draw_heatmap never modulated colour by confidence
    # in the original -- matched here too, not a behaviour change.

    half = size
    offs = torch.arange(-half, half + 1)
    K = offs.numel()
    ranks = torch.arange(n)

    h_rows = ys.unsqueeze(1).expand(-1, K).clamp(0, height - 1).reshape(-1)
    h_cols = (xs.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, width - 1).reshape(-1)
    v_rows = (ys.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, height - 1).reshape(-1)
    v_cols = xs.unsqueeze(1).expand(-1, K).clamp(0, width - 1).reshape(-1)

    all_rows = torch.cat([h_rows, v_rows])
    all_cols = torch.cat([h_cols, v_cols])
    all_ranks = ranks.unsqueeze(1).expand(-1, K).reshape(-1).repeat(2)
    all_colors = colors.unsqueeze(1).expand(-1, K, 3).reshape(-1, 3).repeat(2, 1)

    linear = all_rows * width + all_cols
    combined_key = linear * (n + 1) + all_ranks
    order = torch.argsort(combined_key)
    linear_sorted = linear[order]
    colors_sorted = all_colors[order]
    uniq_linear, counts = torch.unique_consecutive(linear_sorted, return_counts=True)
    last_of_group = torch.cumsum(counts, 0) - 1
    win_colors = colors_sorted[last_of_group].numpy()
    win_rows = (uniq_linear // width).numpy()
    win_cols = (uniq_linear % width).numpy()
    heatmap[win_rows, win_cols] = win_colors

    return heatmap
#----------------------------------------------------------
#----------------------------------------------------------

# ---------------------------------------------------------------------------
# Tile decision gate: how a tile becomes "clean" vs "some defect".
#
# GATE_DEFECT_MASS ("defect_mass") -- default, recommended.
#     score = 1 - P(clean): the total probability mass on *any* defect class.
#     Flags a tile when the model is collectively sure it is a defect, even when
#     it cannot decide *which* defect. This is the detector our binding KPI
#     (skipped defects) actually asks for.
#
# GATE_MAX_PROB ("max_prob") -- legacy, reproduces pre-2026-07-17 behaviour.
#     score = max_c P(c), and the tile must also not argmax to clean.
#     Known bug: a tile at 0.40 Welding / 0.40 Seal / 0.20 clean scores 0.40 and
#     is discarded as clean though it is 80% likely a defect. Welding/Seal
#     confusion is common enough that this drops real defects in bulk. Measured
#     on val_altinay/customwide at a MATCHED 27.0% false-alarm rate:
#     miss 27.1% (max_prob) vs 18.9% (defect_mass), every class better.
#
# GATE_OFF ("off") -- no gate; plain argmax over all classes (clean wins only if
#     it is the argmax). Same as the old threshold<=0 path. FA 37.7 / miss 13.1
#     on val_altinay/customwide, so the gate is not automatically an improvement:
#     max_prob@0.50 is *worse* than no gate at all.
# ---------------------------------------------------------------------------
GATE_DEFECT_MASS = "defect_mass"
GATE_MAX_PROB    = "max_prob"
GATE_OFF         = "off"


# ---------------------------------------------------------------------------
# Deployment presets (recommended_configuration.json)
# ---------------------------------------------------------------------------
# Which model to run and at what operating point, committed to git so a deployment
# site picks up new models and thresholds with a plain `git pull` -- deliberately not
# environment variables, which are awkward to change on-site.
#
# Lives HERE rather than in liveClassifierTorchROS so every consumer shares one
# definition: the ROS node, wxAnnotator (which cannot import rclpy), and anything
# else. The first entry of "configurations" is the default.
# ---------------------------------------------------------------------------
RECOMMENDED_CONFIG_FILE = os.path.join(repo_root(),
                                       "recommended_configuration.json")

# Used only if the file is missing or unreadable, so a bad preferences file can never
# stop a caller from running.
FALLBACK_PRESET = {
    "name": "fallback",
    "model": "mix_convnext_tiny",
    "gate": {"mode": GATE_DEFECT_MASS, "threshold": 0.90, "assign_best_defect_class": True},
    "runtime": {"step": 18, "target_fps": 23.0, "erosion_kernel": 1, "min_votes": 2,
                "majority_voting": True, "frame_limiter": True, "two_stage": False},
}


def load_recommended_configuration(name=None, path=RECOMMENDED_CONFIG_FILE, quiet=False):
    """Return one preset dict from recommended_configuration.json.

    name=None -> the first entry (the committed default); otherwise the entry whose
    "name" matches. Never raises: falls back to FALLBACK_PRESET, because failing to
    read a preferences file must not stop the caller from running.
    """
    try:
        with open(path, "r") as f:
            doc = json.load(f)
        presets = doc.get("configurations") or []
        if not presets:
            raise ValueError("no 'configurations' entries")
        if name is None:
            chosen = presets[0]
        else:
            matches = [p for p in presets if p.get("name") == name]
            if not matches:
                raise ValueError(f"preset {name!r} not found; "
                                 f"available: {[p.get('name') for p in presets]}")
            chosen = matches[0]
        # Merge over the fallback so a partial preset cannot leave a key undefined.
        merged = dict(FALLBACK_PRESET)
        merged.update(chosen)
        merged["gate"] = {**FALLBACK_PRESET["gate"], **(chosen.get("gate") or {})}
        merged["runtime"] = {**FALLBACK_PRESET["runtime"], **(chosen.get("runtime") or {})}
        return merged
    except Exception as e:
        if not quiet:
            print(f"[config] could not use {path} ({e!r}) — falling back to "
                  f"{FALLBACK_PRESET['model']}")
        return dict(FALLBACK_PRESET)


def recommended_configuration_available(path=RECOMMENDED_CONFIG_FILE):
    """True when a usable presets file is present (so callers can say 'if detected')."""
    try:
        with open(path, "r") as f:
            return bool((json.load(f).get("configurations") or []))
    except Exception:
        return False


def gate_tiles(probs, cleanClassID, threshold,
               gateMode=GATE_DEFECT_MASS,
               assignBestDefectClass=True):
    """Turn per-class probabilities into class IDs, forcing weak tiles to clean.

    probs                 : (N, K) softmax probabilities.
    cleanClassID          : index of the clean class. Must genuinely be clean --
                            every mode below reasons about "defect vs clean".
    threshold             : cut on the mode's score. NOT portable across modes or
                            models: defect_mass needs ~0.655 to reproduce the
                            false-alarm rate max_prob gave at 0.50 (customwide);
                            the same FA needs 0.544 for mobilenet_pfc05. Always
                            re-derive per model from the trainer's threshold
                            sweep. threshold <= 0 disables the gate.
    gateMode              : GATE_DEFECT_MASS / GATE_MAX_PROB / GATE_OFF, above.
    assignBestDefectClass : above the gate, label the tile with its best *defect*
                            class instead of the global argmax. Matters because
                            the argmax can still be clean when mass is spread
                            thinly across defects (0.8% of gate-passing tiles);
                            without this those tiles pass the gate and then get
                            labelled clean anyway, which is incoherent. Ignored
                            by GATE_MAX_PROB (which is defined on the argmax).

    Returns (predictions, number_of_tiles_forced_to_clean).
    """
    if gateMode == GATE_OFF or threshold <= 0.0 or cleanClassID is None:
        return probs.argmax(dim=1), 0

    if gateMode == GATE_MAX_PROB:
        max_probs, predictions = torch.max(probs, dim=1)
        mask = max_probs < threshold
        predictions = predictions.clone()
        predictions[mask] = cleanClassID
        return predictions, int(mask.sum().item())

    if gateMode != GATE_DEFECT_MASS:
        raise ValueError(f"unknown gateMode {gateMode!r}; expected one of "
                         f"{GATE_DEFECT_MASS!r}, {GATE_MAX_PROB!r}, {GATE_OFF!r}")

    if assignBestDefectClass:
        defect_probs = probs.clone()
        defect_probs[:, cleanClassID] = -1.0
        predictions = defect_probs.argmax(dim=1)
    else:
        predictions = probs.argmax(dim=1).clone()
    mask = (1.0 - probs[:, cleanClassID]) < threshold
    predictions[mask] = cleanClassID
    return predictions, int(mask.sum().item())


@torch.no_grad()
def classify_tiles(model, rgba_image, tile_size=64, step=0,
                   chunks=0, majorityVote=True,
                   thresholdMaxProbability=0.655,
                   forceLowMaxProbToThisClass=None,
                   gateMode=GATE_DEFECT_MASS,
                   assignBestDefectClass=True,
                   return_torch=False,
                   return_tiles=False,
                   majority_window=3):
    """
    Classify tiles efficiently, returning integer class IDs.

    thresholdMaxProbability : cut on the gate's score — see gate_tiles(). The name
                  is historical; under the default gateMode it thresholds
                  1 - P(clean), NOT the max probability, so 0.50 here is not the
                  0.50 of the old gate (it would raise false alarms 27% -> 50%).
                  Default 0.655 reproduces the old 27.0% false-alarm rate on
                  customwide while cutting miss 27.1% -> 18.9%.
    forceLowMaxProbToThisClass : the clean class ID; None disables the gate.
    gateMode, assignBestDefectClass : see gate_tiles().
    return_torch : if True, return GPU tensors instead of numpy arrays (avoids
                  GPU→CPU copy when the caller immediately wraps back to tensor).
    return_tiles : if True, return the tile tensor (N, C, tile_size, tile_size) as
                  a third return value so the caller can reuse it for stage-2 without
                  re-running unfold.
    majority_window : side of the (odd) square tile window majorityVote takes the mode over.
    """
    start = time.time()

    # Extract tiles and prepare tensor (N, C, H, W)
    npTiles = tile_and_cast_data_torch(rgba_image, tile_size=tile_size, step=step)
    # Keep the tile dtype as-is (uint8 for the live/ensemble paths). Classifier
    # .build_input_features() dequantises uint8 to [0,1] on the GPU; casting to
    # float32 here would bypass that and feed the model 0-255, which collapses
    # predictions towards class_clean.
    # Follow the MODEL's device instead of hardcoding 'cuda' -- ClassifierPnm
    # already resolves cuda/cpu, and a hardcoded .to('cuda') crashed outright on a
    # CPU-only host. fp16 autocast is CUDA-only in practice, so it is enabled only
    # there; on CPU the same code runs in fp32.
    try:
        device = next(model.parameters()).device
    except StopIteration:                       # parameterless module (unexpected)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    npTiles = npTiles.permute(0, 3, 1, 2).contiguous().to(device)
    amp_enabled = (device.type == 'cuda')

    channels = 4
    if npTiles.shape[1:] != (channels, tile_size, tile_size):
        raise ValueError(f"Expected {channels}x{tile_size}x{tile_size}, got {npTiles.shape[1:]}")

    low_activations = 0

    # Predict in one pass or in chunks
    if chunks == 0:
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp_enabled):
            preds = model(npTiles)
        probs = torch.nn.functional.softmax(preds.float(), dim=1)
        # max_probs stays the reported per-tile confidence; the gate decides the class.
        max_probs, predictions = torch.max(probs, dim=1)
        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            predictions, forced = gate_tiles(probs, forceLowMaxProbToThisClass,
                                             thresholdMaxProbability,
                                             gateMode=gateMode,
                                             assignBestDefectClass=assignBestDefectClass)
            low_activations += forced
    else:
        preds_list = []
        for chunk in npTiles.chunk(chunks):
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp_enabled):
                preds_list.append(model(chunk))
        preds = torch.cat(preds_list)
        probs = torch.nn.functional.softmax(preds.float(), dim=1)
        max_probs, predictions = torch.max(probs, dim=1)
        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            predictions, forced = gate_tiles(probs, forceLowMaxProbToThisClass,
                                             thresholdMaxProbability,
                                             gateMode=gateMode,
                                             assignBestDefectClass=assignBestDefectClass)
            low_activations += forced

    #print(f"Low-confidence tiles reassigned: {low_activations}")

    # Spatial smoothing (optional) — must happen before optional CPU conversion
    if majorityVote:
        h, w, _ = rgba_image.shape
        tilesHorizontally = (w - tile_size) // step + 1
        tilesVertically   = (h - tile_size) // step + 1
        predictions_np = majority_vote_2d_pytorch(
            predictions.cpu().numpy(), tilesHorizontally, tilesVertically, window_size=majority_window)
        max_probs_np = max_probs.cpu().numpy().flatten()
        #print(f"classify_tiles done in {time.time() - start:.2f}s, got {predictions_np.size} tiles")
        if return_tiles:
            return predictions_np.flatten(), max_probs_np, npTiles
        return predictions_np.flatten(), max_probs_np

    #print(f"classify_tiles done in {time.time() - start:.2f}s, got {len(predictions)} tiles")

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
              maxProbabilityThreshold=0.655,
              gateMode=GATE_DEFECT_MASS,
              assignBestDefectClass=True,
              erosion_kernel=0,
              erosion_threshold=0,
              name="Model",
              log=True,
              draw=True,
              stats=None,
              majority_window=3):
    """
    Full pipeline: read image, classify tiles, and generate heatmap.
    Uses integer IDs internally for performance.
    """
    #print(f"runSingle: image {image.shape}, tile={tile_size}, step={step}, classes={len(classes)} erosion_kernel={erosion_kernel} erosion_threshold={erosion_threshold}")

    # 1. Read image. Upload as uint8 and let Classifier.build_input_features()
    #    do the /255 on the GPU — same numbers as normalising here, but 4x less
    #    data over PCIe. Matches the EnsembleClassifier path.
    rgba_image = readPolarPNMToRGBALive(image)
    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
    rgba_image = torch.as_tensor(rgba_image, device=device, dtype=torch.uint8)

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
                                              gateMode=gateMode,
                                              assignBestDefectClass=assignBestDefectClass,
                                              majority_window=majority_window,
                                             )
    elapsed = time.time() - start + 1e-4
    hz = 1.0 / elapsed
    tiles_per_sec = len(predictions) / elapsed
    # Hand the numbers to the caller when it asked for them, instead of printing a second
    # line: the live runtime folds these into ONE self-overwriting status line, and a print
    # here would scroll it away every frame. No sink -> keep the standalone behaviour.
    if stats is not None:
        stats.update(name=name, step=step, hz=hz,
                     tiles=len(predictions), tiles_per_sec=tiles_per_sec)
    else:
        print(bcolors.OKGREEN + "%s / step=%u / inference @ %0.2f Hz  (%d tiles, %.0f tiles/sec)" % (name, step, hz, len(predictions), tiles_per_sec) + bcolors.ENDC)

    if (log):
      log_performance("perf.csv", name, step, tile_size, majorityVote, maxProbabilityThreshold, len(predictions), hz)
    
    # 4. rgba_image is already uint8 0-255 (normalisation happens inside the
    #    model), so no intensity restore is needed before the heatmap overlay.

    if dumpTiles:
        dump_tiles_as_png(rgba_image, predictions, classes, tile_size, step)

    # 5. Precompute lookup tables
    class_id_to_name  = classes
    class_id_to_color = [torch.tensor(c, dtype=torch.uint8) for c in class_colors]


    return render_predictions(predictions, confidences, class_id_to_name, class_id_to_color,
                              cleanClassID, rgba_image, tile_size, step,
                              erosion_kernel=erosion_kernel,
                              erosion_threshold=erosion_threshold,
                              draw=draw)


@torch.no_grad()
def render_predictions(predictions, confidences, class_id_to_name, class_id_to_color,
                       cleanClassID, rgba_image, tile_size, step,
                       erosion_kernel=0, erosion_threshold=0, draw=True):
    """
    Turn a full tile grid of predictions into (heatmap, occupancy, responses), applying the
    neighbourhood vote (erosion_kernel/erosion_threshold) when both are non-zero. Shared by
    runSingle and the ensemble/cascade classifiers so min_votes means the same everywhere.
    """
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
          heatmapRGBImage = original_image[:, :, :3].clone().cpu().numpy()

    return heatmapRGBImage, occupancy, responses

def ensure_file(path):
            """Make sure a model's .pth/.json exists locally, fetching it if not.

            Goes through model_download, which reads the SAME server directory the
            uploader writes and the annotator's "Download & Use" list reads
            (http://ammar.gr/magician/models/CameraV2Models/, flat
            {name}_{timestamp}.zip archives holding {name}.pth + {name}.json).

            This used to wget bare files from .../magician/ckpts/ -- a DIFFERENT
            directory that scripts/uploadModels.sh explicitly warns is not the one
            being served, so the fallback silently 404'd and ClassifierPnm exited.
            Because one archive carries both files, fetching for the .pth also
            satisfies the follow-up .json check without a second download.
            """
            if os.path.exists(path):
                return True

            directory = os.path.dirname(os.path.abspath(path)) or "."
            stem = os.path.splitext(os.path.basename(path))[0]
            print(f"File {path} not found. Fetching model '{stem}' from the model server ...")
            try:
                from mvc.inference.model_download import download_model
                download_model(stem, directory)
            except Exception as e:
                print(f"Failed downloading model '{stem}': {e!r}")
                return False
            return os.path.exists(path)



class ClassifierPnm:
    def __init__(self,
                 model_path='/app/src/python/classifier/resnet18.pth', 
                 cfg_path='/app/src/python/classifier/resnet18.json', 
                 tile_classes=['class_neg', 'class_pos', 'class_clean','class_unknown'],
                 tile_size=64,
                 step=16,
                 precache=True,
                 compile_mode=None):

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
        self.compile_mode = compile_mode
        self.name = os.path.basename(model_path)
        self.model_path = model_path
        # Tile decision gate -- see gate_tiles(). Driven by an optional "gate"
        # block in the model's own .json so each model ships the operating point
        # it was calibrated at (thresholds are NOT portable between models):
        #     "gate": {"mode": "defect_mass", "threshold": 0.57,
        #              "assign_best_defect_class": true}
        # wxAnnotator overrides these per-forward from its Classifier tab.
        # Default threshold 0.0 leaves the gate OFF, which is this path's
        # historical behaviour (plain argmax). Do not "enable" it blindly: on
        # val_altinay/customwide, argmax gives FA 37.7 / miss 13.1, and turning
        # a gate on at 0.655 trades that to FA 27.0 / miss 18.9 -- lower false
        # alarms but MORE skipped defects, which is the wrong way on our KPI.
        # The win here is mode, not gating: defect_mass at 0.570 holds FA at
        # 37.7 and cuts miss to 11.9. Re-derive per model from the sweep.
        gate_cfg = self.cfg.get("gate", {}) if isinstance(getattr(self, "cfg", None), dict) else {}
        self.gateMode                = gate_cfg.get("mode", GATE_DEFECT_MASS)
        self.maxProbabilityThreshold = float(gate_cfg.get("threshold", 0.0))
        self.assignBestDefectClass   = bool(gate_cfg.get("assign_best_defect_class", True))
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
        # Logged, not stored. from_config reads both from hparams with these same
        # defaults; keeping them as attributes would be dead state that reads like it
        # configures the model -- which is how the hand-built kwargs drifted in the
        # first place.
        _hp = self.cfg['hparams']
        print(f"Base channels {_hp.get('base_channels', 32)} / "
              f"final dense layer {_hp.get('final_dense_layer', 512)}")
        #-----------------------------------------------------------------
        self.model = self.load_model()
        # Operating curve for this model, so the runtime can report what a gate
        # threshold costs instead of the operator guessing. See _load_threshold_curve.
        self.threshold_curve = self._load_threshold_curve()
        print(self.format_threshold_tradeoff(self.maxProbabilityThreshold))
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
        # Single source of truth (LitClassifier.Classifier.from_config). This block used to
        # enumerate the knobs by hand: the seven derived-channel flags and the CustomCNN
        # ladder, the latter added after allclass_customwide ([128,96,64,64]) was rebuilt at
        # the default width and load_state_dict died on conv1 (128 vs 48).
        #
        # It was still missing two, and both are architecture-changing rather than loud:
        #   timm_stem_stride    -- the stride-2 variants rebuild at stride 4 here
        #   custom_wavelet_stem -- ditto for the wavelet-stem CustomCNNs
        # Neither is hypothetical; the stride-2 arm is a live experiment. That is the whole
        # argument for one translation instead of eleven: this call site was audited,
        # corrected once, commented -- and was still two knobs short.
        #
        # Two deliberate overrides:
        #   pretrained=False  inference always loads a checkpoint over this architecture, so
        #                     fetching ImageNet weights is a network round-trip (which fails
        #                     on an offline deployment box) for weights we discard.
        #   lr=0.1            inert at inference -- no optimizer is ever configured -- but
        #                     kept verbatim so this conversion changes nothing observable.
        # base_channels/final_dense_layer are NOT passed: from_config reads them from
        # hparams with the same 32/512 defaults the old block applied by hand.
        model = Classifier.from_config(self.cfg,
                                       num_classes=len(self.classes),
                                       lr=0.1,
                                       pretrained=False)

        # weights_only=False: torch 2.6+ defaults it to True, which refuses the
        # AttributeDict that save_hyperparameters() stores. See LitClassifier.load_for_eval.
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"Optimizing {self.model_path}")
        # channels_last (NHWC) weights let cuDNN/cuBLAS pick the fp16 tensor-core conv
        # kernels under the autocast in classify_tiles(). Measured on one real polarization
        # frame (5950 tiles of 4x48x48, RTX 4090 laptop, idle GPU), contiguous -> NHWC:
        #   resnext50      260.9 -> 132.4 ms  (-49%)      small_cnn      38.6 -> 30.0 ms (-22%)
        #   resnet18        36.7 ->  29.5 ms  (-20%)      verysmall_cnn  23.1 -> 20.8 ms (-10%)
        #   convnext_tiny  132.0 -> 126.6 / 84.0 -> 84.7 ms  (noise, both directions)
        # End to end that is two-stage 20.2 -> 25.2 Hz (the cheap gate AND every ensemble
        # member get faster), and single-classifier 11.30 -> 11.29 Hz, i.e. unchanged --
        # that path is pure convnext_tiny, the one architecture this does not move. It is a
        # memory-LAYOUT change only -- strides, not values, and not the logical channel
        # order -- so the four polarization planes stay where readPolarPNMToRGBALive put
        # them. Verified through the full debayer path on real shared-memory frames:
        # 285,600 tile decisions (6 shipped architectures x 8 live frames) produced ZERO
        # prediction changes. Confidences move by <=7e-3, which is fp16 accumulation
        # order, not a layout error -- two runs of the UNCHANGED contiguous build already
        # differ by ~6e-4 on the CustomCNNs for the same reason.
        # MUST happen before compile(): converting an already-compiled module trips
        # inductor's stride guards ("expected size 5==5, stride 1==16 at dim=1").
        # run_models_async() converts its input to channels_last to match these weights.
        model = model.to(memory_format=torch.channels_last)
        # compile_mode is opt-in per model because inductor's autotuning is priced per INPUT
        # SHAPE. The single classifier batches the whole tile grid, one constant shape
        # (5950x4x48x48) forever, so autotuning it once buys ~19% on every later frame:
        #   default 85.9 ms (11.64 Hz, 11 s compile) -> 67.7 ms (14.76 Hz, 30 s compile)
        #   on ttcnxtiny_convnext_tiny, verified 0/5950 prediction changes on a real frame.
        # The ensemble members are the opposite case: they run on the non-clean subset, which
        # _bucketed_batch_size() rounds to one of 7 sizes, so they would pay that compile
        # SEVEN times per model for a batch that is already small. Hence caller's choice.
        #
        # "-no-cudagraphs" is not a hedge, it is faster here (67.7 vs 69.5 ms) and compiles in
        # half the time, because at this batch the kernels dwarf any launch overhead CUDA
        # graphs would remove. Plain "max-autotune" ALSO crashes the ensemble outright --
        # run_models_async() keeps several models' outputs alive across calls and the graphs'
        # static output buffers get recycled underneath them:
        #   "RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
        #    by a subsequent run".
        if self.compile_mode:
            model.compile(mode=self.compile_mode)
        else:
            model.compile()
        print(f"Loaded {self.model_path}")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        model = model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Threshold curve: what a gate setting actually costs
    # ------------------------------------------------------------------
    # The trainer writes "<stem>_threshold_curve.json" next to the model with a
    # full defect-vs-clean operating curve (detected / false_alarm sampled every
    # 0.005). Loading it lets the runtime answer "what does threshold 0.90 buy
    # me?" instead of the operator guessing. A high threshold is often the RIGHT
    # deployment choice -- tile false alarms are multiplied by ~thousands of tiles
    # per frame -- but the cost in missed defects should be stated, not implied.
    def _load_threshold_curve(self):
        """Load the trainer's sweep for this model. Returns {} when absent."""
        path = os.path.splitext(self.model_path)[0] + "_threshold_curve.json"
        if not os.path.isfile(path):
            print(f"No threshold curve at {path} — gate trade-offs will not be reported")
            return {}
        try:
            with open(path, "r") as f:
                curve = json.load(f)
            print(f"Loaded threshold curve {path}")
            return curve
        except Exception as e:
            print(f"Failed reading threshold curve {path}: {repr(e)}")
            return {}

    def describe_threshold(self, threshold, gateMode=None):
        """Expected (detected, false_alarm) at `threshold`, linearly interpolated
        from the trainer's sweep. Returns None when no curve is available or the
        curve has no sweep for this gate mode."""
        curve = getattr(self, "threshold_curve", None) or {}
        mode = gateMode or self.gateMode
        sweep = (curve.get("sweeps") or {}).get(mode) or curve.get("sweep")
        if not sweep:
            return None
        pts = sorted(sweep, key=lambda s: s["threshold"])
        t = float(threshold)
        if t <= pts[0]["threshold"]:
            lo = hi = pts[0]
        elif t >= pts[-1]["threshold"]:
            lo = hi = pts[-1]
        else:
            hi = next(p for p in pts if p["threshold"] >= t)
            lo = max((p for p in pts if p["threshold"] <= t), key=lambda p: p["threshold"])
        span = hi["threshold"] - lo["threshold"]
        w = 0.0 if span <= 0 else (t - lo["threshold"]) / span
        lerp = lambda a, b: a + w * (b - a)
        return {
            "threshold":   t,
            "mode":        mode,
            "detected":    lerp(lo["detected"],    hi["detected"]),
            "false_alarm": lerp(lo["false_alarm"], hi["false_alarm"]),
        }

    def format_threshold_tradeoff(self, threshold, gateMode=None):
        """Human-readable summary of `threshold` plus the trainer's named picks,
        for logging whenever the gate is initialised or changed."""
        at = self.describe_threshold(threshold, gateMode)
        if at is None:
            return (f"gate {gateMode or self.gateMode} @ {float(threshold):.3f} "
                    f"(no threshold curve for this model — trade-off unknown)")
        lines = [f"gate {at['mode']} @ {at['threshold']:.3f} -> "
                 f"detects {at['detected']:.1%} of defect tiles, "
                 f"false-alarms on {at['false_alarm']:.2%} of clean tiles"]
        # The trainer's named operating points, for context on where this sits.
        gate_cfg = self.cfg.get("gate", {}) if isinstance(self.cfg, dict) else {}
        named = [("model gate (KPI, misses x2)", gate_cfg.get("threshold"))]
        for label, alt in (gate_cfg.get("alternatives") or {}).items():
            if isinstance(alt, dict):
                named.append((label, alt.get("threshold")))
        for label, t in named:
            if t is None:
                continue
            d = self.describe_threshold(t, gateMode)
            if d:
                mark = "  <-- in use" if abs(float(t) - at["threshold"]) < 1e-6 else ""
                lines.append(f"    {label:32s} {float(t):.3f}  "
                             f"detect {d['detected']:6.1%}  FA {d['false_alarm']:6.2%}{mark}")
        return "\n".join(lines)

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
    def model_locate(directoryPath, name):
        """Path to `{name}.pth` / `{name}.json`, flat directory first. (pth, json) or None.

        Two layouts, and the node has to serve both. A DEPLOYED box gets its models
        unpacked flat into this directory by model_download -- that is the deployment
        layout and it wins. A TRAINING box files each run under
        experiments/<campaign>/<run>/, so a model trained here is not in the flat listing
        at all; before this was added the operator dropdown on the training box went empty
        and ensure_model tried to re-download models that were already on disk.
        """
        flat = (os.path.join(directoryPath, f"{name}.pth"),
                os.path.join(directoryPath, f"{name}.json"))
        if os.path.isfile(flat[0]) and os.path.isfile(flat[1]):
            return flat
        for pth in sorted(glob.glob(os.path.join(directoryPath, 'experiments',
                                                 '*', '*', f'{name}.pth'))):
            cfg = os.path.join(os.path.dirname(pth), f'{name}.json')
            if os.path.isfile(cfg):
                return (pth, cfg)
        return None

    @staticmethod
    def model_scan(directoryPath):
        """Base names of valid .pth/.json pairs -- flat directory plus filed run dirs."""
        if not os.path.isdir(directoryPath):
            print(f"Directory not found: {directoryPath}")
            return []

        pairs = {}
        files = os.listdir(directoryPath)
        pth_files  = {os.path.splitext(f)[0] for f in files if f.endswith('.pth')}
        json_files = {os.path.splitext(f)[0] for f in files if f.endswith('.json')}
        for name in pth_files.intersection(json_files):
            pairs[name] = os.path.join(directoryPath, f"{name}.pth")
        # Locally trained runs, filed beside their configs. setdefault so a flat
        # (deployed) copy always wins over a training-box copy of the same name.
        for pth in sorted(glob.glob(os.path.join(directoryPath, 'experiments',
                                                 '*', '*', '*.pth'))):
            name = os.path.splitext(os.path.basename(pth))[0]
            if os.path.isfile(os.path.join(os.path.dirname(pth), f'{name}.json')):
                pairs.setdefault(name, pth)

        valid = []
        for name in sorted(pairs):
            if ClassifierPnm._is_valid_pth(pairs[name]):
                valid.append(name)
            else:
                print(f"[WARN] Skipping corrupted/incomplete checkpoint: {pairs[name]}")
        return valid
 
    def reload_model(self, directoryPath, name):
        """Unload previous model and reload a new model + config from given name.

        Reads the new config and loads the new model into LOCAL variables first, and
        only commits them onto self.* once load_model() has actually succeeded -- so a
        failed hot-swap leaves self.model/self.cfg/self.classes/self.tile_size exactly
        as they were (previously, self.cfg/self.classes/self.tile_size were overwritten
        with the new model's values before load_model() was even attempted, so a failed
        load left self.model pointing at the OLD model while the rest of self.* already
        described the new one).
        """
        found = ClassifierPnm.model_locate(directoryPath, name)
        if found is None:
            print(f"Missing model or config for '{name}' in {directoryPath} "
                  f"(checked the directory itself and experiments/<campaign>/<run>/)")
            return False
        model_path, cfg_path = found
        new_name = os.path.basename(model_path)

        try:
            with open(cfg_path, "r") as f:
                new_cfg = json.load(f)
                new_tile_classes = new_cfg["classes"]
                new_classes      = new_cfg["classes"]
                new_tile_size    = new_cfg["hparams"]["tile_size"]
        except Exception as e:
            print("Failed reading config:", repr(e))
            return False
        #--------------------------------------------------------------
        _hp = new_cfg['hparams']   # logged only -- see the note in __init__
        print(f"Base channels {_hp.get('base_channels', 32)} / "
              f"final dense layer {_hp.get('final_dense_layer', 512)}")
        #-----------------------------------------------------------------
        print(f"Reloading model '{name}' from {directoryPath} ...")
        # load_model() reads self.model_path/self.cfg/self.tile_size, so stage them on
        # self.* only for the duration of the load attempt; roll back on failure.
        prev = (self.name, self.cfg, self.tile_classes, self.classes, self.tile_size,
                self.model_path)
        self.name, self.cfg, self.tile_classes, self.classes, self.tile_size = (
            new_name, new_cfg, new_tile_classes, new_classes, new_tile_size)
        self.model_path = model_path
        try:
            new_model = self.load_model()
        except (RuntimeError, EOFError, Exception) as e:
            print(f"Failed to load model '{name}': {e}")
            print(f"The file '{model_path}' may be corrupted or incomplete.")
            (self.name, self.cfg, self.tile_classes, self.classes, self.tile_size,
             self.model_path) = prev
            return False
        self.model = new_model
        # Adopt the NEW model's calibrated gate + its curve. Thresholds are not
        # portable between models, so a hot-swap must re-read both.
        gate_cfg = self.cfg.get("gate", {}) if isinstance(self.cfg, dict) else {}
        self.gateMode                = gate_cfg.get("mode", GATE_DEFECT_MASS)
        self.maxProbabilityThreshold = float(gate_cfg.get("threshold", 0.0))
        self.assignBestDefectClass   = bool(gate_cfg.get("assign_best_defect_class", True))
        self.threshold_curve         = self._load_threshold_curve()
        print(self.format_threshold_tradeoff(self.maxProbabilityThreshold))
        self.class_colors = getNDifferentColors(len(self.tile_classes))
        print(f"Reload complete: {name}")
        return True
    
    @torch.no_grad()
    def forward(self, image, majorityVote = False, legend=True, erosion_kernel=0, erosion_threshold=0, log=True,
                stats=None, majority_window=3):
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
                                                  gateMode=self.gateMode,
                                                  assignBestDefectClass=self.assignBestDefectClass,
                                                  erosion_kernel=erosion_kernel,
                                                  erosion_threshold=erosion_threshold,
                                                  name=self.name,
                                                  log=log,   # log = append this frame's timing to perf.csv
                                                  stats=stats,
                                                  majority_window=majority_window)

        if legend:
            heatmap = self.add_legend(heatmap)

        seconds    = time.time() - start
        self.hz    = 1 / (seconds+0.0001)

        return heatmap, occupancy, responses

    def add_legend(self, heatmap):
        """Overlay a class legend onto the heatmap using OpenCV."""
        #print("Overlaying legend")
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


