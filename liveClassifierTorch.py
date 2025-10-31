#!/usr/bin/python3

""" 
Author : "Nikos Vasilikopoulos, Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
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
from trainClassifierTorch import Classifier
from numba import njit #Test
from readData import readPolarPNMToRGBALive#,readPolarPNMToRGBAResized
from SharedMemoryManager import SharedMemoryManager
from torch.nn import functional as F

def dumpListAsCSV(theList,fields,theFilename):
   import csv
   with open(theFilename, 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(theList)


def load_classes_json(filename):
    with open(filename, 'r') as f:
        classes = json.load(f)
    return classes


def checkIfPathIsDirectory(filename):
    return os.path.isdir(filename) 


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
def generate_heatmap(predictions, classes, class_colors, rgba_image, tile_size=24, step=2): # , include_tiles=False
    original_image = torch.tensor(rgba_image, dtype=torch.uint8)
    height, width, _ = original_image.shape
    
    tilesH = (height - tile_size)  // step
    tilesW = (width  - tile_size)  // step

    occupancy = torch.full((tilesH + 1, tilesW + 1), 255, dtype=torch.uint8)

    responses = dict()
    responses["points"]  = list()
    responses["classes"] = list()

    #Heatmap is an RGB image!
    heatmap = original_image[:, :, :3].clone()
    
    cleanClassID = 0
    for i in range(len(classes)):
           if (classes[i] == "class_clean") or (classes[i] == "Clean"):
              cleanClassID = i

    activations = [0] * len(classes)

    print("generate_heatmap tile_size =", tile_size)
    print("step =", step)
    print("expected classes =", len(classes))
    print("clean class =", cleanClassID)
    print("classes colors =", len(class_colors))
    print("predictions =", predictions.shape)
    
    half_tile_size = tile_size // 2
    
    # Create grid indices for tiles
    y_indices = torch.arange(0, height - tile_size, step)
    x_indices = torch.arange(0, width  - tile_size, step)
    
    allTiles = tilesH * tilesW
    totalActivations = 0
    
    #predicted_classes = torch.tensor(predictions, dtype=torch.int64) #nikos code
    predicted_classes = torch.tensor(predictions, dtype=torch.int32)
    
    idx = 0
    for vTile, y in enumerate(y_indices):
        for hTile, x in enumerate(x_indices):
            try:
                #if idx >= len(predicted_classes):
                #   print("idx out of bounds=", idx, "/",len(predicted_classes) )
                #   break  # or continue safely
                predicted_class = int(predicted_classes[idx].item())
                if predicted_class >= len(classes): 
                    print("Predicted class for tile ",idx," is ",predicted_class," but we only have ",len(classes)," classes")
                    raise ValueError("Predicted class for tile ",idx," is ",predicted_class," but we only have ",len(classes)," classes")
                    #continue
                if (predicted_class != cleanClassID):
                    totalActivations += 1
                    color = torch.tensor(class_colors[predicted_class], dtype=torch.uint8)
                    
                    # This is the activation point
                    activationCoordinateX = int(x + half_tile_size)
                    activationCoordinateY = int(y + half_tile_size)

                    # Apply color to heatmap
                    draw_cross(heatmap, (activationCoordinateY, activationCoordinateX), 10, color)

                    # Keep Stats
                    activations[predicted_class]+=1
                    responses["points"].append((activationCoordinateX * 2,activationCoordinateY * 2)) #*2 to restore unpolarized dimensions
                    responses["classes"].append(classes[int(predicted_class)])
                    occupancy[vTile, hTile] = 0 #?
            except Exception as e:
                print("Could not access", x, y)
                print("Failed:", repr(e))
            finally:
                idx += 1
    
    print(f"{totalActivations}/{allTiles} activations")
    print("classes =", classes)
    print("Results : ",activations)
    return heatmap.cpu().numpy(), occupancy.cpu().numpy(), responses


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
def classify_tiles(model, rgba_image, tile_size=64, step=0, chunks=0, tryToSkip = False, majorityVote=True, thresholdMaxProbability=0.50, forceLowMaxProbToThisClass=None):
    start      = time.time()
    #dtype      = torch.float16
    dtype      = torch.float32
    #------------------------------------------------------------------------
    # Extract tiles and cast data in one step
    npTiles = tile_and_cast_data_torch(rgba_image, tile_size=tile_size, step=step).to(dtype).permute(0,3,1,2).contiguous().to('cuda')
    #npTiles = tile_and_cast_data_cpu(rgba_image, tile_size=tile_size, step=step).to(dtype).permute(0,3,1,2).contiguous().to('cuda')
    #npTiles = tileAndCastData(rgba_image, tile_size=tile_size, step=step).float().to('cuda')
    #npTiles = torch.tensor(npTiles).float().permute(0,3,1,2).to('cuda')
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    #print("Before NN @ %0.02f Hz"%hz)   

    #print("classify_tiles -> Will need to evaluate ",npTiles.shape," tiles")
    start      = time.time()    
    #-------------------------------------- ----------------------------------
        
    channels = 4 #4 Polarizations  
    if npTiles.shape[1:] != (channels, tile_size, tile_size):  # Sanity check on desired input size
          raise ValueError(f"classify_tiles input size must be {channels}x{tile_size}x{tile_size}, got {npTiles.shape[1:]}")

    low_activations = 0
    softmax = torch.nn.Softmax(dim=1)
    # Predict the class probabilities for each tile
    #predictions = model.predict(npTiles, batch_size=1024)
    #import ipdb; ipdb.set_trace()
    #Use batch size of 1024
    #num_classes = 4
    if (chunks==0):
        with torch.no_grad(),torch.amp.autocast(device_type='cuda', dtype=dtype):
            preds = model(npTiles)
            #print("preds ",preds)
        probs = softmax(preds)
        #print("probs ",probs)
        max_probs, predictions = torch.max(probs, dim=1)
        #print("Max probs ",max_probs)
        if (forceLowMaxProbToThisClass is not None):
            predictions[max_probs < thresholdMaxProbability] = forceLowMaxProbToThisClass
            low_activations += 1
    
    else:
        chunksB = npTiles.chunk(chunks)
        predictionsList = list()
        for chunk in chunksB:
            #import ipdb; ipdb.set_trace()
            predictionsList.append( model(chunk) )

        preds = torch.cat(predictionsList)
        #import ipdb; ipdb.set_trace()
        probs = softmax(preds)
        max_probs, predictions = torch.max(probs, dim=1)
        #print("Max probs ",max_probs)
        if (forceLowMaxProbToThisClass is not None):
            predictions[max_probs < thresholdMaxProbability] = forceLowMaxProbToThisClass
            low_activations += 1

    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    #print("Actual NN @ %0.02f Hz"%hz)   

    predictions = predictions.cpu().numpy()
    
    if (majorityVote):
      original_image = torch.tensor(rgba_image, dtype=torch.uint8) #<- TODO improve
      height, width, _ = original_image.shape                      #<- TODO We can get shape without cast
      y_indices = torch.arange(0, height - tile_size, step)        #<- TODO We shouldn't get a list of indices just to do this calculation
      x_indices = torch.arange(0, width  - tile_size, step)        #<- TODO We shouldn't get a list of indices just to do this calculation
      tilesHorizontally = len(x_indices)
      tilesVertically   = len(y_indices)
      predictions = majority_vote_2d_pytorch(predictions, tilesHorizontally, tilesVertically, window_size=3)
    #------------------------------------------------------------------------


    predictions = predictions.flatten()
    #------------------------------------------------------------------------
    
    #print("Predictions ",predictions.shape, " Low Activations (",thresholdMaxProbability,") on ",low_activations," tiles")   
    return predictions #,num_classes





@torch.no_grad()
def runSingle(image,model,device,classes,class_colors,tile_size,step, dumpTiles=False, majorityVote=True, maxProbabilityThreshold=0.50):
    print("runSingle image : ",image.shape, " tile_size : ",tile_size,"x",tile_size, " classes : ",classes)
    rgba_image = readPolarPNMToRGBALive(image) #readPolarPNMToRGBAResized(image_path,resize=0.5)

    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA) #Undo opencv crazyness.. <---------------FLIP RGBA to BGRA
    #rgba_image = rgba_image/255.0 #old code
    rgba_image = (rgba_image.astype('float32') / 255.0)
    rgba_image = torch.tensor(rgba_image).float().to(device, dtype=torch.float32)

    #Dataset mean:  [[[[ 89.66365  99.0593  103.32124  94.00895]]]]
    #Dataset std:  [[[[66.99435  72.73822  74.47066  71.135704]]]]
    #data_mean = np.array([ 89.66365,  99.0593,  103.32124,  94.00895])
    #data_std  = np.array([ 66.99435,  72.73822,  74.47066,  71.135704])
    #rgba_image = (rgba_image - data_mean) / data_std

    classIDForCleanTiles = None
    if (maxProbabilityThreshold>0.0):
       try:
         classIDForCleanTiles = classes.index('class_clean')
       except ValueError:
         classIDForCleanTiles = None


    # Perform inference on the image tiles
    predictions = classify_tiles(model, rgba_image, tile_size=tile_size, step=step, majorityVote=majorityVote,  thresholdMaxProbability=maxProbabilityThreshold, forceLowMaxProbToThisClass=classIDForCleanTiles)
    rgba_image  = rgba_image * 255.0

    # Dump all tiles as PNG files
    if dumpTiles:
        dump_tiles_as_png(rgba_image, predictions, classes, tile_size, step)
    

    #minimums,maximums,stds,means = generate_predictionStatistics(predictions,num_classes)
    #print("Minimums ",minimums)
    #print("Maximums ",maximums)
    #print("STDs ",stds)
    #print("Means ",means)

    #predictions = predictions.tolist()
    #dumpListAsCSV(predictions,classes,"predictions.csv")
    #list from numpy array

    # Generate heatmap image
    #start      = time.time()    
    #------------------------------------------------------------------------
    # Extract tiles and cast data in one step
    heatmapRGBImage, occupancy, responses  = generate_heatmap(predictions, classes, class_colors, rgba_image, tile_size=tile_size, step=step)
    #npTiles = torch.tensor(npTiles).float().permute(0,3,1,2).to('cuda')
    #seconds    = time.time() - start
    #hz    = 1 / (seconds+0.0001)
    #print("Generate heatmap %0.02f Hz"%hz)   

    return heatmapRGBImage, occupancy, responses


class ClassifierPnm:
    def __init__(self,
                 model_path='/app/src/python/classifier/resnet18.pth', 
                 cfg_path='/app/src/python/classifier/resnet18.json', 
                 tile_classes=['class_neg', 'class_pos', 'class_clean','class_unknown'],
                 tile_size=64,
                 step=16):

        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    self.cfg = json.load(f)
                    self.tile_classes = self.cfg["classes"]
                    self.classes      = self.cfg["classes"]
                    self.tile_size    = self.cfg["hparams"]["tile_size"]
            except Exception as e: 
                print("Failed reading ",cfg_path)
                print("Failed:", repr(e))
                os.exit(1)

        self.step = step
        self.model_path = model_path

        print("Classes : ",self.tile_classes)
        print("Tile Size : ",self.tile_size)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = self.load_model()

        self.class_colors = getNDifferentColors(len(self.tile_classes))


    def load_model(self):
        model = Classifier(
            model=self.cfg['model'],
            lr=0.1,
            num_classes=len(self.classes),
            tile_size=self.cfg['hparams']['tile_size']
        )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"Loaded {self.model_path}")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        model = model.to(self.device)
        model.eval()
        return model

    def load_model_old(self):
        # Load the trained model
        #model = Classifier.load_from_checkpoint(self.model_path)

        model = Classifier( 
                              model=self.cfg['model'], 
                              lr=0.1,
                              num_classes     = len(self.classes),
                              tile_size       = self.cfg['hparams']['tile_size'],
                              load_checkpoint = self.model_path
                           )

        #model.model = Classifier.load_from_checkpoint(self.model_path)
        model=model.to(self.device)
        model=model.eval()
        #model = model.half()
        return model
    
    @torch.no_grad()
    def forward(self, image, majorityVote = False, legend=True):
        heatmap, occupancy, responses = runSingle(image, self.model, self.device, self.classes, self.class_colors, self.tile_size, self.step, majorityVote=majorityVote)

        if legend:
            heatmap = self.add_legend(heatmap)

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
            text_pos = (bottom_right[0] + 10, bottom_right[1] - 6)
            cv2.putText(
                overlay, cls, text_pos, font,
                font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )

        # Blend the overlay with the original heatmap
        cv2.addWeighted(overlay, alpha, heatmap, 1 - alpha, 0, heatmap)

        return heatmap


if __name__ == "__main__":
    #model_path    = sys.argv[1]
    #model_classes = sys.argv[2]
    #image_path    = sys.argv[3]

    # Load the trained model
    #import ipdb; ipdb.set_trace()
    #from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy, CategoricalFocalCrossentropy
    #total_loss = CategoricalCrossentropy()
    #model  = keras.models.load_model('tile_classifier.keras', custom_objects={'Classifier': Classifier})
    #model = Classifier.load_from_checkpoint(model_path)
    modelC = ClassifierPnm(model_path='last.pth',cfg_path='last.json')
    model=modelC.model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model=model.eval()
    #tiles = torch.rand(2000, 4, 64, 64, dtype=torch.float16,device='cuda')
    #model.half()
    #Compile the model for inference
    model = torch.compile(model)
    #model = torch.jit.trace(model, tiles)
    #model=model.half()
    #Set the model to inference mode
    torch.set_float32_matmul_precision('medium') 

    step=16
    tile_size=64
    
    streamName = "stream1"
    smm = SharedMemoryManager("./libSharedMemoryVideoBuffers.so", 
                              descriptor = "video_frames.shm", 
                              frameName  = streamName,
                              connect    = True)

    # Loop to continuously read frames 

    while True:
        # Capture frame-by-frame
        frame = smm.read_from_shared_memory()
        
        # Check if the frame is captured successfully
        if  (frame is None) or (smm.frame_size==0):
            print("Error: Couldn't read frame from SHM")
        else:        
           #print("frame:",frame)

           # Display the frame in a window
           print("Frame RAW: ",frame.shape)
           if (frame.shape[2]==4):
              frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
              frame = np.transpose(frame, axes=None)
           elif (frame.shape[2]==3):
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           print("Frame Processed : ",frame.shape)
           with torch.inference_mode():
            #heatmap,occupancy = runSingle(frame,model,device,classes,class_colors,tile_size,step)
            heatmap, occupancy, responses = modelC.forward(frame, majorityVote=True)
            #hits, occupancy = detect_4x4_rectangle(heatmap,occupancy)
            cv2.imshow('Live Heatmap', heatmap)
        
           # Break the loop if 'q' is pressed
           if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all OpenCV windows
    cv2.destroyAllWindows()
