#!/usr/bin/python3

# Required python packages :
# python3 -m pip install pytorch_lightning torchvision opencv_python numba

# To run :
# python3 liveClassifierTorch.py tile_classifier.keras /home/ammar/Documents/Programming/Magician/src/python/classifier/average100/sample01.pnm

import os
import sys
import json
import time
import cv2
import numpy as np
#import keras
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from trainClassifierTorch import Classifier
from numba import njit #Test
from readData import readPolarPNMToRGBA
from dumpKerasDataset import check_threshold,check_variation,random_decision ,isTileWorthAnNN
#from trainClassifierKeras import Classifier,build_resnet_4channel,build_efficientnetv2b3_4channel,build_efficientnetv2s_4channel
#from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy, CategoricalFocalCrossentropy
#from tensorflow.keras.optimizers import AdamW, Adam
def dumpListAsCSV(theList,fields,theFilename):
   import csv
   with open(theFilename, 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(theList)

def generate_colors(num_colors):
    if num_colors <= 0:
        return []

    # Initialize a list to store generated colors
    class_colors = []

    # Generate colors with distinct RGB components
    for i in range(num_colors):
        # Use different combinations of RGB components to ensure uniqueness
        red   = (i * 155) % 256
        green = (i * 233) % 256
        blue  = (i * 73) % 256
        class_colors.append((red, green, blue))

    return class_colors


def draw_cross(image, center, half_size, color):
    y, x = center
    # Draw horizontal line of the cross
    image[y, x - half_size:x + half_size + 1] = color
    # Draw vertical line of the cross
    image[y - half_size:y + half_size + 1, x] = color
    return image


def printLabels(heatmap,classes,class_colors):
    x = 30
    y = 30 
    for classID in range(len(classes)):
       cv2.putText(heatmap, classes[classID], (x-2,y-2) , cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0) , 5)
       cv2.putText(heatmap, classes[classID], (x-1,y-1) , cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255) , 5)
       cv2.putText(heatmap, classes[classID], (x,y) , cv2.FONT_HERSHEY_SIMPLEX, 1.5, class_colors[classID], 5)
       y += 40
 
def selectClass(prediction):
    minValue = 0.001
    maxValue = -1.0
    selected = None 
    for i in range(len(prediction)):
       if (minValue<prediction[i]):
         if (maxValue<prediction[i]):
            maxValue = prediction[i]
            selected = i
    return selected


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



def generate_heatmap(predictions, classes, class_colors, rgba_image, tile_size=24, step=2):
    original_image = torch.tensor(rgba_image, dtype=torch.uint8)
    height, width, _ = original_image.shape
    
    tilesH = (height - tile_size) // step
    tilesW = (width - tile_size) // step
    occupancy = torch.full((tilesH + 1, tilesW + 1), 255, dtype=torch.uint8)
    heatmap = original_image[:, :, :3].clone()
    
    num_classes = predictions.max().item() + 2
    print("num_classes =", num_classes)
    
    half_tile_size = tile_size // 2
    
    # Create grid indices for tiles
    y_indices = torch.arange(0, height - tile_size, step)
    x_indices = torch.arange(0, width - tile_size, step)
    
    allTiles = tilesH * tilesW
    activations = 0
    
    predicted_classes = torch.tensor(predictions, dtype=torch.int64)
    
    idx = 0
    for vTile, y in enumerate(y_indices):
        for hTile, x in enumerate(x_indices):
            try:
                predicted_class = predicted_classes[idx].item()
                if classes[predicted_class] != "class_clean":
                    activations += 1
                    color = torch.tensor(class_colors[predicted_class], dtype=torch.uint8)
                    
                    # Apply color to heatmap
                    heatmap[y + half_tile_size, x + half_tile_size, :] = color
                    heatmap[y + half_tile_size, x + half_tile_size + 1, :] = color
                    heatmap[y + half_tile_size + 1, x + half_tile_size, :] = color
                    heatmap[y + half_tile_size + 1, x + half_tile_size + 1, :] = color
                    draw_cross(heatmap, (y + half_tile_size, x + half_tile_size), 10, color)
                    occupancy[vTile, hTile] = 0
                    
                idx += 1
            except Exception as e:
                print("Could not access", x, y)
                print("Failed:", repr(e))
    
    print(f"{activations}/{allTiles} activations")
    return heatmap.cpu().numpy(), occupancy.cpu().numpy()

def castData(data,tile_size):
  numberOfSamples = len(data)
  npInput = np.full([numberOfSamples,tile_size,tile_size,4],fill_value=0,dtype=np.float32,order='C') #np.int8
  for i in range(numberOfSamples):
      npInput[i,:,:,:] = data[i][:,:,:]
  #npInput = npInput / 255.0 #<- This is not needed
  return npInput


@njit
def tileImagesSolo(image,tile_size=24,step=2):
    tiles=[]

    # Get the dimensions of the image
    height, width, channels = image.shape

    # Loop through the image and extract tiles
    for y in range(0, height-tile_size, step):
        start_y = y
        end_y   = y + tile_size
        for x in range(0, width-tile_size, step):
          start_x = x
          end_x   = x + tile_size
            
          tile = image[start_y:end_y,start_x:end_x,:] #<- Why is this the correct order ?

          if (tile.shape[0]==tile_size) and (tile.shape[1]==tile_size):  
             # Append the tile and info to the list
             tiles.append(tile)

    return tiles





def classify_tiles_old(model, rgba_image, tile_size=24,step=2, tryToSkip = False): 
    start      = time.time()    
    #------------------------------------------------------------------------
    # Extract tiles from the image
    tiles = tileImagesSolo(rgba_image,tile_size=tile_size,step=step)
    tileSkips = [0] * len(tiles)

    if (tryToSkip):
     from dumpKerasDataset import isTileWorthAnNN
     totalSkips = 0
     for i,tile in enumerate(tiles):
        if (isTileWorthAnNN(tile)):
            tileSkips[i] = 1
            totalSkips = totalSkips + 1
     print(" ",(100 * tileSkips[i])/len(tiles)," %% tiles can be skipped")

    
    #Potentially dump the tiles to disk to visually inspect them
    #for i,tile in enumerate(tiles):
    #     cv2.imwrite("t%u.jpg"% i, tile)
     
    #from trainClassifier import castData
    npTiles = castData(tiles,tile_size)
    #------------------------------------------------------------------------
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    print("Before NN @ %0.02f Hz"%hz)   


    start      = time.time()    
    #------------------------------------------------------------------------
    print("Will need to evaluate ",npTiles.shape," tiles")
    # Predict the class probabilities for each tile
    predictions = model.predict(npTiles)
    #print(predictions)
    #------------------------------------------------------------------------
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    print("Actual NN @ %0.02f Hz"%hz)   
    
    return predictions








def tileAndCastData(image, tile_size=24, step=2):
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the number of tiles in each dimension
    num_tiles_y = (height - tile_size) // step + 1
    num_tiles_x = (width - tile_size) // step + 1

    # Initialize the numpy array for tiles
    npTiles = np.full((num_tiles_y * num_tiles_x, tile_size, tile_size, channels), fill_value=0, dtype=np.float32)

    index = 0
    for y in range(0, height - tile_size, step):
        start_y = y
        end_y = y + tile_size
        for x in range(0, width - tile_size, step):
            start_x = x
            end_x = x + tile_size
            
            tile = image[start_y:end_y, start_x:end_x, :]
            npTiles[index, :, :, :] = tile
            index += 1

    return npTiles

def tile_and_cast_data_torch(image, tile_size=24, step=2):
    # Convert image to tensor (if it's a NumPy array)
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

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
def majority_vote_2d_manual(predictions_2d, window_size=3):
    """
    Manual implementation of 2D majority voting.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    
    pad_size = window_size // 2
    padded_predictions_2d = np.pad(predictions_2d, pad_size, mode='maximum')
    smoothed_predictions_2d = np.zeros_like(predictions_2d)
    
    for i in range(pad_size, padded_predictions_2d.shape[0] - pad_size):
        for j in range(pad_size, padded_predictions_2d.shape[1] - pad_size):
            window = padded_predictions_2d[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            unique, counts = np.unique(window, return_counts=True)
            smoothed_predictions_2d[i-pad_size, j-pad_size] = unique[np.argmax(counts)]
    
    return smoothed_predictions_2d
def majority_vote_2d_pytorch(predictions_2d, window_size=3):
    """
    Efficient 2D majority voting using PyTorch unfold.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    # Convert to tensor if not already
    if not isinstance(predictions_2d, torch.Tensor):
        predictions_2d = torch.tensor(predictions_2d, dtype=torch.int64)

    pad_size = window_size // 2
    
    # Pad the input (zero padding, but can change to other modes if needed)
    padded = F.pad(predictions_2d, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=-2)
    
    # Extract sliding windows using unfold
    unfolded = padded.unfold(0, window_size, 1).unfold(1, window_size, 1)  # Shape: (H, W, window_size, window_size)
    
    # Reshape to (H, W, window_size * window_size) for mode computation
    unfolded = unfolded.contiguous().view(*unfolded.shape[:2], -1)
    
    # Compute majority vote (mode)
    mode_values, _ = torch.mode(unfolded, dim=2)  # Take mode along window elements
    
    return mode_values
   
softmax = torch.nn.Softmax(dim=1)
def classify_tiles(model, rgba_image, tile_size=64,step=2, tryToSkip = False):
    start      = time.time()    
    #------------------------------------------------------------------------
    # Extract tiles and cast data in one step
    npTiles = tile_and_cast_data_torch(rgba_image, tile_size=tile_size, step=step).float().permute(0,3,1,2)
    #npTiles = torch.tensor(npTiles).float().permute(0,3,1,2).to('cuda')
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    print("Before NN @ %0.02f Hz"%hz)   

    print("Will need to evaluate ",npTiles.shape," tiles")
    start      = time.time()    
    #-------------------------------------- ----------------------------------
        
    # Predict the class probabilities for each tile
    #predictions = model.predict(npTiles, batch_size=1024)
    #import ipdb; ipdb.set_trace()
    #Use batch size of 1024
    #import ipdb; ipdb.set_trace()
    preds = model(npTiles)
    probs = softmax(preds)
    max_probs, predictions = torch.max(probs, dim=1)
    #print("Max probs ",max_probs)
    predictions[max_probs < 0.4] = 3
    
    #predictions = model(npTiles).argmax(dim=1)
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    print("Actual NN @ %0.02f Hz"%hz)   
    num_classes = 4#predictions.shape[1]
    #predictions = predictions.cpu().numpy()
    
    
    #predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    predictions = predictions.reshape((61, -1))
    #gaussian smoothing
    
    #import ipdb; ipdb.set_trace()
    #predictions = majority_vote_2d_manual(predictions, window_size=5)
    predictions = majority_vote_2d_pytorch(predictions, window_size=5)
    #predictions = weighted_majority_vote(predictions, min_size=5)
    #predictions = smooth_predictions(predictions)
    predictions = predictions.flatten()
    #------------------------------------------------------------------------
    
    return predictions,num_classes


def remove_large_blobs(binary_image, max_size):
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an output binary image
    output_image = np.zeros_like(binary_image)  
    
    for i in range(1, num_labels):  # Skip the background label 0
        if stats[i, cv2.CC_STAT_AREA] <= max_size:
            output_image[labels == i] = 255
    
    return output_image


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


def remove_byflood(occupancy):
    for y in range(occupancy.shape[0]):
        for x in range(occupancy.shape[0]):
              if (occupancy[y,x]>0):
                  num,occupancy = flood_fill_with_threshold(occupancy, (x,y) , 0, 32)
    return occupancy



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









def load_classes_json(filename):
    with open(filename, 'r') as f:
        classes = json.load(f)
    return classes

#def load_keras_model(model_path):
#       return keras.saving.load_model(model_path, compile=True, safe_mode=False)


def checkIfPathIsDirectory(filename):
    return os.path.isdir(filename) 

from torchvision import transforms


def runSingle(image_path,model,device,classes,class_colors,tile_size,step):
    rgba_image = readPolarPNMToRGBA(image_path)
    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA) #Undo opencv crazyness.. <---------------FLIP RGBA to BGRA
    #Normalize the image
    rgba_image = rgba_image / 255.0
    rgba_image = torch.tensor(rgba_image).float().to(device)
    #Normalize the image using tranform
    #mean=[0.485, 0.456, 0.406, 0.406]
    #std=[0.229, 0.224, 0.225, 0.225]
    #rgba_image = (rgba_image - mean) / std
    #Dataset mean:  [[[[ 89.66365  99.0593  103.32124  94.00895]]]]
    #Dataset std:  [[[[66.99435  72.73822  74.47066  71.135704]]]]
    #data_mean = np.array([ 89.66365,  99.0593,  103.32124,  94.00895])
    #data_std  = np.array([ 66.99435,  72.73822,  74.47066,  71.135704])
    #rgba_image = (rgba_image - data_mean) / data_std

    # Perform inference on the image tiles
    predictions,num_classes = classify_tiles(model,rgba_image,tile_size=tile_size, step=step)
    rgba_image= rgba_image * 255.0
    

            
    #minimums,maximums,stds,means = generate_predictionStatistics(predictions,num_classes)
    #print("Minimums ",minimums)
    #print("Maximums ",maximums)
    #print("STDs ",stds)
    #print("Means ",means)

    #predictions = predictions.tolist()
    #dumpListAsCSV(predictions,classes,"predictions.csv")
    #list from numpy array

    # Generate heatmap image
    heatmap,occupancy = generate_heatmap(predictions,classes,class_colors, rgba_image,tile_size=tile_size, step=step)
    return heatmap,occupancy





if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python inference_script.py <model_path> <model_classes> <image_path>")
        print("e.g. python3 evaluate.py tile_classifier.keras tile_classifier.classes magician/16-0.02-lights-negativedents/colorFrame_0_00234.pnm")
        sys.exit(1)

    model_path    = sys.argv[1]
    model_classes = sys.argv[2]
    image_path    = sys.argv[3]

    # Load the trained model
    #import ipdb; ipdb.set_trace()
    #from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy, CategoricalFocalCrossentropy
    #total_loss = CategoricalCrossentropy()
    #model  = keras.models.load_model('tile_classifier.keras', custom_objects={'Classifier': Classifier})
    #model = load_keras_model(model_path)
    #model.compile()
    #Pl Tester
    model = Classifier.load_from_checkpoint(model_path)
    model.eval()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    #Set the model to inference mode
    #classes = load_classes_json(model_classes)
    classes = ['class_neg', 'class_pos', 'class_clean','class_unknown']
    num_classes = len(classes)
    print("num_classes = ",num_classes)

    # Define colors for each class
    #class_colors = generate_colors(num_classes)
    class_colors = []
    class_colors.append((203,192,255)) #<- Black for the background
    class_colors.append((0,0,255)) #<- White for the background
    class_colors.append((0,255,0)) #<- Blue for the background
    class_colors.append((255,0,0)) #<- Red for the background
    
    for i in range(len(classes)):
          print("Class ",i," -> ",classes[i])    


    step=16
    tile_size=64

    if (checkIfPathIsDirectory(image_path)):
        directory = image_path
        print("Doing Directory Dataset ",directory)
        all_files = os.listdir(directory)
        all_files.sort()
        outputNumber = 0
        for image_path in all_files:
          if image_path.lower().endswith('.pnm'):
            print("Image ",image_path)
            heatmap,occupancy = runSingle("%s/%s"%(directory,image_path),model,device,classes,class_colors,tile_size,step)
            hits, occupancy = detect_4x4_rectangle(heatmap,occupancy)
            outputNumber = outputNumber + 1
            #printLabels(heatmap,classes,class_colors)
            cv2.imwrite(f'heatmap_%05u.png'%outputNumber, heatmap)
            cv2.imwrite(f'occupancy_%05u.png'%outputNumber, occupancy)
        os.system("ffmpeg -i heatmap_%05d.png -y heatmap.mp4")
    else:
        heatmap = runSingle(image_path,model,classes,class_colors,tile_size,step)
        printLabels(heatmap,classes,class_colors)
        cv2.imwrite(f'heatmap.png', heatmap[0])
        #cv2.imwrite(f'heatmap.png'%outputNumber, heatmap)
        os.system("timeout 8 gpicview heatmap.png")
