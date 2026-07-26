import sys
import os
import json
import shutil
import cv2
import numpy as np
import random

from readData import loadMoreImages,checkIfFileExists,checkIfPathExists,check_threshold,check_variation

def random_decision(percentage):
    """Return True with probability = percentage/100."""
    return random.random() <= percentage / 100

def sum_non_clean_entries(dictionary):
    """Sum the counts of all non-'clean' entries in a dictionary."""
    total = 0
    for key, value in dictionary.items():
        if key != 'clean':
            total += value
    return total

def max_non_clean_entry(dictionary):
    """Return the maximum value among non-'clean' entries, or None if none exist."""
    max_value = float('-inf')  # Initialize with negative infinity to handle negative values
    for key, value in dictionary.items():
        if key != 'clean' and value > max_value:
            max_value = value
    return max_value if max_value != float('-inf') else None


def isTileWorthAnNN(tile):
      """Check whether a tile has enough visual content to be worth saving as training data."""
      doDump = True
      if not check_threshold(tile,20):
                  doDump = False #<- Ommit very dark tile
      if not check_variation(tile, 30):
                  doDump = False #<- Ommit very uniform tile
      return doDump


#https://keras.io/api/data_loading/image/
"""
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
"""
def dump_dataset_to_keras_data_loader(tiles,tile_classes,occurances,cleanThreshold,outputDirectory="keras_dataset"):
  """Dump tiled images into an ImageFolder-style directory structure for Keras."""
  for sampleID in range(len(tiles)):
      thisClass = tile_classes[sampleID]
      if (thisClass==""):
                thisClass="clean"
      if ( thisClass in occurances ):
          pass
      else:
          occurances[thisClass]=1

  for sampleID in range(len(tiles)):

      #Auto-balance of clean samples
      if (sum_non_clean_entries(occurances)<occurances["clean"]):
          cleanThreshold = min(99.5,cleanThreshold + 0.001)
      else:
          cleanThreshold = max(50.0,cleanThreshold - 0.001)
      #print("Clean Threshold ",cleanThreshold)

      thisClass = tile_classes[sampleID]
      if (thisClass==""):
                thisClass="clean"
 
      #Decide on clean samples
      doDump = True
      if (thisClass=="clean"):
        #doDump = isTileWorthAnNN(tiles[sampleID])
        if random_decision(cleanThreshold):
            doDump = False #<- Ommit 90% of the other tiles
     
      
      if (doDump):
           thisClassNoSpace = thisClass.replace(" ", "")

           if not checkIfPathExists("%s/class_%s" % (outputDirectory,thisClassNoSpace) ):
               os.makedirs("%s/class_%s" % (outputDirectory, thisClassNoSpace), exist_ok=True)

           targetPath = '%s/class_%s/%s_image_%u.png' % (outputDirectory,thisClassNoSpace,thisClassNoSpace,occurances[thisClass]) 
           cv2.imwrite(targetPath, tiles[sampleID])
           #print("Target ",targetPath)
           occurances[thisClass]+=1
  print("Clean Threshold ",cleanThreshold)
  return occurances,cleanThreshold


if __name__ == '__main__':
  #---------------------------------------
  step      = 32
  tile_size = 64 
  files               = []
  tiles               = []
  tile_classes        = []
  class_dict          = dict() 
  occurances          = dict()
  occurances["clean"] = 1 #<- force this
  #---------------------------------------
  #Dynamic way to balance clean with non-clean data
  cleanThreshold = 60
  #---------------------------------------

  output_file = "keras_dataset/"

  argumentStart = 1
  if len(sys.argv) > 1:
    for i in range(0, len(sys.argv)):
        if sys.argv[i] == "--subdirectories":
            directory = sys.argv[i + 1]
            argumentStart += 2
            if os.path.isdir(directory):
                for root, _, filenames in os.walk(directory):
                    for file in sorted(filenames):
                        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.pnm')):
                            files.append(os.path.join(root, file))
            else:
                print(f"Error: Directory '{directory}' does not exist.")
                sys.exit(1)
        elif sys.argv[i] == "--directory":
            directory = sys.argv[i + 1]
            argumentStart += 2
            # Populate files with image (.jpg, .png) contents of directory
            if os.path.isdir(directory):
                directoryList = os.listdir(directory)
                directoryList.sort()
                for file in directoryList:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg', '.pnm')):
                        files.append(os.path.join(directory, file))
            else:
                print(f"Error: Directory '{directory}' does not exist.")
                sys.exit(1)
        elif sys.argv[i] in ("--output", "-o"):
            output_file = sys.argv[i + 1]
            argumentStart += 2

  for i in range(argumentStart, len(sys.argv)):
    files.append(sys.argv[i])

  #---------------------------------------
  if os.path.exists(output_file):
      shutil.rmtree(output_file)
  os.makedirs(output_file)
  #---------------------------------------


  #---------------------------------------
  for index, arg in enumerate(files, start=1):
     if (checkIfFileExists(arg) and checkIfFileExists("%s.json"%arg)):
        print(f"Loading Images / Argument {index}: {arg}")
        tiles        = [] #Erase so that we dont run out of memory
        tile_classes = []
        tiles,tile_classes        = loadMoreImages(arg,index,tiles=tiles,tile_classes=tile_classes,tile_size=tile_size,step=step)
        occurances,cleanThreshold = dump_dataset_to_keras_data_loader(tiles,tile_classes,occurances,cleanThreshold, outputDirectory=output_file)
        print(" Had ",occurances," occurances")
     else:
        print(f"NOT LOADING Argument {index}: {arg}")
  #---------------------------------------

