#!/usr/bin/python3
""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH" 
"""

import os
import sys
import json
import torch
import cv2
import numpy as np
from torchvision import transforms
from liveClassifierTorch import ClassifierPnm, classify_tiles

def get_png_comment(filename):
    from PIL import Image
    try:
        img = Image.open(filename)
        comment = img.info.get("Comment", "")
        img.close()
        return comment
    except Exception:
        return ""

def load_rgba_image_with_comment(image_path):
    """
    Loads an RGBA image and extracts its PNG comment.
    Returns: (numpy_image, comment_string)
    """
    #try:
    #    # Read comment using PIL
   #     pil_img = Image.open(image_path)
    #    comment = pil_img.info.get("Comment", "")
    #    pil_img.close()
    #except Exception:
    #    comment = ""

    comment = get_png_comment(image_path)
    # Load image using OpenCV
    rgba_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if rgba_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Ensure RGBA order (OpenCV uses BGRA)
    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_BGRA2RGBA)
    rgba_image = rgba_image.astype('float32') / 255.0
    return rgba_image, comment

def find_clean_confusions_from_directory_tiled(
    model,
    directory,
    class_names,
    device,
    tile_size=64,
    step=0,
    chunks=0,
    thresholdMaxProbability=0.50,
    forceLowMaxProbToThisClass=None,
    majorityVote=True,
):
    """
    Iterates through directory/<class_name>/*.png,
    classifies each using classify_tiles(),
    and prints filenames + comments when predicted as class_clean
    while the true class is not class_clean.
    """
    model.eval()
    clean_idx = class_names.index("class_clean")
    confused_samples = []


    clean_idx = 0
    for i in range(len(class_names)):
           if (class_names[i] == "class_clean") or (class_names[i] == "Clean"):
              clean_idx = i

    with torch.no_grad():
        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                print(f"Path: {class_dir} is not a directory with tiles inside")
                continue
            if (class_name == "class_clean"):
                continue # Do not process class clean


            print(f"\nProcessing class directory: {class_dir}")

            for filename in os.listdir(class_dir):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                filepath = os.path.join(class_dir, filename)

                try:
                    rgba_image, comment = load_rgba_image_with_comment(filepath)
                    #print(f"{filepath}: {rgba_image.shape}")
                except Exception as e:
                    print(f"⚠️ Failed to load {filepath}: {e}")
                    continue

                try:
                    predictions = classify_tiles(
                        model=model,
                        rgba_image=rgba_image,
                        tile_size=tile_size,
                        step=step,
                        chunks=chunks,
                        majorityVote=majorityVote,
                        thresholdMaxProbability=thresholdMaxProbability,
                        forceLowMaxProbToThisClass=forceLowMaxProbToThisClass,
                    )
                except Exception as e:
                    print(f"⚠️ classify_tiles failed on {filepath}: {e}")
                    continue


                # Aggregate predicted class over all tiles (majority vote)
                if len(predictions) == 0:
                    print(f"⚠️ No predictions for {filepath}")
                    continue

                final_pred = predictions[0]
                #print("Prediction ",final_pred," clean ",clean_idx)

                if final_pred == clean_idx:
                    confused_samples.append({
                        "filename": filepath,
                        "true_class": class_name,
                        "predicted_class": class_names[final_pred],
                        "comment": comment,
                    })
                    print(f"[CONFUSED] {filepath}")
                    if comment:
                        print(f"  → Comment: {comment}")

    print(f"\nFound {len(confused_samples)} files misclassified as 'class_clean'\n")
    return confused_samples

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_classifier = ClassifierPnm(model_path='last.pth',cfg_path='last.json')

         

    confused = find_clean_confusions_from_directory_tiled(
    model=model_classifier.model,
    directory="keras_dataset/",
    class_names=model_classifier.classes,
    device=device,
    tile_size=model_classifier.tile_size,
    step=model_classifier.tile_size,
    chunks=0,
    thresholdMaxProbability=0.5,
    forceLowMaxProbToThisClass=None,
    majorityVote=False
)


    print("Saving the Confusions")
    with open('confusions.json', 'w') as f:
       json.dump(confused, f, indent=2)

