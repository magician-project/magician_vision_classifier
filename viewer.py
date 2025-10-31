import sys
import os 
import cv2
import numpy as np
import time

from evaluate import generate_predictionStatistics, generate_colors, generate_heatmap, classify_tiles, load_classes_json, load_keras_model, checkIfPathIsDirectory, printLabels, castData

def convertPolarCVMATToRGBA(image):
    if image is None:
        print("Error: Unable to read the image.")
        return None

    height, width, channels = image.shape
    #if channels == 3: 
    #    print("Casting RGB image as monochrome")
    #    image = image[:,:,0]
    image = image[:,:,0]

    # Split into polarization images
    from readData import debayerPolarImage
    polarization_0_deg, polarization_45_deg, polarization_90_deg, polarization_135_deg = debayerPolarImage(image)

    # Create an RGBA image
    rgba_image = np.zeros((int(height/2),int(width/2), 4), dtype=np.uint8)

    # Assign each polarization image to a specific channel
    rgba_image[:, :, 0] = polarization_0_deg
    rgba_image[:, :, 1] = polarization_45_deg
    rgba_image[:, :, 2] = polarization_90_deg
    rgba_image[:, :, 3] = polarization_135_deg
    return rgba_image

def runSingleFrame(image, startC=0, endC=3):
    rgba_image = convertPolarCVMATToRGBA(image)
    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA) #Undo opencv crazyness.. <---------------FLIP RGBA to BGRA

    return rgba_image[:,:,startC:endC]

def main(streamName):
    from SharedMemoryManager import SharedMemoryManager
    smm = SharedMemoryManager("libSharedMemoryVideoBuffers.so", 
                              descriptor = "video_frames.shm", 
                              frameName  = streamName,
                              connect    = True)

    snapshotNumber = 0
    startC=0 
    endC  =3 
    # Loop to continuously read frames 
    while True:
        # Capture frame-by-frame
        frame = smm.read_from_shared_memory()
        
        # Check if the frame is captured successfully
        if  (frame is None) or (smm.frame_size==0):
            print("Error: Couldn't read frame from SHM")
        else:
           raw = runSingleFrame(frame, startC=startC, endC=endC)
           cv2.imshow('Camera ', raw)
           # Break the loop if 'q' is pressed

           key = cv2.waitKey(1)
           if key & 0xFF == ord('1'):
               startC=0 
               endC  =3 
           elif key & 0xFF == ord('2'):
               startC=1 
               endC  =4 
           elif key & 0xFF == ord('s'):
               snapshotNumber = snapshotNumber + 1 
               cv2.imwrite('viewer_%05u.png'%snapshotNumber,raw)
           elif key & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python viewer.py")
        sys.exit(1)

    #Link to SharedMemoryVideoBuffers :
    #      cd ../../../dependencies && git clone git@github.com:AmmarkoV/SharedMemoryVideoBuffers.git
    os.system("ln -s ../../../dependencies/SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so")
    os.system("ln -s ../../../dependencies/SharedMemoryVideoBuffers/src/python/SharedMemoryManager.py")


    streamName    = "stream1"


    main(streamName)

