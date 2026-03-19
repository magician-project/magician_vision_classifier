import sys
import os
import pygame
import cv2
import numpy as np
import time

from evaluate import generate_predictionStatistics, generate_colors, generate_heatmap, classify_tiles, load_classes_json, load_keras_model, checkIfPathIsDirectory, printLabels, castData, detect_4x4_rectangle

def play_mp3(file_path):
    print("Play mp3",file_path)
    # Load the MP3 file
    pygame.mixer.music.load(file_path)
    
    # Play the MP3 file
    pygame.mixer.music.play()
     
    while pygame.mixer.music.get_busy():
            time.sleep(0.1)

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

def runSingleFrame(image,model,classes,class_colors,tile_size,step):

    # Convert the incoming image to polarisations
    start      = time.time()    
    #---------------------------------------------
    rgba_image = convertPolarCVMATToRGBA(image)
    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA) #Undo opencv crazyness.. <---------------FLIP RGBA to BGRA
    #---------------------------------------------
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    print("Pre-processing image @ %0.02f Hz"%hz)    


    # Perform inference on the image tiles
    start      = time.time()    
    #---------------------------------------------
    predictions = classify_tiles(model, rgba_image,tile_size=tile_size, step=step)
    #---------------------------------------------
    seconds    = time.time() - start
    NNhz    = 1 / (seconds+0.0001)
    print("Classify tiles @ %0.02f Hz"%NNhz)    


    #minimums,maximums,stds,means = generate_predictionStatistics(predictions)
    #print("Minimums ",minimums)
    #print("Maximums ",maximums)
    #print("STDs ",stds)
    #print("Means ",means)
    #dumpListAsCSV(predictions,classes,"predictions.csv")

    # Generate heatmap image
    start      = time.time()    
    #---------------------------------------------
    heatmap,occupancy = generate_heatmap(predictions,classes,class_colors, rgba_image,tile_size=tile_size, step=step)
    #---------------------------------------------
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    print("Generate Heatmap @ %0.02f Hz"%hz)  


    cv2.putText(heatmap,"NN @ %0.2f Hz" % NNhz, (20,900) , cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255) , 5)  

    return heatmap,occupancy




def main(model,classes,class_colors,streamName):
    from SharedMemoryManager import SharedMemoryManager
    smm = SharedMemoryManager("libSharedMemoryVideoBuffers.so", 
                              descriptor = "video_frames.shm", 
                              frameName  = streamName,
                              connect    = True)


    step=16
    tile_size=64

    mode=0
    # Loop to continuously read frames
    while True:
        # Capture frame-by-frame
        frame = smm.read_from_shared_memory()
        
        # Check if the frame is captured successfully
        if  (frame is None) or (smm.frame_size==0):
            print("Error: Couldn't read frame from SHM")
        else:        
           #print("frame:",frame)

           # Display the frame in a window -> This is pretty big
           #cv2.imshow('SharedMemoryVideoBuffer', frame)

           #pygame.mixer.music.play()
           heatmap,occupancy = runSingleFrame(frame,model,classes,class_colors,tile_size,step)

           hits = []
           if (mode==1):
             hits, occupancy = detect_4x4_rectangle(heatmap,occupancy)
             
           #printLabels(heatmap,classes,class_colors)
           cv2.imshow('Detection', heatmap)
           
           # Break the loop if 'q' is pressed
           key = cv2.waitKey(1)

           if (len(hits)>0):
               play_mp3("denied.mp3") 

           if key & 0xFF == ord('1'):
               mode=0
           elif key & 0xFF == ord('2'):
               mode=1  
           elif key & 0xFF == ord('a'):
               step=16
           elif key & 0xFF == ord('s'):
               step=32  
           elif key & 0xFF == ord('d'):
               step=64  
           elif key & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python inference_script.py <model_path> <model_classes>")
        print("e.g. python3 client.py tile_classifier.keras tile_classifier.classes")
        sys.exit(1)

    #Link to SharedMemoryVideoBuffers :
    #      cd ../../../dependencies && git clone git@github.com:AmmarkoV/SharedMemoryVideoBuffers.git
    os.system("ln -s ../../../dependencies/SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so")
    os.system("ln -s ../../../dependencies/SharedMemoryVideoBuffers/src/python/SharedMemoryManager.py")


    # Play sound files
    pygame.mixer.init() 
    play_mp3("ping.mp3") 
        

    streamName    = "stream1"
    model_path    = sys.argv[1]
    model_classes = sys.argv[2] 

    # Load the trained model
    model   = load_keras_model(model_path)
    classes = load_classes_json(model_classes)
    
    num_classes = len(classes)
    print("num_classes = ",num_classes)

    # Define colors for each class
    class_colors = generate_colors(num_classes)

    for i in range(len(classes)):
          print("Class ",i," -> ",classes[i])    


    main(model,classes,num_classes,class_colors,streamName)

