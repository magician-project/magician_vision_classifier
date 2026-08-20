#!/usr/bin/python3

""" 
Author : "Nikos Vasilikopoulos, Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 

Live/CLI streaming entry point. The classifier core (ClassifierPnm, tiling,
heatmaps, model_scan, ...) lives in classifierPnm.py; everything is re-exported
here so existing `from liveClassifierTorch import ...` imports keep working.
"""

from mvc.inference.classifier_pnm import *   # noqa: F401,F403 -- re-export the classifier core

if __name__ == "__main__":
    #model_path    = sys.argv[1]
    #model_classes = sys.argv[2]
    #image_path    = sys.argv[3]

    step=16
    threshold = 0.9

    modelC = ClassifierPnm(model_path='last.pth',cfg_path='last.json',step=step)
    modelC.maxProbabilityThreshold = threshold
    model=modelC.model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model=model.eval()
    #model.half()
    #Compile the model for inference
    model = torch.compile(model)
    #model = torch.jit.trace(model, tiles)
    #model=model.half()
    #Set the model to inference mode
    torch.set_float32_matmul_precision('medium') 


    if checkIfFileExists("libSharedMemoryVideoBuffers.so"):
            print("Found a shared memory video buffer library..!")
    else:
            print("Bootstrapping a new shared memory video buffer library")
            #os.system("ln -s %s/libSharedMemoryVideoBuffers.so" % classifier_relative_directory)
            os.system("git clone https://github.com/AmmarkoV/SharedMemoryVideoBuffers")
            os.system("cd SharedMemoryVideoBuffers && make && cd ..")
            os.system("ln -s SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so" )
            os.system("SharedMemoryVideoBuffers/server --nokb&")
    
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
