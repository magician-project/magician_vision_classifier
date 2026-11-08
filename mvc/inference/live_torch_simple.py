#!/usr/bin/python3

""" 
Author : "Nikos Vasilikopoulos, Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 

Live/CLI streaming entry point. The classifier core (ClassifierPnm, tiling,
heatmaps, model_scan, ...) lives in classifierPnm.py; everything is re-exported
here so existing `from liveClassifierTorch import ...` imports keep working.
"""

import time

from mvc.inference.classifier_pnm import *   # noqa: F401,F403 -- re-export the classifier core
from mvc.core.shared_memory import SharedMemoryManager

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
    
    streamName = "stream1"
    smm = SharedMemoryManager("./libSharedMemoryVideoBuffers.so", 
                              descriptor = "video_frames.shm", 
                              frameName  = streamName,
                              connect    = True)

    last_processed_timestamp = None   # frame already classified: running the network on it again wastes the GPU
    reported_error           = False  # report a stream that can't be read once, not on every poll

    # Loop to continuously read frames
    while True:
        frame = None
        # Only copy and classify frames that are new
        timestamp = smm.get_timestamp()
        if (timestamp is not None) and (timestamp != last_processed_timestamp):
            frame = smm.read_from_shared_memory()

        if (frame is None) or (smm.frame_size==0):
            if (timestamp is None) and (not reported_error):
                print("Error: Couldn't read frame from SHM")
                reported_error = True
            time.sleep(0.001) # nothing new yet: don't spin
        else:
            reported_error = False
            last_processed_timestamp = smm.unix_timestamp
            # The classifier takes the frame as published (4 polarization channels, or the raw
            # 1 channel mosaic), the same way live_torch.py passes it
            with torch.inference_mode():
                heatmap, occupancy, responses = modelC.forward(frame, majorityVote=True)
                cv2.imshow('Live Heatmap', heatmap)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all OpenCV windows
    cv2.destroyAllWindows()
