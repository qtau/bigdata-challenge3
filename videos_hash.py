import os
import cv2
import sys
import numpy as np 
import imagehash
import time
import json
import PIL
import dhash

with open("videos_name.json", "r") as file:
    videos_name = json.load(file)

with open("videos_indice.json", "r") as file:
    videos_indice = json.load(file)

def get_hash(image):
    row, col = dhash.dhash_row_col(PIL.Image.fromarray(image))
    return dhash.format_hex(row, col)

def get_video_average(video_name, nb_frame_selection = 1,plot = False):
    start_time = time.time()
    # Input: 
    # video_name: path of a video
    # frame_rate_selection: step between two selected frames
    # Output: 4 dimension np.array of the selected frames 
    vidcap = cv2.VideoCapture(video_name)
    
    nb_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames
    
    success,image = vidcap.read() # Read the first frame                          
    video_dim = image.shape       # Shape of the frames
    
    # indice of the frame to select
    indice_selection_frame = [0]
    indice_selection_frame.extend(list(range(0,nb_frame-1,(nb_frame-1)//(nb_frame_selection+1)))[1:(nb_frame_selection+1)])
    indice_selection_frame.append(nb_frame-1)
    
    # Preallocation of the np.array  
    video_frames = np.empty(video_dim + (len(indice_selection_frame),), dtype= 'uint8')
    
    # Initialization 
    video_frames[:,:,:,0] = image

    # Saving
    count = 1
    for indice in indice_selection_frame[1:]:
        vidcap.set(1,indice)
        success, image = vidcap.read()
        video_frames[:,:,:,count] = image
        count +=1

    vidcap.release()

    video_average_image = video_frames.mean(axis=3).astype('uint8')
    
    return(video_average_image)

########## MAIN #############
print(" ################  GET_AVERAGE_IMAGE.PY   ####################")

videos_hash = {}
start_time = time.time()
for indice, video_name in enumerate(videos_name):
    video_average_image = get_video_average(video_name,nb_frame_selection=10)
    video_hash = get_hash(video_average_image)

    videos_hash[videos_indice[indice]] = video_hash
    
    
    if indice % 100 == 0:
        print("Number of processed videos: " + str(indice) + " --- time: " + str(time.time()-start_time))
        

print("Total time: " + str(time.time()-start_time))
print("Saving the file")

with open("videos_hash_2.json", "w") as file:
    json.dump(videos_hash,file)

print("Job done !")
