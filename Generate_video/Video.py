# importing libraries
import os
import pwd
from generate_video import generate_video
import numpy as np
  
#print(os.getcwd()) 
  

# path = "../SegTrackv2_small/JPEGImages/hummingbird"
# os.chdir(os.path.join(pwd,path)) 
  
# mean_height = 0
# mean_width = 0

num_of_images = len(os.listdir('.'))
print(num_of_images)

files = np.array(os.listdir('.'))
#print(files)

video_name = './Videos/low resolution color/hummingbird.avi'
image_folder = '../SegTrackv2_small/JPEGImages/hummingbird'
#text_file = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/SegTrackv2_small/ImageSets/bmx.txt'

generate_video(video_name, image_folder)