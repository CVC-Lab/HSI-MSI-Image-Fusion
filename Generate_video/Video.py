# importing libraries
import os
from generate_video import generate_video
import numpy as np
  
#print(os.getcwd()) 
  
os.chdir("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/SegTrackv2_small/JPEGImages/bmx")  
path = "/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/SegTrackv2_small/JPEGImages/bmx"
  
# mean_height = 0
# mean_width = 0
  
num_of_images = len(os.listdir('.'))
print(num_of_images)

files = np.array(os.listdir('.'))
#print(files)

video_name = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/grayscale/bmx.avi'
image_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/SegTrackv2_small/JPEGImages/bmx'
#text_file = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/SegTrackv2_small/ImageSets/bmx.txt'

generate_video(video_name, image_folder)