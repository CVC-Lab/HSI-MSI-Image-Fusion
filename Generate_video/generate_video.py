import os
import cv2 
import sys
import warnings
from generate_hsi import generate_hsi
import numpy as np

sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/simulation')
warnings.filterwarnings('ignore')

from denoising import denoising
from add_noise import add_noise

# Video Generating function
def generate_video(video_name, image_folder):
    video_name = video_name
    os.chdir(image_folder)
      
    images = [img for img in sorted(os.listdir(image_folder))
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
     
    print(images) 
  
    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
    print(frame.shape)
  
    video = cv2.VideoWriter(video_name, 0, 5, (width, height)) 

    #Uncomment while generating lower resolution video
    #video = cv2.VideoWriter(video_name, 0, 1, (int(width/4), int(height/4))) 
  
    # Appending the images to the video one by one
    for image in images: 

        im = cv2.imread(os.path.join(image_folder, image))
        
        # Uncomment the lines below to denoise image
        im = im/255
        im, SNR_db = denoising(im)
        im = np.round(im*255)
        im = im.astype(np.uint8)

        #Uncomment the line below to generate lower resolution image
        #im = generate_hsi(im, height, width, layers)

        # Uncomment the lines below to generate msi colored
        # im = im/255
        # im,SNR_db = denoising(im)
        # im, noise = add_noise(im, 40)
        # im = np.round(im*255)
        # im = im.astype(np.uint8)
        # grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # print(grayscale.shape)
        # im = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        # #im.show
        
        video.write(im)

        #video.write(im) 
        # print('Image written')
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated