import os
import cv2 
import sys
import warnings
from generate_hsi import generate_hsi
from generate_hsi import createnewbands
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../HSI-MSI_Fusion2/codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, '../HSI-MSI_Fusion2/codes/hyperspectral_image_processing/simulation')
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
  
    # video = cv2.VideoWriter(video_name, 0, 5, (width, height)) 
    video = cv2.VideoWriter(video_name, 0, 5, (height, height)) 
    # video = cv2.VideoWriter(video_name, 0, 5, (144, 144)) 

    #Uncomment while generating lower resolution video
    #video = cv2.VideoWriter(video_name, 0, 5, (int(width/4), int(height/4))) 
    # video = cv2.VideoWriter(video_name, 0, 5, (int(height/4), int(height/4))) 
    # video = cv2.VideoWriter(video_name, 0, 5, (36, 36)) 
  
    # Appending the images to the video one by one
    for image in images: 

        im = cv2.imread(os.path.join(image_folder, image))
        
        

        im_cropped = im[:,140:500]
        # im_cropped = im[90:270,320:500]
        im = im_cropped

        

        # im = cv2.resize(im, (36, 36),interpolation = cv2.INTER_NEAREST)
        # im = cv2.resize(im, (144, 144),interpolation = cv2.INTER_NEAREST)
        # im = cv2.resize(im, (36, 36),interpolation = cv2.INTER_CUBIC)
        print(im.shape)
        # plt.imshow(im)
        # plt.show()
        
        im, noise = add_noise(im, 10)

        # Uncomment the lines below to denoise image
        # im = im/255
        # im, SNR_db = denoising(im)
        # im, _ = add_noise(im, 20)
        plt.imshow(im)
        plt.show()        
        # im = np.round(im*255)
        # im = im.astype(np.uint8)

        # cv2.imshow("original", im)
        # # plt.imshow(im)
        # plt.show()
        

        #Uncomment the line below to generate lower resolution image
        # im = generate_hsi(im, height, width, layers)
        # im = generate_hsi(im, height, height, layers)
        # im = generate_hsi(im, 144, 144, layers)
        # im = createnewbands(im, 144, 144, layers)
        # plt.imshow(im)
        # plt.show()

        # Uncomment the lines below to generate msi colored
        # im = im/255
        # im,SNR_db = denoising(im)
        # im, noise = add_noise(im, 60)
        # im = np.round(im*255)
        # im = im.astype(np.uint8)
        # grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # # print(grayscale.shape)
        # # print(np.max(grayscale))
        # im = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        # # plt.imshow(im)
        # # plt.show()

        # #im.show
        
        video.write(im)

        #video.write(im) 
        # print('Image written')
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated