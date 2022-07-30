import numpy as np
import cv2
import sys
import warnings
import os
from generate_video import generate_video
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI_Fusion2/codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI_Fusion2/codes/hyperspectral_image_processing/simulation')
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI_Fusion2/codes/hyperspectral_image_processing/fusion')
warnings.filterwarnings('ignore')

from BGLRF_main import BGLRF_main
from Generation import generation
from denoising import denoising

hsi_vid = cv2.VideoCapture('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/low_resolution_color/bmx.avi')
msi_vid = cv2.VideoCapture('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/grayscale/bmx.avi')

hsi_success, hsi_image = hsi_vid.read()
count = 0
while hsi_success:
  cv2.imwrite("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/hsi/frame%d.jpg" % count, hsi_image)     # save frame as JPEG file      
  hsi_success,hsi_image = hsi_vid.read()
  #print('Read a new frame: ', hsi_success)
  count += 1

msi_success, msi_image = msi_vid.read()
count = 0
while msi_success:
  cv2.imwrite("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/msi/frame%d.jpg" % count, msi_image)     # save frame as JPEG file      
  msi_success,msi_image = msi_vid.read()
  #print('Read a new frame: ', msi_success)
  count += 1

hsi_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/hsi'
msi_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/msi'
for i in range(len(os.listdir('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/hsi'))):
    hsi = cv2.imread(os.path.join(hsi_folder, 'frame%d.jpg' %i))
    msi = cv2.imread(os.path.join(msi_folder, 'frame%d.jpg' %i))
    msi = cv2.cvtColor(msi, cv2.COLOR_BGR2GRAY)
    msi = msi.reshape((msi.shape[0],msi.shape[1],1))
    print(msi)
    # msi = msi/np.max(msi)
    msi = msi/255

    # hsi = hsi/np.max(hsi)
    hsi = hsi/255
    #print(hsi.shape)
    denoised_hsi, SNR_db = denoising(hsi)

    print(denoised_hsi.shape)
    print(msi.shape)
    # denoised_hsi = np.round(denoised_hsi*255)
    # denoised_hsi = denoised_hsi.astype(np.uint8)

    SRI_fused,K_est = BGLRF_main(denoised_hsi,msi,10,10,6)

    SRI_fused = np.round(SRI_fused*255)
    SRI_fused = SRI_fused.astype(np.uint8)

    cv2.imwrite('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/fused/frame.jpg', SRI_fused)

    SRI_fused = np.reshape(SRI_fused, (360,360,3))
    # cv2.imshow(SRI_fused)
    plt.imshow(SRI_fused)
    plt.show()
    cv2.waitKey(0) 

cv2.destroyAllWindows() 

video_name = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/fused/bmx.avi'
image_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/fused'

generate_video(video_name, image_folder)