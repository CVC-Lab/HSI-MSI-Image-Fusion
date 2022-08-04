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
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI_Fusion2/codes/hyperspectral_image_processing/quality_metrics')
warnings.filterwarnings('ignore')

from BGLRF_main import BGLRF_main
from Generation import generation
from denoising import denoising
from quality_assessment import quality_assessment

# hsi_vid = cv2.VideoCapture('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/low_resolution_color/bmx.avi')
# msi_vid = cv2.VideoCapture('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/grayscale/bmx.avi')

hsi_vid = cv2.VideoCapture('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/low_resolution_color/hummingbird.avi')
msi_vid = cv2.VideoCapture('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/grayscale/hummingbird.avi')
sri_vid = cv2.VideoCapture('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/originals/hummingbird.avi')

hsi_success, hsi_image = hsi_vid.read()
count = 0
while hsi_success:
#   cv2.imwrite("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/hsi/frame%d.jpg" % count, hsi_image)     # save frame as JPEG file      
  cv2.imwrite("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/hsi/frame%d.jpg" % count, hsi_image)     # save frame as JPEG file      
  hsi_success,hsi_image = hsi_vid.read()
  #print('Read a new frame: ', hsi_success)
  count += 1

msi_success, msi_image = msi_vid.read()
count = 0
while msi_success:
#   cv2.imwrite("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/msi/frame%d.jpg" % count, msi_image)     # save frame as JPEG file      
  cv2.imwrite("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/msi/frame%d.jpg" % count, msi_image)     # save frame as JPEG file      
  msi_success,msi_image = msi_vid.read()
  #print('Read a new frame: ', msi_success)
  count += 1

sri_success, sri_image = sri_vid.read()
count = 0
while sri_success:
#   cv2.imwrite("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/msi/frame%d.jpg" % count, msi_image)     # save frame as JPEG file      
  cv2.imwrite("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/sri_gt/frame%d.jpg" % count, sri_image)     # save frame as JPEG file      
  sri_success,sri_image = sri_vid.read()
  #print('Read a new frame: ', msi_success)
  count += 1

# hsi_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/hsi'
# msi_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/msi'
# for i in range(len(os.listdir('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/bmx/hsi'))):

hsi_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/hsi'
msi_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/msi'
sri_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/sri_gt'

# for i in range(len(os.listdir('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/hsi'))):
# for i in range(17, 28):
for i in 20,21,22,23:
    print("Frame",i+1)
    hsi = cv2.imread(os.path.join(hsi_folder, 'frame%d.jpg' %i))
    msi = cv2.imread(os.path.join(msi_folder, 'frame%d.jpg' %i))
    sri = cv2.imread(os.path.join(sri_folder, 'frame%d.jpg' %i))
    msi = cv2.cvtColor(msi, cv2.COLOR_BGR2GRAY)
    msi = msi.reshape((msi.shape[0],msi.shape[1],1))

    # plt.imshow(msi)
    # plt.show()
    # print(msi)
    # msi = msi/np.max(msi)
    msi = msi/255

    # hsi = hsi/np.max(hsi)
    hsi = hsi/255
    sri = sri/255
    #print(hsi.shape)
    denoised_hsi, SNR_db = denoising(hsi)
    sri, noise = denoising(sri)

    print(denoised_hsi.shape)
    print(msi.shape)
    # print(type(denoised_hsi))
    # print(np.max(msi))
    # denoised_hsi = np.round(denoised_hsi*255)
    # denoised_hsi = denoised_hsi.astype(np.uint8)

    # plt.imshow(sri)
    # plt.show()

    # plt.imshow(hsi)
    # plt.show()

    # plt.imshow(msi)
    # plt.show()

    SRI_fused,K_est = BGLRF_main(denoised_hsi,msi,10,10,6)

    quality_assessment(sri,SRI_fused,0,1/4)

    # SRI_fused = SRI_fused.T
    SRI_fused = np.reshape(SRI_fused, (msi.shape[0],msi.shape[1],3))

   
    # SRI_fused = cv2.rotate(SRI_fused, cv2.ROTATE_90_CLOCKWISE)
    # SRI_fused = cv2.flip(SRI_fused, 1)
    # plt.imshow(SRI_fused)
    # plt.show()

    SRI_final = np.zeros((sri.shape[0],sri.shape[1],3))
    SRI_final[:,:,0] = SRI_fused[:,:,0].T
    SRI_final[:,:,1] = SRI_fused[:,:,1].T
    SRI_final[:,:,2] = SRI_fused[:,:,2].T
    print(SRI_final.shape)

    # plt.imshow(SRI_final)
    # plt.show()

    # SRI_fused = np.round(SRI_fused*255)
    # SRI_fused = SRI_fused.astype(np.uint8)

    # plt.imshow(SRI_fused)
    # plt.show()

    SRI_fused = SRI_final
    SRI_fused = np.round(SRI_fused*255)
    SRI_fused = SRI_fused.astype(np.uint8)
    # SRI_fused = cv2.rotate(SRI_fused, cv2.ROTATE_90_CLOCKWISE)
    # SRI_fused = cv2.flip(SRI_fused, 1)
    

    cv2.imwrite('/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/fused/frame%d.jpg' %i, SRI_fused)

    # SRI_fused = np.reshape(SRI_fused, (msi.shape[0],msi.shape[1],3))
    # cv2.imshow(SRI_fused)
    
    # cv2.waitKey(0) 

# cv2.destroyAllWindows() 

video_name = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/Videos/fused/hummingbird.avi'
image_folder = '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/Generate_video/fused_frames/hummingbird/fused'

# generate_video(video_name, image_folder)




# Video Generating function

video_name = video_name
os.chdir(image_folder)
      
images = [img for img in sorted(os.listdir(image_folder))
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
     
print(images) 
  
frame = cv2.imread(os.path.join(image_folder, images[0]))
  
height, width, layers = frame.shape  
print(frame.shape)
  
video = cv2.VideoWriter(video_name, 0, 5, (width, height)) 
    # Appending the images to the video one by one
for image in images: 

    im = cv2.imread(os.path.join(image_folder, image))
        
    video.write(im)

    # Deallocating memories taken for window creation
    # cv2.destroyAllWindows() 
video.release()  # releasing the video generated