import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy.io
import cv2
from scipy.interpolate import CubicSpline

import sys
import warnings

<<<<<<< HEAD:hmi_fusion/models/hip/generate_video/generate_hsi.py
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI_Fusion2/codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI_Fusion2/codes/hyperspectral_image_processing/simulation')
=======
sys.path.insert(0, '../HSI-MSI_Fusion2/codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, '../HSI-MSI_Fusion2/codes/hyperspectral_image_processing/simulation')
>>>>>>> main:Generate_video/generate_hsi.py
warnings.filterwarnings('ignore')

from denoising import denoising
from add_noise import add_noise
from MSG import matlab_style_gauss2D

def generate_hsi(image, height, width, layers):

    #print(type(image[0][0][0]))
    image = image/255
    sri,a = denoising(image)
    sri = sri[0:height-1,0:width-1,:]

    N1 = int(height-1)
    N2 = int(width-1)
    N3 = int(layers)

    # Shifted blur kernel
    k0 = matlab_style_gauss2D()
    radius = 6
    k = np.zeros((13,13))
    k[0:9,0:9] = k0
    
    center = np.zeros((2,1))
    center[0]=(height-1)/2+1
    center[1]=(width-1)/2+1
    
    I = np.linspace(center[0]-radius,center[1]+radius,13)
    I2 = np.linspace(center[0]-radius,center[1]+radius,13)
    for _ in range(12):
        I2 = np.append(I2,I)
    
    I2 = np.reshape(I2,(169,1))
    J = np.linspace(center[1]-radius,center[1]+radius,13)
    J = np.repeat(J,13,axis = 0)

    arr = np.array([I2,J])
    arr = np.reshape(arr,(2,169))
    arr = arr.astype(int)
    I2 = I2.astype(int)
    J = J.astype(int)
    inds = np.ravel_multi_index(arr,(height-2,width-2))
    a=0
    kk = np.zeros((height-1,width-1))
    for i in range(13):
        for j in range(13):
            
            kk[I2[a],J[a]] = k[j,i]
            a=a+1
        
    #Shift Kernel
    kk = np.roll(kk,-int((height+2)/2), axis=0)
    kk = np.roll(kk,-int((height+2)/2), axis=1)
    fft = np.fft.fft2(kk)

    n1 = int(height/4)
    n2 = int(width/4)
    hsi = np.zeros((n1,n2,N3))

    for band in range(N3):
        x = sri[:,:,band]
        Fx = np.fft.fft2(x)
        x = np.multiply(Fx,fft)
        x = np.real(np.fft.ifft2(x))
        hsi[:,:,band] = x[1:-1:4,1:-1:4]
    
    #print(hsi)
    
    hsi,noise_ten2 = add_noise(hsi,40)
    hsi = np.round(hsi*255)
    hsi = hsi.astype(np.uint8)
    #print(type(hsi))
    return hsi

def createnewbands(image, height, width, layers):
    
    mat2 = scipy.io.loadmat("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion original/codes/hyperspectral_image_processing/simulation/Landsat_TM5.mat")
    
    S1 = mat2["blue"]
    S2 = mat2["green"]
    S3 = mat2["red"]
    S4 = mat2["nir"]
    S5 = mat2["swir1"]
    S6 = mat2["swir2"]
    
    S = [S1,S2,S3,S4,S5,S6]

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # (thresh, bwimage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # plt.imshow(bwimage)
    # plt.show

    image = image/255
    sri,a = denoising(image)
    sri = sri[0:height-1,0:width-1,:]

    wave = np.linspace(400,2500, num=3)
    # wave = np.delete(wave,bands_removed[:]-1)

    P3 = np.zeros((3,6))
    
    for i in range(6):
        s = S[i]
        temp1 = s[:,0]
        temp2 = s[:,1]
        ind1 = np.where(wave>temp1[0])
        inds1 = ind1[0]
        inds1 = inds1[0]
        ind2 = np.where(wave<temp1[-1])
        inds2 = ind2[-1]
        inds2 = inds2[-1]

        x = scipy.interpolate.CubicSpline(temp1,temp2)
        yy = x(wave[inds1:inds2+1])
        
        P3[inds1:inds2+1,i] = yy
        P3[:,i] = P3[:,i]/np.sum(P3[:,i])

    sri2 = np.zeros((height*height,220))
    
    for i in range(3):
        sri2[:,i]= np.reshape(sri[:,:,i].T,(height*height))
    msi = np.matmul(sri2,P3)
    msi = np.reshape(msi,(height,height,6))
    for i in range(6):
        msi[:,:,i] = msi[:,:,i].T

    hyp = np.zeros((height,height,9))
    hyp[:,:,0] = image[:,:,0]
    hyp[:,:,1] = image[:,:,1]
    hyp[:,:,2] = image[:,:,2]
    hyp[:,:,3] = msi[:,:,0]
    hyp[:,:,4] = msi[:,:,1]
    hyp[:,:,5] = msi[:,:,2]
    hyp[:,:,6] = msi[:,:,3]
    hyp[:,:,7] = msi[:,:,4]
    hyp[:,:,8] = msi[:,:,5]

    plt.imshow(msi[:,:,:])
    plt.show()

    plt.imshow(hyp[:,:,:])
    plt.show()