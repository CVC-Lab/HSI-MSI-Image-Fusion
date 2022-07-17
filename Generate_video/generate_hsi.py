import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import sys
import warnings

sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/simulation')
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
    kk = np.roll(kk,-73, axis=0)
    kk = np.roll(kk,-73, axis=1)
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
    
    hsi,noise_ten2 = add_noise(hsi,30)
    hsi = np.round(hsi*255)
    hsi = hsi.astype(np.uint8)
    #print(type(hsi))
    return hsi


