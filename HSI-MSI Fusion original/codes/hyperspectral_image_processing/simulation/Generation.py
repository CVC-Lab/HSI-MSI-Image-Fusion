import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys
import warnings
from scipy.interpolate import CubicSpline
import cv2
import scipy.io

sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/simulation')
warnings.filterwarnings('ignore')

from denoising import denoising
from add_noise import add_noise
from MSG import matlab_style_gauss2D

def generation():
    filename ='/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Downloads/HSI-MSI-Image-Fusion-main/HSI-MSI Fusion/codes/hyperspectral_image_processing/test_data/indian_pines.mat'
    mat = scipy.io.loadmat(filename)
    bands_removed = mat["bands_removed"]
    indian_pines_c = mat["indian_pines_corrected"]
    scaling = mat["scaling"]
    
    sri = indian_pines_c/scaling
    sri,a = denoising(sri)
    sri = sri[0:144,0:144,:]
    mat2 = scipy.io.loadmat("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Downloads/HSI-MSI-Image-Fusion-main/HSI-MSI Fusion/codes/hyperspectral_image_processing/simulation/Landsat_TM5.mat")
    
    S1 = mat2["blue"]
    S2 = mat2["green"]
    S3 = mat2["red"]
    S4 = mat2["nir"]
    S5 = mat2["swir1"]
    S6 = mat2["swir2"]
    
    S = [S1,S2,S3,S4,S5,S6]
    
    N1 = 144
    N2 = 144
    N3 = 200
    n3 = 6
    P3 = np.zeros((N3,n3))
    
    wave = np.linspace(400,2500, num=220)
    wave = np.delete(wave,bands_removed[:]-1)
    
    for i in range(n3):
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
        
    sri2 = np.zeros((N1*N2,N3))
    
    for i in range(N3):
        sri2[:,i]= np.reshape(sri[:,:,i].T,(N1*N2))
    msi = np.matmul(sri2,P3)
    msi = np.reshape(msi,(N1,N2,n3))
    for i in range(n3):
        msi[:,:,i] = msi[:,:,i].T
    
    msi,noise_ten1 = add_noise(msi,40)
    
    # Shifted blur kernel
    k0 = matlab_style_gauss2D()
    radius = 6
    k = np.zeros((13,13))
    k[0:9,0:9] = k0
    
    center = np.zeros((2,1))
    center[0]=144/2+1
    center[1]=144/2+1
    
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
    inds = np.ravel_multi_index(arr,(143,143))
    a=0
    kk = np.zeros((144,144))
    for i in range(13):
        for j in range(13):
            
            kk[I2[a],J[a]] = k[j,i]
            a=a+1
        
    #Shift Kernel
    kk = np.roll(kk,-73, axis=0)
    kk = np.roll(kk,-73, axis=1)
    fft = np.fft.fft2(kk)
    
    n1 = 36
    n2 = 36
    hsi = np.zeros((n1,n2,N3))
    
    for band in range(N3):
        x = sri[:,:,band]
        Fx = np.fft.fft2(x)
        x = np.multiply(Fx,fft)
        x = np.real(np.fft.ifft2(x))
        hsi[:,:,band] = x[1:-1:4,1:-1:4]
    
    
    hsi,noise_ten2 = add_noise(hsi,35)
    return hsi,msi