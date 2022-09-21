import numpy as np
import warnings
warnings.filterwarnings('ignore')

from ..denoise.denoising import denoising
from .add_noise import add_noise
from .MSG import matlab_style_gauss2D

def generate_lrhsi(sri):

    sri,a = denoising(sri)

    N1 = sri.shape[0]
    N2 = sri.shape[1]
    N3 = sri.shape[2]

    n1 = int(N1/4)
    n2 = int(N2/4)
    n3 = 3

    # Shifted blur kernel
    k0 = matlab_style_gauss2D()
    radius = 6
    k = np.zeros((13,13))
    k[0:9,0:9] = k0
    
    center = np.zeros((2,1))
    center[0]=N1/2+1
    center[1]=N2/2+1
    
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
    inds = np.ravel_multi_index(arr,(N1-1,N2-1))
    a=0
    kk = np.zeros((N1,N2))
    for i in range(13):
        for j in range(13):
            
            kk[I2[a],J[a]] = k[j,i]
            a=a+1
        
    #Shift Kernel
    kk = np.roll(kk,-int((N1+2)/2), axis=0)
    kk = np.roll(kk,-int((N2+2)/2), axis=1)
    fft = np.fft.fft2(kk)
    
    hsi = np.zeros((n1,n2,N3))
    
    for band in range(N3):
        x = sri[:,:,band]
        Fx = np.fft.fft2(x)
        x = np.multiply(Fx,fft)
        x = np.real(np.fft.ifft2(x))
        hsi[:,:,band] = x[1:-1:4,1:-1:4]
    
    
    hsi,noise_ten2 = add_noise(hsi,30)

    return hsi