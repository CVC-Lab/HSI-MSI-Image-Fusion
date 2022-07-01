from turtle import clear
from scipy.io import loadmat
import pandas as pd
import numpy as np
from PIL import Image as im
#import sklearn
#from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import UnivariateSpline
import numpy as np
import warnings
import sys

clear

sys.path.insert(0, '../simulation')
sys.path.insert(0, '../denoising')
sys.path.insert(0, '../fusion')
sys.path.insert(0, '../band_selection')
sys.path.insert(0, '../quality_metrics')
sys.path.insert(0, '../svm')
warnings.filterwarnings('ignore')

import denoising
import MNBS
#import demo_bs
import add_noise



def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


annots = loadmat('../test_data/indian_pines.mat')

SRI_prev = annots['indian_pines_corrected']

SRI,_ = denoising.denoising(SRI_prev)

N1 = 144 
N2 = 144
N3 = SRI.shape[2]

SRI = SRI[0:N1,0:N2,:]
SRI_prev = SRI_prev[0:N1,0:N2,:]

n3 = SRI.shape[2]
n2 = SRI.shape[1]
n1 = SRI.shape[0]

X = np.reshape(SRI_prev,(n1*n2,n3))
Y = np.reshape(SRI,(n1*n2,n3))

#get the wavelength of each band
# for i in range(400,(2500-400)): arr1.append(i) 
# for i in range((220-1),2500): arr2.append(i) 

wave = np.linspace(400,2500,num=220)
ind = MNBS.MNBS(X,X-Y,16);
wave = wave[ind];

# generate MSI using spectral responses from the sensor of Landsat_TM5 
# satellite, to simulate another sensor, need to get its spectral responses

annots2 = loadmat('../simulation/Landsat_TM5.mat')
#S = {annots2['blue'], annots2['green'], annots2['red'], annots2['nir'], annots2['swir1'], annots2['swir2']}

S = []
S.append(annots2['blue'])
S.append(annots2['green'])
S.append(annots2['red'])
S.append(annots2['nir'])
S.append(annots2['swir1'])
S.append(annots2['swir2'])

n3 = 6
P3 = np.zeros((N3,n3))

for i in range(n3):
    
    s = S[i];
    
    temp = s[:,0]
    TEMP = s[:,1]
  
    IND1 = np.argwhere(wave>temp[0]);
    ind1 = IND1[0];
    IND2 = np.argwhere(wave<temp[-1]);
    #ind2 = IND2[-1]
    print(IND1)
    print(ind1)
    print(IND2)
    if IND2.shape[0]==0:
        spline = UnivariateSpline(temp,TEMP, wave[ind1])
    else:
        wl = []
        ind2 = IND2[-1]
        for j in range (ind1,ind2):
            wl.append(wave[j])

        spline = UnivariateSpline(temp,TEMP, wl)
    
    yy = spline(temp,TEMP);
    
    P3[ind1:ind2,i] = yy;
    P3[:,i] = P3[:,i]/sum(P3[:,i]);
    
    SRI = np.reshape(SRI,(N1*N2,N3));
    MSI = SRI*P3;
    MSI = np.reshape(MSI,(N1,N2,n3));
    SRI = np.reshape(SRI,(N1,N2,N3));

    # add noise to the MSI
    MSI,noise_tenM = add_noise.add_noise(MSI,40); # noisy MSI of SNR about 40

    # generate HSI of downsampling ratio 4, with periodic boundary condition
    ratio = 4;
    #K0 = fspecial('gaussian',[9 9],ratio/(2.355)); # 9 = 2*4+1
    K0 = fspecial_gaussian((9,9), ratio/(2.355))

    # blur kernel shifted 2 pixels vertically and horizontally to the top left
    radius = 6;
    K = np.zeros((2*radius+1,2*radius+1));
    K[0:8,0:8] = K0;

    center = np.zeros((1,2));
    center[0] = N1/2+1;
    center[1] = N2/2+1;

    I = np.linspace((center[0]-radius),(center[1]+radius));
    I = I.T;
    I = np.repmat(I,(1,2*radius+1));
    I = I[:];
    J = np.linspace((center[1]-radius),(center[1]+radius));
    J = np.repmat(J,(2*radius+1,1));
    J = J[:];
    ind = sub2ind([N1,N2],I,J); # indices for blur kernel

    KK = np.zeros((N1,N2));
    KK[ind] = K;
    Fk = np.fft.fft2(np.roll(KK,1-center));

    n1 = 36;
    n2 = 36;
    HSI = np.zeros((n1,n2,N3));

    for band in range(N3):
        
        x = SRI[:,:,band]; 
        Fx = np.fft.fft2(x);
        x = np.real(np.fft.ifft2(Fk @ Fx));
        end = x.shape[0]-1
        HSI[:,:,band] = x[1:ratio:end,1:ratio:end];

    # add noise to the HSI
    HSI,noise_tenH = add_noise.add_noise(HSI,30); # noisy HSI of SNR about 35
