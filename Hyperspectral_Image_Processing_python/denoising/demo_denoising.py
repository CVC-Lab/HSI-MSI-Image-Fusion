from turtle import clear
from scipy.io import loadmat
import pandas as pd
import numpy as np
from PIL import Image as im
import sklearn
from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

annots = loadmat('../test_data/indian_pines.mat')

X = annots['indian_pines_corrected']
print(X.shape)

Y,SNR_dB= denoising.denoising(X)

plt.figure(figsize=(12,8))
plt.plot(SNR_dB)
plt.show()

im1 = im.fromarray(X[:,:,1])
im1.save('OriginalHSI_slice1.png')

im2 = im.fromarray(Y[:,:,1])
im2.save('DenoisedHSI_slice1.png')