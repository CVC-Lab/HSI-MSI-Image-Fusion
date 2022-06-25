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
import MNBS

annots = loadmat('../test_data/indian_pines.mat')

X = annots['indian_pines_corrected']
print(X.shape)

Y,_ = denoising.denoising(X)

n3 = X.shape[2]
n2 = X.shape[1]
n1 = X.shape[0]

X = np.reshape(X,(n1*n2,n3))
Y = np.reshape(Y,(n1*n2,n3))

# select 16 most informative bands
IND = MNBS.MNBS(X,X-Y,16) # get indices for the selected bands 

Z = Y[:,IND] #denosied HSI with selected bands for further processing