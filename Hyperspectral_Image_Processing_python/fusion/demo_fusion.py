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
import demo_sim        # generate simulated HSI and MSI
import BGLRF_main
import quality_assessment

SRI = demo_sim.SRI
K = demo_sim.K
HSI = demo_sim.HSI
MSI = demo_sim.MSI

HSI_denoised = denoising.denoising(HSI)

# Assume we know the true blur kernel is of size no more than 13*13, radius
# is than chosen to be 6
SRI_fused,K_est,_ = BGLRF_main(HSI_denoised,MSI,10,10,6);

# evaluate the fused result
psnr,rmse,ergas,sam,uiqi,ssim,DD,CCS = quality_assessment(SRI,SRI_fused,0,1/4);

