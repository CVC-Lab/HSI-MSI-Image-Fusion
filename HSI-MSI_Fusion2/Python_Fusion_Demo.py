import matplotlib.pyplot as plt
import sys
import warnings
import numpy as np

sys.path.insert(0, './codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, './codes/hyperspectral_image_processing/simulation')
sys.path.insert(0, './codes/hyperspectral_image_processing/fusion')
sys.path.insert(0, './codes/hyperspectral_image_processing/quality_metrics')
warnings.filterwarnings('ignore')

from BGLRF_main import BGLRF_main
from Generation import generation
from denoising import denoising
from quality_assessment import quality_assessment

SRI, hsi,msi = generation()
# denoised_hsi,SNR_dB = denoising(hsi)
# SRI_fused,K_est = BGLRF_main(denoised_hsi,msi,10,10,6)

# plt.imshow(SRI[:,:,10])
# plt.show()

print(SRI.shape)

# np.save("SRI_fused", SRI_fused)
SRI_fused = np.load("SRI_fused.npy")

# evaluate the fused result
# psnr,rmse,ergas,sam,uiqi,ssim,DD,CCS = quality_assessment(SRI,SRI_fused,0,1/4)
quality_assessment(SRI,SRI_fused,0,1/4)

# print("psnr", psnr)
# print("rmse", psnr)
# print("ergas", psnr)
# print("sam", psnr)
# print("uiqi", psnr)

# print(SRI_fused)
# plt.imshow(SRI_fused[:,:,10].T)
# plt.show()
