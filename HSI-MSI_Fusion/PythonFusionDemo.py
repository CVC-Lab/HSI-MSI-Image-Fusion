# Note you will have to correct the path save location of the starting data in the generation function before running the code
import matplotlib.pyplot as plt
import sys
import warnings

sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/denoise')
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/simulation')
sys.path.insert(0, '/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/HSI-MSI Fusion/codes/hyperspectral_image_processing/fusion')
warnings.filterwarnings('ignore')

from BGLRF_main import BGLRF_main
from Generation import generation
from denoising import denoising

hsi,msi = generation()
denoised_hsi,SNR_dB = denoising(hsi)
SRI_fused,K_est = BGLRF_main(denoised_hsi,msi,10,10,6)
plt.imshow(SRI_fused[:,:,10].T)
plt.show()