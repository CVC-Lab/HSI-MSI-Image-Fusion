import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pdb

sys.path.insert(0, "/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/hmi_fusion/datasets")
sys.path.insert(0, './codes/hyperspectral_image_processing/quality_metrics')
warnings.filterwarnings('ignore')

from cave_dataset import load_data
from .denoise.denoising import denoising
# from .simulation.generation import generate_indian_pines_data
from .fusion.bglrf_main import BGLRF_main

classes, class2id, data = load_data("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/hmi_fusion/datasets/data/CAVE", "test")

# denoised_hsi,SNR_dB = denoising(hsi)
# # SRI_fused,K_est = BGLRF_main(denoised_hsi,msi,10,10,6)


for i in range(len(data)):
    print(data[i][0])
    print(data[i][1].shape)
    print(data[i][2].shape)

    pdb.set_trace()

    # msi_img = data[i][1]
    # rgb_img = data[i][2]

    # denoised_msi, SNR_dB = denoising(msi_img)
    # SRI_fused, K_est = BGLRF_main(denoised_msi, rgb_img, 10, 10, 6)

    # plt.imshow(SRI_fused[:,:,10])
    # plt.show()

    




