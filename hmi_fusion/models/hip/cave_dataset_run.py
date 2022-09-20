import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pdb

sys.path.insert(0, "/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/hmi_fusion/datasets")
# sys.path.insert(0, './codes/hyperspectral_image_processing/denoise')
# sys.path.insert(0, './codes/hyperspectral_image_processing/simulation')
# sys.path.insert(0, './codes/hyperspectral_image_processing/fusion')
# sys.path.insert(0, './codes/hyperspectral_image_processing/quality_metrics')
warnings.filterwarnings('ignore')

from cave_dataset import load_data
from hip.denoise.denoising import denoising
from hip.simulation.generate_hsi import generate_lrhsi
from hip.simulation.add_noise import add_noise
from hip.fusion.bglrf_main import BGLRF_main
from hip.quality_metrics.quality_assessment import quality_assessment

classes, class2id, data = load_data("/Users/pronomabanerjee/Dropbox/My Mac (Pronoma’s MacBook Air)/Desktop/UT Austin/HSI-MSI-Image-Fusion/hmi_fusion/datasets/data/CAVE", "test")

# denoised_hsi,SNR_dB = denoising(hsi)
# # SRI_fused,K_est = BGLRF_main(denoised_hsi,msi,10,10,6)

print(len(data))

for i in range(len(data)):
    print(data[i][0])
    print(data[i][1].shape)
    print(data[i][2].shape)

    

    print("Class: ", data[i][0])
    msi_img = data[i][1]
    rgb_img = data[i][2]

    msi_denoised, a = denoising(msi_img)

    # pdb.set_trace()
    hsi = generate_lrhsi(msi_img)
    rgb_denoised, a = denoising(rgb_img)
    msi, a = add_noise(rgb_denoised, 40)

    denoised_hsi, SNR_dB = denoising(hsi)
    SRI_fused, K_est = BGLRF_main(denoised_hsi, msi, 10, 10, 6)

    plt.imshow(SRI_fused[:,:,10])
    plt.show()

    quality_assessment(msi_denoised,SRI_fused,0,1/4)

    




