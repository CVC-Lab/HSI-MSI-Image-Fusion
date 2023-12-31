import sys
sys.path.append("../")

from torch.optim import Adam
import cv2
import torch
import torch.nn.functional as F
# from colour.plotting import *
import numpy as np
import matplotlib.pyplot as plt
from datasets.cave_dataset import CAVEDataset, R
from .motion_code import MotionCode
from .create_point_level_dataset import prepare_point_ds
# from datasets.cave_dataset import HarvardDataset, R
import os
import pdb
from models.metrics import (
    compare_mpsnr,
    compare_mssim,
    find_rmse,
    # compare_sam,
    compare_ergas
)
from torchmetrics import SpectralAngleMapper
from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS
from einops import rearrange, reduce, repeat
sam = SpectralAngleMapper()
ergas = ERGAS(ratio=1/8)
import mogptk
torch.manual_seed(1)

data_path = "./datasets/data/CAVE"
dataset = CAVEDataset(data_path, None, mode="train")
test_dataset = CAVEDataset(data_path, None, mode="test")
train_ds = prepare_point_ds(dataset=dataset)
test_ds = prepare_point_ds(dataset=test_dataset)

pdb.set_trace()

model_path="./artifacts/mosm_R.pt"


n_epochs =100



pdb.set_trace()

# for epoch in range(n_epochs):
#     total_loss = 0
#     total_mse_loss = 0
#     total_sam_loss = 0
#     for items in dataset:
#         # c, x_k, lr_hsi, hr_msi, hr_hsi, to_torch_sparse(lz.tocoo()), yiq_downsampled, Zd, idx
#         y, z, x_gt, _  = items
#         pdb.set_trace()
#         # z is you X_train, x_gt is Y_train
#         X_train = []

        

        



# model.eval()
# newR = model.weight
# test_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="test")

# total_psnr, total_ssim, total_rmse, total_sam, total_ergas =0,0,0,0,0
# with torch.no_grad():
#     for items in test_dataset:
#         y, z, x, _  = items
#         pdb.set_trace()
#         zc, N1, N2 = z.shape 
#         x2 = (newR @ z.reshape(zc, -1)).reshape(y.shape[0], N1, N2)
    
#         # x = x.squeeze()
#         x2 = x2.squeeze()
#         x = x.permute(1, 2, 0).detach().cpu().numpy()
#         x2 = x2.permute(1, 2, 0).detach().cpu().numpy()
        
#         total_ssim += compare_mssim(x, x2)
#         rmse,  mse, rmse_per_band = find_rmse(x, x2)
#         total_rmse += rmse
#         total_psnr += compare_mpsnr(x, x2, mse)
#         total_sam += torch.nan_to_num(sam(torch.from_numpy(x).permute(2, 0, 1)[None, ...], 
#                                 torch.from_numpy(x2).permute(2, 0, 1)[None, ...]), nan=0, posinf=1.0) * (180/torch.pi)
#         total_ergas += compare_ergas(x, x2, 8, rmse_per_band)
#         # total_sam += compare_sam(x, x2)
#         # total_ergas += ergas(torch.from_numpy(x).permute(2, 0, 1)[None, ...], 
#         #                     torch.from_numpy(x2).permute(2, 0, 1)[None, ...])
#         # total_ergas += compare_ergas(x, x2,1/sf, rmse_per_band)

# opt = f"""## Metric scores:
# psnr:{total_psnr/len(test_dataset)},
# ssim:{total_ssim/len(test_dataset)},
# rmse:{total_rmse/len(test_dataset)},
# sam:{total_sam/len(test_dataset)},
# ergas:{total_ergas/len(test_dataset)},
# """
# print(opt)