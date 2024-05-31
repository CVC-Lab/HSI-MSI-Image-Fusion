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
from einops import rearrange, pack, unpack
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
sam = SpectralAngleMapper()
ergas = ERGAS(ratio=1/8)

# dataset = CAVEDataset("../datasets/data/CAVE", None, mode="train")
dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
model = torch.nn.Linear(4, 31, bias=False)
optimizer = Adam(model.parameters(), lr=1e-3)
model.train()
mse = torch.nn.MSELoss()

model_path = "./artifacts/R_new.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

# baseline_total = 0
# # lets check mse loss for R provided by other vendors
# for items in dataset:
#     c, x_old, y, z, x_gt, lz, idx = items
#     z_ipt = z.numpy()
#     Zd = torch.zeros(z.shape[0], y.shape[1], y.shape[2])
#     C, N1, N2 = Zd.shape
#     for c in range(Zd.shape[0]):
#         Zd[c, :, :] = torch.FloatTensor(cv2.resize(z_ipt[c, :, :], (N1, N2), interpolation=cv2.INTER_CUBIC))
    
#     y_pred = (R.T @ Zd.reshape(C, -1)).reshape(y.shape[0], N1, N2)
#     loss = mse(y_pred, y)
#     baseline_total += loss.item()

# print(baseline_total/len(dataset))

n_epochs =100

# model.train()
# # best_mse_loss = baseline_total/len(dataset)
best_mse_loss = 0.002
# best_mse_los
for epoch in range(n_epochs):
    total_loss = 0
    total_mse_loss = 0
    total_sam_loss = 0
    for items in dataset:
        optimizer.zero_grad()
        # c, x_k, lr_hsi, hr_msi, hr_hsi, to_torch_sparse(lz.tocoo()), yiq_downsampled, Zd, idx
        y, z, x_gt, seg_map, max_vals  = items
        
        seg_map = rearrange(seg_map, "h w c -> c h w")
        z_ipt, _  = pack([seg_map, z], "* h w")
        # z_ipt = z_ipt.numpy()
        Zd = torch.zeros(z.shape[0]+1, y.shape[1], y.shape[2])
        C, N1, N2 = Zd.shape
        for c in range(Zd.shape[0]):
            Zd[c, :, :] = torch.FloatTensor(cv2.resize(z_ipt[c, :, :], (N1, N2), interpolation=cv2.INTER_CUBIC))
        
        Zd = Zd.permute(1, 2, 0).reshape(-1, C)[None, ...]
        
        y_pred = model(Zd)
        # y_pred = (newR @ Zd.reshape(C, -1)).reshape(*y.shape)
        # pdb.set_trace()
        y_pred = y_pred.transpose(1, 2).reshape(1, y.shape[0], N1, N2)
        # sam_loss = torch.nan_to_num(sam(y_pred, y[None, ...]), nan=0.0, posinf=1.0)
        sam_loss = ergas(y_pred, y[None, ...])**2
        mse_loss = mse(y_pred, y[None, ...])
        loss = mse_loss + sam_loss
        loss.backward()
        optimizer.step()
        total_mse_loss += mse_loss.item()
        total_sam_loss += sam_loss.item()
        total_loss += loss.item()

    print(f"epoch {epoch} loss: {total_loss/len(dataset)}, ergas^2: {total_sam_loss/len(dataset)}, mse: {total_mse_loss/len(dataset)}")
    if total_mse_loss/len(dataset) < best_mse_loss:
        best_mse_loss =  total_mse_loss/len(dataset)
        print("saving ...")
        torch.save(model.state_dict(), "../artifacts/R_new_one.pt")

torch.save(model.state_dict(), "./artifacts/R_new_one.pt")


model.eval()
newR = model.weight
test_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="test")

total_psnr, total_ssim, total_rmse, total_sam, total_ergas =0,0,0,0,0
with torch.no_grad():
    for items in test_dataset:
        y, z, x, _  = items
        pdb.set_trace()
        zc, N1, N2 = z.shape 
        x2 = (newR @ z.reshape(zc, -1)).reshape(y.shape[0], N1, N2)
    
        # x = x.squeeze()
        x2 = x2.squeeze()
        x = x.permute(1, 2, 0).detach().cpu().numpy()
        x2 = x2.permute(1, 2, 0).detach().cpu().numpy()
        
        total_ssim += compare_mssim(x, x2)
        rmse,  mse, rmse_per_band = find_rmse(x, x2)
        total_rmse += rmse
        total_psnr += compare_mpsnr(x, x2, mse)
        total_sam += torch.nan_to_num(sam(torch.from_numpy(x).permute(2, 0, 1)[None, ...], 
                                torch.from_numpy(x2).permute(2, 0, 1)[None, ...]), nan=0, posinf=1.0) * (180/torch.pi)
        total_ergas += compare_ergas(x, x2, 8, rmse_per_band)
        # total_sam += compare_sam(x, x2)
        # total_ergas += ergas(torch.from_numpy(x).permute(2, 0, 1)[None, ...], 
        #                     torch.from_numpy(x2).permute(2, 0, 1)[None, ...])
        # total_ergas += compare_ergas(x, x2,1/sf, rmse_per_band)

opt = f"""## Metric scores:
psnr:{total_psnr/len(test_dataset)},
ssim:{total_ssim/len(test_dataset)},
rmse:{total_rmse/len(test_dataset)},
sam:{total_sam/len(test_dataset)},
ergas:{total_ergas/len(test_dataset)},
"""
print(opt)