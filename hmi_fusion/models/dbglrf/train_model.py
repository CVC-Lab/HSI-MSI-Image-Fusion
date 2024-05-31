import sys
sys.path.append("../")

from torch.optim import Adam
import cv2
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
# from colour.plotting import *
import numpy as np
import matplotlib.pyplot as plt
from datasets.cave_dataset import CAVEDataset, R
from models.dbglrf.dbglrf import Downsampler, Downsampler2
import os
from models.metrics import (
    compare_mpsnr,
    compare_mssim,
    find_rmse,
    # compare_sam,
    compare_ergas
)
import pdb
from models.dbglrf.tv_layers_for_cv.tv_opt_layers.layers.general_tv_2d_layer import GeneralTV2DLayer

from torchmetrics import SpectralAngleMapper
from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS

sam = SpectralAngleMapper()
ergas = ERGAS(ratio=1/8)
batch_size = 4
train_dataset = CAVEDataset("../datasets/data/CAVE", None, mode="train")
test_dataset = CAVEDataset("../datasets/data/CAVE", None, mode="test")
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=False, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=False, batch_size=batch_size)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()
device = torch.device("cuda:0")
Rmodel = torch.nn.Linear(3, 31, bias=False)
# ds_model = Downsampler(4, 3)
ds_model = Downsampler2(3, 3)
optimizer = Adam([
    {"params":Rmodel.parameters()},
   { "params":ds_model.parameters()}
    ], lr=1e-3)
mse = torch.nn.MSELoss()
model_name = "ds2"
if not os.path.exists(f"../artifacts/{model_name}"):
    os.makedirs(f"../artifacts/{model_name}")
    Rmodel_path =f"../artifacts/R.pt"
    shutil.copy(Rmodel_path, f"../artifacts/{model_name}/R.pt")

Rmodel_path =f"../artifacts/{model_name}/R.pt"
ds_model_path = f"../artifacts/{model_name}/ds_model.pt"
if os.path.exists(Rmodel_path):
    Rmodel.load_state_dict(torch.load(Rmodel_path))
if os.path.exists(ds_model_path):
    ds_model.load_state_dict(torch.load(Rmodel_path), strict=False)

Rmodel = Rmodel.to(device)
ds_model = ds_model.to(device)   

def freeze_network(params):
    for param in params:
        param.require_grad = False

def unfreeze_network(params):
    for param in params:
        param.require_grad = True

def freeze_layer(layer):
    layer.require_grad = False
        
n_epochs =60
mode = "train_downsampler"
# best_mse_loss = baseline_total/len(dataset)
best_mse_loss = 0.008
for epoch in range(n_epochs):
    total_mse_loss = 0
    for items in train_loader:
        optimizer.zero_grad()
        c, x_old, y, z, x_gt, lz, yiq, Zd, idx = items
        Zd = Zd.to(device)
        yiq = yiq.to(device)
        y = y.to(device)
        B,_, N1, N2 = y.shape
        _,C, _, _ = z.shape
        if mode == "train_R":
            Rmodel.train()
            freeze_network(ds_model)
            Zd_enhanced = ds_model(Zd, yiq)
            y_pred = Rmodel(Zd_enhanced.permute(0, 2, 3, 1).reshape(B, -1, 3))
            # Zd = Zd.permute(1, 2, 0).reshape(-1, C)[None, ...].to(device)
            y_pred = Rmodel(Zd)
        if mode == "train_downsampler":
            freeze_layer(Rmodel)
            Zd_enhanced = ds_model(Zd, yiq)
            # pdb.set_trace()
            # y_pred = Rmodel(Zd_enhanced)
            y_pred = Rmodel(Zd_enhanced.permute(0, 2, 3, 1).reshape(B, -1, 3))
            # y_pred = y_pred.permute(0, 2, 1).reshape(B, y.shape[1], N1, N2)

        # y_pred = (newR @ Zd.reshape(C, -1)).reshape(*y.shape)
        y_pred = y_pred.transpose(1, 2).reshape(y.shape[0], y.shape[1], N1, N2)
        loss = mse(y_pred, y)
        loss.backward()
        optimizer.step()
        total_mse_loss += loss.item()

    print(f"epoch {epoch} mse: {total_mse_loss/len(train_dataset)}")
    if total_mse_loss/len(train_dataset) < best_mse_loss:
        best_mse_loss =  total_mse_loss/len(train_dataset)
        print("saving ...")
        if mode == "train_downsampler":
            torch.save(ds_model.state_dict(), ds_model_path)
        if mode == "train_R":
            torch.save(Rmodel.state_dict(),ds_model_path)
    

torch.save(Rmodel.state_dict(), Rmodel_path)
torch.save(ds_model.state_dict(), ds_model_path)
unfreeze_network(ds_model)
ds_model.eval()
newR = Rmodel.weight
test_dataset = CAVEDataset("../datasets/data/CAVE", None, mode="test")

total_psnr, total_ssim, total_rmse, total_sam, total_ergas =0,0,0,0,0
with torch.no_grad():
    for items in test_dataset:
        c, x_old, y, z, x_gt, lz, yiq, Zd, idx = items

        zc, N1, N2 = z.shape 
        # Zd_enhanced = ds_model(Zd, yiq)
        x2 = Rmodel(z.permute(0, 2, 3, 1).reshape(B, -1, 3))
        # x2 = newR(z_)
        # x2 = (newR @ z_out.reshape(zc, -1)).reshape(y.shape[0], N1, N2)
    
        x = x.squeeze()
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