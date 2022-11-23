import os
import cv2
import torch
import gpytorch
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.priors import NormalPrior
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.distributions import MultivariateNormal
from datasets.cave_dataset import CAVEDataset
from gpytorch.means import ConstantMean, LinearMean, ZeroMean

import gpytorch.settings as settings
from models.dbglrf.dbglrf import MultitaskVariationalGPModel
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import numpy as np
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

smoke_test = True
batch_size = 100                 # Size of minibatch
milestones = [20, 150, 300]       # Epochs at which we will lower the learning rate by a factor of 0.1
num_inducing_pts = 300            # Number of inducing points in each hidden layer
num_epochs = 400                  # Number of epochs to train for
initial_lr = 0.01                 # Initial learning rate
hidden_dim = 3                    # Number of GPs (i.e., the width) in the hidden layer.
output_dim = 31
num_quadrature_sites = 8          # Number of quadrature sites (see paper for a description of this. 5-10 generally works well).
device = torch.device("cuda:0")
## Modified settings for smoke test purposes
num_epochs = num_epochs if not smoke_test else 5

dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
## create train_x [pixel_ij, 3], train_y [pixel_ij, 31]
new_ds = torch.cat([data[0].reshape(31, -1) for data in dataset], -1).permute(1, 0) # num_samples, 31

train_x = []
train_y = []
for data in dataset:
    y, z, x, _ = data
    # downsample z
    z_ipt = z.numpy()
    Zd = torch.zeros(z.shape[0], y.shape[1], y.shape[2])
    C, N1, N2 = Zd.shape
    for c in range(Zd.shape[0]):
        Zd[c, :, :] = torch.FloatTensor(cv2.resize(z_ipt[c, :, :], (N1, N2), interpolation=cv2.INTER_CUBIC))
    # Zd = Zd.permute(1, 2, 0).reshape(-1, C)[None, ...]
    
    train_y.append(y.reshape(31, -1))
    train_x.append(Zd.reshape(C, -1))


train_x = torch.cat(train_x, -1).permute(1, 0)#.cuda()
train_y = torch.cat(train_y, -1).permute(1, 0)#.cuda()

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



    
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=31)
model = MultitaskVariationalGPModel(num_inducing_points=500)

model.to(device)
likelihood.to(device)
model.train()

model_path = "./artifacts/sgp.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))



adam = torch.optim.Adam([{'params': model.parameters(), 
                            "params": likelihood.parameters()}], lr=initial_lr, betas=(0.9, 0.999))
sched = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=milestones, gamma=0.1)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_dataset))


training_iterations = 2 if smoke_test else 50


def train():
    # Find optimal model hyperparameters
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        for x, y in train_loader:
            adam.zero_grad()
            output = model(x.to(device))
            loss = -mll(output, y.to(device))
            loss.backward()
            adam.step()
            sched.step()
            total_train_loss += loss.item()
        print(f"epoch: {epoch} train loss: {total_train_loss/len(train_loader)}")
            # if epoch % 5  == 0:
            #     print(f"train loss: {total_train_loss/len(train_loader)}")


train()
torch.save(model.state_dict(), model_path)

pdb.set_trace()

model.eval()
test_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="test")
test_x = []
test_y = []
for data in test_dataset:
    y, z, x, _ = data
    test_x.append(z[z.shape[0], -1])
    test_y.append(x.reshape(x.shape[0], -1))



test_x = torch.cat(test_x, -1).permute(1, 0)
test_y = torch.cat(test_y, -1).permute(1, 0)
test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
means, vars, ll = model.predict(test_loader)
weights = model.quad_weights.unsqueeze(-1).exp().cpu()
# `means` currently contains the predictive output from each Gaussian in the mixture.
# To get the total mean output, we take a weighted sum of these means over the quadrature weights.
rmse = ((weights * means).sum(0) - test_y.cpu()).pow(2.0).mean().sqrt().item()
ll = ll.mean().item()

total_psnr, total_ssim, total_rmse, total_sam, total_ergas =0,0,0,0,0
with torch.no_grad():
    for items in test_dataset:
        y, z, x  = items
        zc, N1, N2 = z.shape

        z_reshaped = z.reshape(zc, -1)
        x2 = model(z_reshaped).reshape(y.shape[0], N1, N2) 
        # x2 = (newR @ z.reshape(zc, -1)).reshape(y.shape[0], N1, N2)
    
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
    

opt = f"""## Metric scores:
psnr:{total_psnr/len(test_dataset)},
ssim:{total_ssim/len(test_dataset)},
rmse:{total_rmse/len(test_dataset)},
sam:{total_sam/len(test_dataset)},
ergas:{total_ergas/len(test_dataset)},
"""
print(opt)