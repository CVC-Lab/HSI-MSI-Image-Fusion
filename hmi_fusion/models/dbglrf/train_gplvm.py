# gpytorch imports
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt

# from tqdm.notebook import trange
from tqdm import tqdm
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.grid import ScaleToBounds
from datasets.cave_dataset import CAVEDataset
import torch.nn.functional as F
import torch.distributions as dist
import torch.nn as nn
import numpy as np
import os
import pdb


def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))


class VAE(nn.Module):
    def __init__(self, input_channels=31, latent_dim=20) -> None:
        super().__init__()
        self.e_fc1_mean = nn.Linear(input_channels, input_channels)
        self.e_fc2_mean = nn.Linear(input_channels, latent_dim)
        self.e_fc1_var = nn.Linear(input_channels, input_channels)
        self.e_fc2_var = nn.Linear(input_channels, latent_dim)
        
        # we can use kumarswamy, currently using simple gaussian
        self.d_fc1 = nn.Linear(latent_dim, input_channels)
        self.d_fc2 = nn.Linear(input_channels, input_channels)
        self.d_fc3 = nn.Linear(input_channels, input_channels)

    def encode(self, x):
        x_m =  self.e_fc1_mean(x)
        x_m = F.gelu(x_m)
        x_m = self.e_fc2_mean(x_m)
        x_m = F.gelu(x_m)

        x_var =  self.e_fc1_var(x)
        x_var = F.gelu(x_var)
        x_var = self.e_fc2_var(x_var)
        x_var = F.gelu(x_var)
        return x_m, x_var.exp()

    def decoder(self, x):
        x = self.d_fc1(x)
        x = F.gelu(x)
        x = self.d_fc2(x)
        x = F.gelu(x)
        x = self.d_fc3(x)
        return x

    def forward(self, x):
        x_m, x_var = self.encode(x)
        z = dist.Normal(x_m, x_var)
        x_reco = self.decode(z)
        return x_reco, z

    # def elbo_loss(self, x, x_reco, x_m, x_var):
    #     mse_loss = ((x -  x_reco)**2).sum(-1)




class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        params = torch.normal(0, 1, (n, latent_dim))
        X_init = torch.nn.Parameter(params)

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        self.scale_to_bounds = ScaleToBounds(-1., 1.)

    def forward(self, X):
        X = self.scale_to_bounds(X)
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


device = torch.device("cuda:0")
dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
# lets create simple dataset for y
new_ds = torch.cat([data[0].reshape(31, -1) for data in dataset], -1).permute(1, 0) # num_samples, 31

smoke_test = False
N = new_ds.shape[0]
vae_data_dim = 31 #31  # 12
vae_latent_dim = 20
n_inducing = 2048
pca = False



# Model
gplvm = bGPLVM(N, data_dim=vae_latent_dim, latent_dim=vae_latent_dim, n_inducing=n_inducing, pca=pca)

vae = VAE(31, latent_dim=20, gplvm=gplvm)
# Likelihood
likelihood = GaussianLikelihood(batch_shape=model.batch_shape)

if os.path.exists( "./artifacts/YGP.pt"):
    loaded = torch.load("./artifacts/YGP.pt")
    model.load_state_dict(loaded["model_state_dict"])
    likelihood.load_state_dict(loaded['likelihood'])
    print("loaded models!")

model = model.to(device)
likelihood = likelihood.to(device)

# Declaring the objective to be optimised along with optimiser
# (see models/latent_variable.py for how the additional loss terms are accounted for)
mll = VariationalELBO(likelihood, model, num_data=new_ds.shape[0])# total training points - 1000

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
], lr=0.001)

loss_list = []
batch_size = 100
num_batches = len(new_ds) // batch_size
print("num_batches:", num_batches)
epochs = 10
iterator = range(epochs*num_batches if not smoke_test else 4)
total_loss = 0
model.train()
likelihood.train()

for i in tqdm(iterator):
    batch_index = model._get_batch_idx(batch_size)
    optimizer.zero_grad()
    sample = model.sample_latent_variable()  # a full sample returns latent x across all N
    sample_batch = sample[batch_index]
    output_batch = model(sample_batch)
    target = new_ds[batch_index].T
    # normalize target between -1 and 1
    target = target - target.min(1).values[..., None] / target.max(1).values[..., None]
    loss = -mll(output_batch, target.to(device)).sum()
    total_loss += loss.item()
    loss_list.append(loss.item())

    loss.backward()
    optimizer.step()
    if (i+1) % (num_batches) == 0:
        print('Loss: ' + str(float(np.round(total_loss/num_batches,2))) + ", epoch no: " + str((i) // num_batches))
        total_loss = 0    
        print("saving ...")
        torch.save(
            {
                'iter': i,
                'model_state_dict': model.state_dict(),
                'likelihood': likelihood.state_dict()
                }
                , "./artifacts/YGP.pt")