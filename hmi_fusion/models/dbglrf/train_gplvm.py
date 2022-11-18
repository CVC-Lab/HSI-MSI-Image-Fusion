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
        self.d_fc1_m = nn.Linear(latent_dim, input_channels)
        self.d_fc2_m = nn.Linear(input_channels, input_channels)
        self.d_fc3_m = nn.Linear(input_channels, input_channels)

        self.d_fc1_var = nn.Linear(latent_dim, input_channels)
        self.d_fc2_var = nn.Linear(input_channels, input_channels)
        self.d_fc3_var = nn.Linear(input_channels, input_channels)

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

    def decode(self, x):
        x_m = self.d_fc1_m(x)
        x_m = F.gelu(x_m)
        x_m = self.d_fc2_m(x_m)
        x_m = F.gelu(x_m)
        x_m = self.d_fc3_m(x_m)

        x_var = self.d_fc1_var(x)
        x_var = F.gelu(x_var)
        x_var = self.d_fc2_var(x_var)
        x_var = F.gelu(x_var)
        x_var = self.d_fc3_var(x_var)

        return x_m, x_var.exp()

    def forward(self, x):
        x_m, x_var = self.encode(x)
        z = dist.Normal(x_m, x_var)
        x_reco = self.decode(z)
        return x_reco, z


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

    def forward(self, Zq):
        Zq = self.scale_to_bounds(Zq)
        mean_zq = self.mean_module(Zq)
        covar_zq = self.covar_module(Zq)
        # pdb.set_trace()
        Zdist = MultivariateNormal(mean_zq, covar_zq)
        return Zdist
        

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


class SGPVAE(nn.Module):
    def __init__(self, vae, gplvm, ) -> None:
        super().__init__()
        self.vae = vae
        self.gplvm = gplvm
    
    def forward(self, X, k):
        Zqm, Zqv  = self.vae.encode(X)

        # pdb.set_trace()
        Zq_dist = dist.Normal(Zqm, Zqv)
        Zq = Zq_dist.rsample(torch.Size([k]))
        ## GP Prior
        Zdist = self.gplvm(Zq)
        # sample from dist and then decode
        Z = Zdist.rsample(torch.Size([k]))
        ## decode GP prediction
        S, latent_size, batch_size = Z.shape
        Z = Z.permute(0, 2, 1)
        Z = Z.reshape(S*batch_size, latent_size)
        Ym, Yvar = self.vae.decode(Z)
        Ydist = dist.Normal(Ym, Yvar)
        
        return Ydist, Zdist, Zq_dist, Zq, Z

    def calc_loss(self, Ydist, Zq_dist, Zq, Y):
        lp_zq = dist.Normal(torch.zeros_like(Zq), torch.ones_like(Zq)).log_prob(Zq).sum(-1)

        # pdb.set_trace()
        lq_zq_x = Zq_dist.log_prob(Zq).sum(-1)
        # lq_z_x = Zdist.log_prob(Z).sum(-1)
        # pdb.set_trace()
        lp_x_z = Ydist.log_prob(Y.T).sum(-1)
        lw = lp_x_z + lp_zq - lq_zq_x
        return -lw.sum(0).mean(0)



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
vae = VAE(31, latent_dim=20)
gplvm = bGPLVM(n=N, data_dim=vae_latent_dim, latent_dim=vae_latent_dim, n_inducing=n_inducing, pca=pca)
sgpvae = SGPVAE(vae, gplvm)
# Likelihood
likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)

if os.path.exists( "./artifacts/YGP.pt"):
    loaded = torch.load("./artifacts/YGP.pt")
    sgpvae.load_state_dict(loaded["model_state_dict"],   strict=False)
    likelihood.load_state_dict(loaded['likelihood'])
    print("loaded models!")

# vae.to(device)
# gplvm = gplvm.to(device)
sgpvae = sgpvae.to(device)
likelihood = likelihood.to(device)

# Declaring the objective to be optimised along with optimiser
# (see models/latent_variable.py for how the additional loss terms are accounted for)
mll = VariationalELBO(likelihood, gplvm, num_data=new_ds.shape[0])# total training points - 1000

optimizer = torch.optim.Adam([
    {'params': sgpvae.parameters()},
    {'params': likelihood.parameters()}
], lr=0.001)

loss_list = []
batch_size = 100
num_batches = len(new_ds) // batch_size
print("num_batches:", num_batches)
epochs = 10
iterator = range(epochs*num_batches if not smoke_test else 4)
total_loss = 0

sgpvae.train()
likelihood.train()
total_sgp_loss = 0
total_vae_loss = 0
for i in tqdm(iterator):
    batch_index = gplvm._get_batch_idx(batch_size)
    optimizer.zero_grad()
    # sample = gplvm.sample_latent_variable()  # a full sample returns latent x across all N
    # sample_batch = sample[batch_index]
    sample_batch = new_ds[batch_index].to(device)
    Ydist, Zdist, Zq_dist, Zq, Z = sgpvae(sample_batch, k=1)
    
    Y = new_ds[batch_index].T
    # normalize Y between -1 and 1
    Y = Y - Y.min(1).values[..., None] / Y.max(1).values[..., None]
    loss_vae = sgpvae.calc_loss(Ydist, Zq_dist, Zq, Y.to(device))
    
    loss_sgp = -mll(Zdist, Z.permute(1, 0)).sum() # SGP ELBO Loss
    
    loss = loss_vae + loss_sgp
    total_sgp_loss += loss_sgp.item()
    total_vae_loss += loss_vae.item()
    total_loss += loss.item()
    loss_list.append(loss.item())
    if (i+1) == 100:
        sgpvae.vae.eval()

    loss.backward()
    optimizer.step()
    if (i+1) % (50) == 0:
        print('Loss: ' + str(float(np.round(total_loss/(i+1),2))) + 
        ' sgp_loss ' + str(float(np.round(total_sgp_loss/(i+1),2))) +
        ' vae_loss ' + str(float(np.round(total_vae_loss/(i+1),2))) 
        + ", iter no: " + str((i)))
    if (i+1) % (num_batches) == 0:
        print('Loss: ' + str(float(np.round(total_loss/num_batches,2))) + ", epoch no: " + str((i) // num_batches))
        total_loss = 0    
        print("saving ...")
        torch.save(
            {
                'iter': i,
                'sgpvae_state_dict': sgpvae.state_dict(),
                'likelihood': likelihood.state_dict()
                }
                , "./artifacts/YGP.pt")