import os
import torch
import tqdm
import math
import yaml
import gpytorch
import numpy as np
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt
from datasets.cave_dataset import CAVEDataset, R
from .mosm_models import MultitaskDeepGP
from .create_point_level_dataset import prepare_point_ds
from einops import pack, unpack
from torchmetrics import SpectralAngleMapper
from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS
from itertools import islice
import pdb

sam = SpectralAngleMapper()
ergas = ERGAS(ratio=1/8)
# Load parameters from config.yaml
with open('models/mosm/dssp_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access parameters
gpu_idx = config['gpu_idx']
num_tasks = config['num_tasks']
smoke = config['smoke']
data_path = config['data_path']
num_epochs = config['num_epochs']
save_path = config['save_path']
num_inducing = config['num_inducing']
num_hidden_dgp_dims = config['num_hidden_dgp_dims']
batch_size = config['batch_size']
num_batches_to_train = config['num_batches_to_train']
initial_lr = 0.1
milestones = [i*num_epochs//3 for i in range(3)]
torch.cuda.set_device(gpu_idx)
print(f"training on cuda: {gpu_idx}")
dataset = CAVEDataset(data_path, None, mode="train")
test_dataset = CAVEDataset(data_path, None, mode="test")
train_x, train_y = prepare_point_ds(dataset=dataset)
test_x, test_y = prepare_point_ds(dataset=test_dataset)
# print(f'total train batches: {train_x.shape[0]/batch_size}')
print(f"num_batches_to_train: {num_batches_to_train}")
train_ds = TensorDataset(train_x, train_y)
test_ds = TensorDataset(test_x, test_y)


train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)
num_tasks = train_y.size(-1)
num_data = num_batches_to_train * batch_size


def predict(test_ds):
    with torch.no_grad():

        # The output of the model is a multitask MVN, where both the data points
        # and the tasks are jointly distributed
        # To compute the marginal predictive NLL of each data point,
        # we will call `to_data_independent_dist`,
        # which removes the data cross-covariance terms from the distribution.
        means, vars = [], []
        lowers, uppers = [], []
        for tx, ty in tqdm.tqdm(test_ds):
        
            # preds = model.likelihood(model(tx)).to_data_independent_dist()
            preds = model.likelihood(model(tx.cuda()))
            lower, upper = preds.confidence_region()
            mean = preds.mean.mean(0)
            lower = lower.mean(0)
            upper = upper.mean(0)
            # mean, var = preds.mean.mean(0), preds.variance.mean(0)
            means.append(mean.detach().cpu())
            lowers.append(lower.detach().cpu())
            uppers.append(upper.detach().cpu())

    means, _ = pack(means, '* d')
    lowers, _ = pack(lowers, '* d')
    uppers, _ = pack(uppers, '* d')
    return means, lowers, uppers


model = MultitaskDeepGP(train_x.shape, 
                        num_inducing=num_inducing,
                        num_hidden_dgp_dims=num_hidden_dgp_dims, 
                        num_tasks=num_tasks)

model = model.cuda()
model.train()

optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=initial_lr, betas=(0.9, 0.999))
sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=num_data))

num_epochs = 1 if smoke else num_epochs
epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

print('total batches per epoch: ', len(train_loader))

best_loss = np.inf
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    # optimizer.zero_grad()
    
    train_loss = 0
    for i, (tx, ty) in enumerate(islice(train_loader, num_batches_to_train)):
        # variational_ngd_optimizer.zero_grad()
        # hyperparameter_optimizer.zero_grad()
        optimizer.zero_grad()
        output = model(tx.cuda())
        pdb.set_trace()
        # sam_loss = ergas(y_pred, y[None, ...])**2
        # mse_loss = mse(y_pred, y[None, ...])
        loss = -mll(output, ty.cuda())
        train_loss += loss.item()
        epochs_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        # variational_ngd_optimizer.step()
        # hyperparameter_optimizer.step()
        # print(f"Batch {i + 1}/{num_batches_to_train} processed")
        # Break the loop if you have reached the desired number of batches
        if i + 1 == num_batches_to_train:
            break
        
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), save_path)


# Make predictions
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mean, lower, upper = predict(test_loader)
    # lower = mean - 2 * var.sqrt()
    # upper = mean + 2 * var.sqrt()

# Initialize plots
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
num_pnts = 5000
for task, ax in enumerate(axs):
    # Plot training data as black stars
    for dim in range(train_x.shape[1]):
        obs, = ax.plot(train_x[:num_pnts, dim].detach().numpy(), 
                       train_y[:num_pnts, task].detach().numpy(), 'k*')

    # Predictive mean as blue line
    for dim in range(test_x.shape[1]):
        m, = ax.plot(test_x[:num_pnts, dim].numpy(), 
                     mean[:num_pnts, task].numpy(), 'b')

    # Shade in confidence
    for dim in range(test_x.shape[1]):
        conf = ax.fill_between(test_x[:num_pnts, dim].numpy(), 
                               lower[:num_pnts, task].numpy(), 
                               upper[:num_pnts, task].numpy(), alpha=0.5)

    ax.set_ylim([-3, 3])
    ax.legend([obs, m, conf], ['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'Task {task + 1}')

# Specify the file location and format (e.g., PNG, PDF, SVG, etc.)
file_location = 'artifacts/dssp_opt_4.png'

# Save the figure to the specified file location
plt.savefig(file_location)

fig.tight_layout()
