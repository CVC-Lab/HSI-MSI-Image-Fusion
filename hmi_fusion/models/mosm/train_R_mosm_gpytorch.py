import math
import torch
import gpytorch
import tqdm
from matplotlib import pyplot as plt
from datasets.cave_dataset import CAVEDataset, R
from .create_point_level_dataset import prepare_point_ds
import pdb
import numpy as np
import yaml
from .mosm_models import MultitaskGPModel



# Load parameters from config.yaml
with open('models/mosm/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access parameters
num_tasks = config['num_tasks']
num_latents = config['num_latents']
smoke = config['smoke']
data_path = config['data_path']
num_epochs = config['num_epochs']
save_path = config['save_path']
# data_path = "./datasets/data/CAVE"
dataset = CAVEDataset(data_path, None, mode="train")
test_dataset = CAVEDataset(data_path, None, mode="test")
train_x, train_y = prepare_point_ds(dataset=dataset)
test_x, test_y = prepare_point_ds(dataset=test_dataset)

# channels come at the end
print(train_x.shape, train_y.shape)
# print('total batches: train_x.shape[0]/batch_size')

model = MultitaskGPModel(num_tasks=num_tasks, num_latents=num_latents)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
num_epochs = 1 if smoke else num_epochs
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

best_loss = np.inf
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), save_path)


# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

num_pnts = 5000
for task, ax in enumerate(axs):
    # Plot training data as black stars
    for dim in range(train_x.shape[1]):
        obs, = ax.plot(train_x[:num_pnts, dim].detach().numpy(), train_y[:num_pnts, task].detach().numpy(), 'k*')

    # Predictive mean as blue line
    for dim in range(test_x.shape[1]):
        m, = ax.plot(test_x[:num_pnts, dim].numpy(), mean[:num_pnts, task].numpy(), 'b')

    # Shade in confidence
    for dim in range(test_x.shape[1]):
        conf = ax.fill_between(test_x[:num_pnts, dim].numpy(), lower[:num_pnts, task].numpy(), upper[:num_pnts, task].numpy(), alpha=0.5)

    ax.set_ylim([-3, 3])
    ax.legend([obs, m, conf], ['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'Task {task + 1}')

# Specify the file location and format (e.g., PNG, PDF, SVG, etc.)
file_location = 'artifacts/mosm_opt.png'

# Save the figure to the specified file location
plt.savefig(file_location)

fig.tight_layout()
None