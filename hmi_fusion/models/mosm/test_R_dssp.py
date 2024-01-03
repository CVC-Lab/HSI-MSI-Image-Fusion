
from einops import rearrange, reduce, repeat, pack, unpack
from datasets.cave_dataset import CAVEDataset
from .mosm_models import MultitaskGPModel
from .create_point_level_dataset import prepare_point_ds
from torch.utils.data import TensorDataset, DataLoader
from .mosm_models import MultitaskDeepGP

import gpytorch
import torch
import yaml
import tqdm
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
# Load parameters from config.yaml
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

print(save_path)
test_dataset = CAVEDataset(data_path, None, mode="test")

state_dict = torch.load(save_path)

model = MultitaskDeepGP([4096, 3], 
                        num_inducing=num_inducing,
                        num_hidden_dgp_dims=num_hidden_dgp_dims, 
                        num_tasks=num_tasks)
model = model.cuda()
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
model.load_state_dict(state_dict)


def predict(test_loader):
    with torch.no_grad():

        # The output of the model is a multitask MVN, where both the data points
        # and the tasks are jointly distributed
        # To compute the marginal predictive NLL of each data point,
        # we will call `to_data_independent_dist`,
        # which removes the data cross-covariance terms from the distribution.
        means, vars = [], []
        lowers, uppers = [], []
        for tx, ty in tqdm.tqdm(test_loader):
            tx = tx.cuda() # [4096, 3]
            # preds = model.likelihood(model(tx)).to_data_independent_dist()
            preds = model.likelihood(model(tx)) # [10, 4096, 31]
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


# load image from dataset convert to pointwise then extract mean and convert back to image
total_psnr, total_ssim, total_rmse, total_sam, total_ergas =0,0,0,0,0
for items in test_dataset:
    y, z, x_gt, _  = items
    test_x = rearrange(z, 'c h w -> (h w) c')
    test_y = rearrange(x_gt, 'c h w -> (h w) c')
    
    test_ds = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mean, lower, upper = predict(test_loader)
        pred_x = torch.reshape(mean, x_gt.shape)
        # pred_x = rearrange(mean, '* c -> c h w')

    # print(pred_x.shape)
    # pdb.set_trace()
    x_gt = rearrange(x_gt, 'c h w -> h w c').detach().cpu().numpy()
    pred_x = rearrange(pred_x, 'c h w -> h w c').detach().cpu().numpy()
    total_ssim += compare_mssim(x_gt, pred_x)
    rmse,  mse, rmse_per_band = find_rmse(x_gt, pred_x)
    total_rmse += rmse
    total_psnr += compare_mpsnr(x_gt, pred_x, mse)
    total_sam += torch.nan_to_num(sam(torch.from_numpy(x_gt).permute(2, 0, 1)[None, ...], 
                            torch.from_numpy(pred_x).permute(2, 0, 1)[None, ...]), nan=0, posinf=1.0) * (180/torch.pi)
    total_ergas += compare_ergas(x_gt, pred_x, 8, rmse_per_band)
    

opt = f"""## Metric scores:
psnr:{total_psnr/len(test_dataset)},
ssim:{total_ssim/len(test_dataset)},
rmse:{total_rmse/len(test_dataset)},
sam:{total_sam/len(test_dataset)},
ergas:{total_ergas/len(test_dataset)},
"""
print(opt)


