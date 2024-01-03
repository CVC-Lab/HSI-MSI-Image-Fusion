
from einops import rearrange, reduce, repeat, pack, unpack
from datasets.cave_dataset import CAVEDataset
from .mosm_models import MultitaskGPModel
import gpytorch
import torch
import yaml
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
with open('models/mosm/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access parameters
num_tasks = config['num_tasks']
num_latents = config['num_latents']
smoke = config['smoke']
data_path = config['data_path']
num_epochs = config['num_epochs']
save_path = config['save_path']

print(save_path)
test_dataset = CAVEDataset(data_path, None, mode="test")
state_dict = torch.load(save_path)
model = MultitaskGPModel(num_tasks=num_tasks, num_latents=num_latents)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
model.load_state_dict(state_dict)

# load image from dataset convert to pointwise then extract mean and convert back to image
total_psnr, total_ssim, total_rmse, total_sam, total_ergas =0,0,0,0,0
for items in test_dataset:
    y, z, x_gt, _  = items
    test_x = rearrange(z, 'c h w -> (h w) c')
    test_y = rearrange(x_gt, 'c h w -> (h w) c')
    
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
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


