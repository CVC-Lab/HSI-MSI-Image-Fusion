from ..metrics import (
    compare_mpsnr,
    compare_mssim,
    find_rmse,
    # compare_sam,
    compare_ergas
)
import torch
from torchmetrics import SpectralAngleMapper
from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS
# from datasets.cave_dataset import CAVEDataset
# from .dbglrf import AutoEncoder
sam = SpectralAngleMapper()
ergas = ERGAS(ratio=1/8)

def get_final_metric_scores(model, test_dataset, sf):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=False, batch_size=1)
    model.eval()
    total_psnr, total_ssim, total_rmse, total_sam, total_ergas =0,0,0,0,0
    with torch.no_grad():
        for items in test_loader:
            c, x_old, y, z, x, lz, idx = items
            x_old = x_old.cuda()
            z = z.cuda()
            y = y.cuda()
            
            # lz = lz.cuda()
            _, x2 = model(x_old, z, lz)
            # _, x2 = model(y, z)
            # x2 = model(x_old, z, lz)
            # c, _, y, z, x, lz, idxs = items
            # _, x2 = model(y)
            # x2 = x
            x = x.squeeze()
            x2 = x2.squeeze()
            x = x.permute(1, 2, 0).detach().cpu().numpy()
            x2 = x2.permute(1, 2, 0).detach().cpu().numpy()
            
            total_ssim += compare_mssim(x, x2)
            rmse,  mse, rmse_per_band = find_rmse(x, x2)
            total_rmse += rmse
            total_psnr += compare_mpsnr(x, x2, mse)
            total_sam += sam(torch.from_numpy(x).permute(2, 0, 1)[None, ...], 
                                    torch.from_numpy(x2).permute(2, 0, 1)[None, ...]) * (180/torch.pi)
            total_ergas += compare_ergas(x, x2, 1/8, rmse_per_band)
            # total_sam += compare_sam(x, x2)
            # total_ergas += ergas(torch.from_numpy(x).permute(2, 0, 1)[None, ...], 
            #                     torch.from_numpy(x2).permute(2, 0, 1)[None, ...])
            # total_ergas += compare_ergas(x, x2,1/sf, rmse_per_band)

    opt = f"""## Metric scores:
    psnr:{total_psnr/len(test_loader)},
    ssim:{total_ssim/len(test_loader)},
    rmse:{total_rmse/len(test_loader)},
    sam:{total_sam/len(test_loader)},
    ergas:{total_ergas/len(test_loader)},
    """
    print(opt)
    return opt
    

# sf = 8
# in_channels = 31
# dec_channels = 31
# model_name = "ae"
# model_path = f"./artifacts/{model_name}/gmodel.pth"
# model = AutoEncoder(in_channels=in_channels, dec_channels=dec_channels)
# test_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="test")
# get_final_metric_scores(model, test_dataset, sf)