import sys
sys.path.append("../")
from datasets.harvard_dataset import HarvardDataset
import scipy.io as sio
import pdb



# dataset = HarvardDataset("../datasets/data/Harvard", sf=32, mode="train")
# for i in range(len(dataset)):
#     final_dataset = {
#         'lr_hsi': [],
#         'hr_msi': [],
#         'hr_hsi': []
#     }
#     name, lr_hsi, hr_msi, hr_hsi = dataset[i]
#     final_dataset['lr_hsi'] = lr_hsi.permute(1, 2, 0).numpy()
#     final_dataset['hr_msi'] = hr_msi.permute(1, 2, 0).numpy()
#     final_dataset['hr_hsi'] = hr_hsi.permute(1, 2, 0).numpy()
#     sio.savemat(f"../datasets/data/Harvard/matfiles/harvard_{name}.mat", final_dataset)

dataset = HarvardDataset("../datasets/data/Harvard", sf=32, mode="test")
for i in range(len(dataset)):
    final_dataset = {
        'lr_hsi': [],
        'hr_msi': [],
        'hr_hsi': []
    }
    name, lr_hsi, hr_msi, hr_hsi = dataset[i]
    final_dataset['lr_hsi'] = lr_hsi.permute(1, 2, 0).numpy()
    final_dataset['hr_msi'] = hr_msi.permute(1, 2, 0).numpy()
    final_dataset['hr_hsi'] = hr_hsi.permute(1, 2, 0).numpy()

    sio.savemat(f"../datasets/data/Harvard/matfiles/harvard_{name}.mat", final_dataset)