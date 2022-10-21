import sys
sys.path.append("../")
from datasets.cave_dataset import CAVEDataset
import scipy.io as sio
import pdb


# classes = ["balloons_ms", "cd_ms", "cloth_ms", "photo_and_face_ms", "thread_spools_ms"]

# for cl in classes:

dataset = CAVEDataset("../datasets/data/CAVE", cl=None, sf=32, mode="train")
for i in range(len(dataset)):
    final_dataset = {
        'lr_hsi': [],
        'hr_msi': [],
        'hr_hsi': []
    }
    cl, lr_hsi, hr_msi, hr_hsi = dataset[i]
    final_dataset['lr_hsi'] = lr_hsi.permute(1, 2, 0).numpy()
    final_dataset['hr_msi'] = hr_msi.permute(1, 2, 0).numpy()
    final_dataset['hr_hsi'] = hr_hsi.permute(1, 2, 0).numpy()
    sio.savemat(f"../datasets/data/CAVE/matfiles/cave_{cl}.mat", final_dataset)

dataset = CAVEDataset("../datasets/data/CAVE", cl=None, sf=32, mode="test")
for i in range(len(dataset)):
    final_dataset = {
        'lr_hsi': [],
        'hr_msi': [],
        'hr_hsi': []
    }
    cl, lr_hsi, hr_msi, hr_hsi = dataset[i]
    final_dataset['lr_hsi'] = lr_hsi.permute(1, 2, 0).numpy()
    final_dataset['hr_msi'] = hr_msi.permute(1, 2, 0).numpy()
    final_dataset['hr_hsi'] = hr_hsi.permute(1, 2, 0).numpy()
    sio.savemat(f"../datasets/data/CAVE/matfiles/cave_{cl}.mat", final_dataset)