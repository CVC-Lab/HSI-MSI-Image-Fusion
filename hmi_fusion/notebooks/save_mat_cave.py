import sys
sys.path.append("../")
from datasets.cave_dataset import CAVEDataset
import scipy.io as sio
import pdb


classes = ["balloons_ms", "cd_ms", "cloth_ms", "photo_and_face_ms", "thread_spools_ms"]

for cl in classes:
    final_dataset = {
        'lr_hsi': [],
        'hr_msi': [],
        'hr_hsi': []
    }
    dataset = CAVEDataset("../datasets/data/CAVE", cl=cl, mode="test")
    for i in range(len(dataset)):
        lr_hsi, hr_msi, hr_hsi = dataset[i]
        final_dataset['lr_hsi'].append(lr_hsi.numpy())
        final_dataset['hr_msi'].append(hr_msi.numpy())
        final_dataset['hr_hsi'].append(hr_hsi.numpy())
    sio.savemat(f"../datasets/data/CAVE/cave_{cl}.mat", final_dataset)