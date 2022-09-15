from torch.utils.data import Dataset
import scipy.io as sio
import pdb

class PaviaDataset(Dataset):
    def __init__(self, dataset_dir) -> None:
        super().__init__()
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # need to convert to Tensor
        return self.data[idx]

# gt = sio.loadmat("/Users/shubham1.bhardwaj/Documents/masters_coursework/3D_Prof_Chandrajit/HSI-MSI-Image-Fusion/hmi_fusion/datasets/data/PaviaUniversity/PaviaU_gt.mat")
# # gt shape - (610, 340)
# fpath = "/Users/shubham1.bhardwaj/Documents/masters_coursework/3D_Prof_Chandrajit/HSI-MSI-Image-Fusion/hmi_fusion/datasets/data/PaviaUniversity/Pavia.mat"
# tr = sio.loadmat(fpath)
# pdb.set_trace()