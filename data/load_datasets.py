
import os, sys, glob
sys.path.append("../")
import pdb
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize
import torch.nn.functional as F
from scipy.ndimage.interpolation import rotate
import pypeln as pl

from data.data_utils import load_data, load_ms_img, R, T, para_setting, save_laplacians, loadData

save_laplacians_flag = False
YIQ = torch.Tensor([[0.299 ,0.587 ,0.114], [0.5959 ,-0.2746 ,-0.3213], [0.2115 ,-0.5227 ,0.3112]]) # 3 x3


class CAVEDataset(Dataset):
    def __init__(self, dataset_dir, cl, mode="train", sf=8, transform=ToTensor()) -> None:
        super().__init__()
        
        # pdb.set_trace()
        self.classes, self.class2id, self.data = load_data(dataset_dir, mode, cl=cl)
        self.sizeI = 512
        self.factor = sf # scaling factor
        self.mode = mode
        self.x_states = [
            None for _ in range(len(self.data))
        ]

    def H_z(self, z, factor, fft_B):
        # f = torch.fft.rfft(z, 2, onesided=False)
        f = torch.fft.fft2(z) # [1, 31, 512, 512]
        f = torch.view_as_real(f) #[1, 31, 512, 512, 2]
        
        # -------------------complex myltiply-----------------#
        if len(z.shape) == 3:
            ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
            M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                           (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
            # Hz = torch.fft.irfft(M, 2, onesided=False)
            Hz = torch.fft.ifft2(torch.view_as_complex(M))
            x = Hz[:, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
            M = torch.cat(
                ((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
                 (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)
            # Hz = torch.irfft(M, 2, onesided=False)
            Hz = torch.fft.ifft2(torch.view_as_complex(M))
            x = Hz[:, :, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
       
        return torch.view_as_real(x)[..., 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # need to convert to Tensor
        # c, ms_img, rgb_img = self.data[idx]
        c, _, HR_HSI, HR_MSI, lz = self.data[idx]
        # print(HR_HSI.max())
        # print(c)
        # print(HR_HSI.shape, HR_MSI.shape)
        sz = [self.sizeI, self.sizeI]
        sigma = 2.0
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
        fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

        px      = random.randint(0, 512-self.sizeI)
        py      = random.randint(0, 512-self.sizeI)
        # print("c", c, "hrhsi before:", HR_HSI.shape)
        hr_hsi  = HR_HSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        hr_msi  = HR_MSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        # print(hr_hsi.shape, hr_msi.shape)
        # print("hrhsi after:", hr_hsi.shape)
        hr_hsi = torch.FloatTensor(hr_hsi.copy()).permute(2,0,1).unsqueeze(0)
        hr_msi = torch.FloatTensor(hr_msi.copy()).permute(2,0,1).unsqueeze(0)
        lr_hsi = self.H_z(hr_hsi, self.factor, fft_B)
        lr_hsi = torch.FloatTensor(lr_hsi)

        # hr_hsi = self.transform['x'](hr_hsi.squeeze(0).float())
        # hr_msi = self.transform['z'](hr_msi.squeeze(0).float())
        # lr_hsi = self.transform['y'](lr_hsi.squeeze(0).float())
        # print(hr_hsi.max())
        max_vals = (lr_hsi.max(), hr_msi.max(), hr_hsi.max())
        hr_hsi = hr_hsi.squeeze(0).float()/hr_hsi.max()
        hr_msi = hr_msi.squeeze(0).float()/hr_msi.max()
        lr_hsi = lr_hsi.squeeze(0).float()/lr_hsi.max()

        if type(self.x_states[idx])  == type(None):
            ipt = lr_hsi.numpy()
            x_k = torch.zeros_like(hr_hsi)
            N1, N2 = x_k.shape[1:]
            # print(x_k.shape)
            for c in range(x_k.shape[0]):
                x_k[c, :, :] = torch.FloatTensor(cv2.resize(ipt[c, :, :], (N1, N2), interpolation=cv2.INTER_CUBIC))

            yiq_downsampled = torch.zeros(3, 64, 64)
            C, N1, N2 = hr_msi.shape
            yiq = (YIQ @ hr_msi.reshape(C, -1)).reshape(C, N1, N2)
            # yiq_downsampled = torch.FloatTensor(
            #     cv2.resize(yiq.permute(1, 2, 0).numpy(), (64, 64), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1)[0][None, ...]
            # Zd = torch.FloatTensor(cv2.resize(hr_msi.permute(1, 2, 0).numpy(), (64, 64), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1)
            yiq_downsampled = yiq
            Zd = hr_msi
        else:
            x_k = self.x_states[idx]
            yiq_downsampled = None
        # print(f'lr_hsi.shape: {lr_hsi.shape}, hr_hsi.shape: {hr_hsi.shape}, hr_msi.shape: {hr_msi.shape}')
        # [31, 64, 64], [31, 512, 512], [3, 512, 512]
        # to_torch_sparse(lz.tocoo())
        # c, y, z, x, lz
        # c, x_old, y, z, _, lz, idx
        return lr_hsi, hr_msi, hr_hsi, max_vals
        # return c, x_k, lr_hsi, hr_msi, hr_hsi, to_torch_sparse(lz.tocoo()), yiq_downsampled, Zd, idx#torch.from_numpy(lz.diagonal())# Yh, Ym, X

# from inside hmi_fusion run - python -m datasets.cave_dataset and set get_laplacian = True initially
        

if save_laplacians_flag:
    save_laplacians("./datasets/data/CAVE", cl=None, sf=8, mode="train")
    save_laplacians("./datasets/data/CAVE", cl=None, sf=8, mode="test")

'''
# Path setting
DATASET = "Harvard"
SCALINGS = ([int(sys.argv[1])] if len(sys.argv) >= 2 else [4,8,16])
GT_PATH = f"./datasets/data/{DATASET}/GT/"
MS_PATH = f"./datasets/data/{DATASET}/MS/"
HS_PATH = f"./datasets/data/{DATASET}/HS/"

# if not os.path.exists(GT_PATH):
#     os.makedirs(GT_PATH)
# if not os.path.exists(MS_PATH):
#     os.makedirs(MS_PATH)
# if not os.path.exists(HS_PATH):
#     os.makedirs(HS_PATH)
# if not os.path.exists("CZ_hsdbi.tgz"):
#     os.system("wget http://vision.seas.harvard.edu/hyperspec/d2x5g3/CZ_hsdbi.tgz")
# if not os.path.exists("CZ_hsdbi.tgz"):
#     os.system("wget http://vision.seas.harvard.edu/hyperspec/d2x5g3/CZ_hsdb.tgz")
os.makedirs(GT_PATH, exist_ok = True)
# os.system(f"mv ./datasets/data/{DATASET}/CZ_hsdbi/* {GT_PATH}")
# os.system(f"mv ./datasets/data/{DATASET}/CZ_hsdb/* {GT_PATH}")
# os.system("rm -r data/GT/CZ_hsdbi")
# os.system("rm -r data/GT/CZ_hsdb")
os.makedirs(MS_PATH, exist_ok = True)
for sf in SCALINGS:
    os.makedirs(f"{HS_PATH}/{sf}", exist_ok = True)

for mat_path in glob.iglob(f"{GT_PATH}/*.mat"):
    name = Path(mat_path).stem
    print(name)
    mat = scipy.io.loadmat(mat_path)
    hsi = mat["ref"][:1024, :1024, :]
    # pdb.set_trace()
    # downsampling HS image
    for sf in SCALINGS:
        hsi_downsampled = None
        for i in range(hsi.shape[2]):
            img = Image.fromarray(hsi[:,:,i])
            img = img.resize((hsi.shape[1]//sf, hsi.shape[0]//sf),
            Image.LANCZOS)
            # from Image to np
            img = np.expand_dims(np.asarray(img), axis=2)
            hsi_downsampled = img if hsi_downsampled is None else np.concatenate((hsi_downsampled , img), axis=2)
        scipy.io.savemat(f"{HS_PATH}/{sf}/{name}.mat", {"hsi":
        hsi_downsampled})
        # simulate RGB photo with Nikon D700 camera
    msi = np.dot(hsi,T)
    scipy.io.savemat(f"{MS_PATH}/{name}.mat", {"msi": msi})
    mat['ref'] = hsi
    scipy.io.savemat(mat_path, mat)
'''

class HarvardDataset(Dataset):
    def __init__(self, dataset_dir, mode="train", sf=8) -> None:
        super().__init__()
        self.data = load_data(dataset_dir, sf=sf, mode=mode)
        # img size - 1040, 1392
        self.sz = [1024, 1024]
        self.factor = sf # scaling factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        HSI, MSI, GT, name = self.data[idx]
        HSI = torch.FloatTensor(HSI).permute(2, 1, 0)
        MSI = torch.FloatTensor(MSI).permute(2, 1, 0)
        GT = torch.FloatTensor(GT).permute(2, 1, 0)

        HSI = HSI/HSI.max()
        MSI = MSI/MSI.max()
        GT = GT/GT.max()
        # print(f'lr_hsi.shape: {lr_hsi.shape}, hr_hsi.shape: {hr_hsi.shape}, hr_msi.shape: {hr_msi.shape}')
        # lr_hsi.shape: torch.Size([1, 130, 174]), hr_hsi.shape: torch.Size([1, 1040, 1392]), hr_msi.shape: torch.Size([31, 1040, 1392])
        # lr_hsi, hr_msi, hr_hsi
        return HSI, MSI, GT # Yh, Ym, X

        # return c, x_k, lr_hsi, hr_msi, hr_hsi, torch.T, yiq_downsampled, Zd, idx

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

class IndianPinesDataset(Dataset):
    def __init__(self, dataset_dir) -> None:
        super().__init__()
        self.data = loadData(dataset_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # need to convert to Tensor
        return self.data[idx]

# tensorflow dataset
class SegTrackV2Dataset(Dataset):
    def __init__(self, folder_path) -> None:
        super().__init__()
        self.folder_path = folder_path
        classes_file_path = open(os.path.join(self.folder_path, "ImageSets", "all.txt"), "r")
        self.classes = [filename[1:].strip() for filename in classes_file_path.readlines()]
        self.class2id = {c: idx for idx, c in enumerate(self.classes)}
        print(self.class2id)
        # read ground truth and make tuples
        gt_folder = os.path.join(self.folder_path, "GroundTruth")
        self.data = [] 
        for c in os.listdir(gt_folder):
            if c.startswith("."): continue
            c_path = os.path.join(gt_folder, c)
            print(c_path)
            files = [f for f in os.listdir(c_path) if not f.startswith('.')]
            print(files[:4])
            if os.path.isdir(os.path.join(c_path, files[0])):
                for obj_id in files:
                    obj_id_dir = os.path.join(c_path, obj_id)
                    for fname in os.listdir(obj_id_dir):
                        rgb_img_path = os.path.join(self.folder_path, "JPEGImages", c, fname)
                        seg_mask_path = os.path.join(obj_id_dir, fname)
                        self.data.append((obj_id, self.class2id[c], rgb_img_path, seg_mask_path))
                else:
                    obj_id = 1
                    obj_id_dir = c_path
                    for fname in os.listdir(obj_id_dir):
                        rgb_img_path = os.path.join(self.folder_path, "JPEGImages", c, fname)
                        seg_mask_path = os.path.join(obj_id_dir, fname)
                        self.data.append((int(obj_id), self.class2id[c], rgb_img_path, seg_mask_path))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj_id, c_id, rgb_path, seg_mask_path = self.data[idx]
        # load rgb_img
        # load seg_mask
        # hsi, msi, sri = generate(rgb_img)
        return obj_id, c_id, rgb_path, seg_mask_path
        # object_id, class_id, RGB_image, segmentation_mask
       
if __name__ == '__main__':
    # git clone segtrackv2.zip -> unzip-> folder_path
    dataset = SegTrackV2Dataset("/Users/shubham1.bhardwaj/Documents/masters_coursework/3D_Prof_Chandrajit/HSI-MSI-Image-Fusion/hmi_fusion/datasets/data/SegTrackv2_small")
    pdb.set_trace()

    dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
    # c, x_k, lr_hsi, hr_msi, hr_hsi, lz, idx = dataset[0]
    # pdb.set_trace()

    # load_data("./datasets/data/Harvard")
    # dataset = HarvardDataset("./datasets/data/Harvard", mode="test")
    # pdb.set_trace()
