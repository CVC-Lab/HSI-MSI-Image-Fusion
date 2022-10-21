from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from scipy.ndimage.interpolation import rotate
import torch
from PIL import Image
import numpy as np
from .utils import psf2otf
import random
import cv2
import pdb
import os


R = [[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019,
        0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007,
        0.012, 0.013, 0.015, 0.016, 0.017, 0.02, 0.013, 0.011, 0.009, 0.005,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022,
        0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]]

R = np.array(R).astype(np.float32)
R = R.astype(np.float32)

def load_ms_img(ms_path):
    ms_img = []
    for p in ms_path:
        # im = Image.open(p)
        im = cv2.imread(p, 0)
        ms_img.append(np.array(im))
    # print(np.moveaxis(np.array(ms_img), 0, -1).shape)
    return np.moveaxis(np.array(ms_img), 0, -1)

def para_setting(kernel_type, sf, sz, sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


test_classes = ["balloons_ms", "cd_ms", "cloth_ms", "photo_and_face_ms", "thread_spools_ms"]


def load_data(dataset_dir, mode, cl):
    data =  []
    dataset_dir = os.path.join(dataset_dir)
    classes = os.listdir(dataset_dir)
    class2id = {c: idx for idx, c in enumerate(classes)}

    if cl:
        classes = cl
    elif mode == "test":
        classes = test_classes
    else:
        classes = [c for c in classes if c not in test_classes]
    
    for c in os.listdir(dataset_dir):
        if c not in classes: continue
        
        c_path = os.path.join(dataset_dir, c, c)
        if not os.path.isdir(c_path): continue
        ms_path = []
        rgb_path = None
        # print(c)
        for im in os.listdir(c_path):
            if "_ms_" in im:
                ms_path.append(os.path.join(c_path, im))
            elif im.endswith(".bmp"):
                rgb_path = os.path.join(c_path, im)
        # print(ms_path)
        ms_img = load_ms_img(ms_path=ms_path)
        
        rgb_img = np.array(Image.open(rgb_path))
        data.append((c, ms_img, rgb_img))
    return classes, class2id, data
        

class CAVEDataset(Dataset):
    def __init__(self, dataset_dir, cl, mode="train", sf=8, transform=ToTensor()) -> None:
        super().__init__()
        self.transform = transform
        # pdb.set_trace()
        self.classes, self.class2id, self.data = load_data(dataset_dir, mode, cl=cl)
        self.sizeI = 512
        self.factor = sf # scaling factor
        self.mode = mode

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
        c, HR_HSI, HR_MSI = self.data[idx]
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

        hr_hsi = hr_hsi.squeeze(0)
        hr_msi = hr_msi.squeeze(0)
        lr_hsi = lr_hsi.squeeze(0)
        # print(f'lr_hsi.shape: {lr_hsi.shape}, hr_hsi.shape: {hr_hsi.shape}, hr_msi.shape: {hr_msi.shape}')
        # [31, 64, 64], [31, 512, 512], [3, 512, 512]
        return lr_hsi, hr_msi, hr_hsi # Yh, Ym, X

# dataset = CAVEDataset("./data/CAVE", mode="train")


# final_dataset = {
#     'lr_hsi': [],
#     'hr_msi': [],
#     'hr_hsi': []
# }
# dataset = CAVEDataset("./data/CAVE", mode="test", cl="balloons_ms")
# pdb.set_trace()
# for i in range(len(dataset)):
#     lr_hsi, hr_msi, hr_hsi = dataset[i]
#     pdb.set_trace()
#     final_dataset['lr_hsi'].append(lr_hsi.numpy())
#     final_dataset['hr_msi'].append(hr_msi.numpy())
#     final_dataset['hr_hsi'].append(hr_hsi.numpy())


# class CAVEDatasetSingleShape(Dataset):
#     def __init__(self, dataset_dir, cl, mode="train", sf=8, transform=ToTensor()) -> None:
#         super().__init__()
#         self.transform = transform
#         self.classes, self.class2id, self.data = load_data(dataset_dir, mode, cl=cl)
#         self.sizeI = 512
#         self.factor = sf # scaling factor
#         self.mode = mode

#     def H_z(self, z, factor, fft_B):
#         # f = torch.fft.rfft(z, 2, onesided=False)
#         f = torch.fft.fft2(z) # [1, 31, 512, 512]
#         f = torch.view_as_real(f) #[1, 31, 512, 512, 2]
        
#         # -------------------complex myltiply-----------------#
#         if len(z.shape) == 3:
#             ch, h, w = z.shape
#             fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
#             M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
#                            (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
#             # Hz = torch.fft.irfft(M, 2, onesided=False)
#             Hz = torch.fft.ifft2(torch.view_as_complex(M))
#             x = Hz[:, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
#         elif len(z.shape) == 4:
#             bs, ch, h, w = z.shape
#             fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
#             M = torch.cat(
#                 ((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
#                  (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)
#             # Hz = torch.irfft(M, 2, onesided=False)
#             Hz = torch.fft.ifft2(torch.view_as_complex(M))
#             x = Hz[:, :, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
       
#         return torch.view_as_real(x)[..., 0]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # need to convert to Tensor
#         # c, ms_img, rgb_img = self.data[idx]
#         c, HR_HSI, HR_MSI = self.data[idx]
#         sz = [self.sizeI, self.sizeI]
#         sigma = 2.0
#         fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
#         fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
#         fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

#         px      = random.randint(0, 512-self.sizeI)
#         py      = random.randint(0, 512-self.sizeI)
#         # print("c", c, "hrhsi before:", HR_HSI.shape)
#         hr_hsi  = HR_HSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
#         hr_msi  = HR_MSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]

#         # print("hrhsi before:", hr_hsi.shape)
#         # if self.mode == "train":
#             # rotTimes = random.randint(0, 3)
#             # vFlip    = random.randint(0, 1)
#             # hFlip    = random.randint(0, 1)
#             # rot_angle = random.randint(0, 10)
#             # hr_hsi = rotate(hr_hsi, angle=rot_angle)
#             # hr_msi  = rotate(hr_msi, angle=rot_angle)
            

#         # print("hrhsi after:", hr_hsi.shape)
#         hr_hsi = torch.FloatTensor(hr_hsi.copy()).permute(2,0,1).unsqueeze(0)
#         hr_msi = torch.FloatTensor(hr_msi.copy()).permute(2,0,1).unsqueeze(0)
#         lr_hsi = self.H_z(hr_hsi, self.factor, fft_B)
#         lr_hsi = torch.FloatTensor(lr_hsi)

#         hr_hsi = hr_hsi.squeeze(0)
#         hr_msi = hr_msi.squeeze(0)
#         lr_hsi = lr_hsi.squeeze(0)
#         # print(f'lr_hsi.shape: {lr_hsi.shape}, hr_hsi.shape: {hr_hsi.shape}, hr_msi.shape: {hr_msi.shape}')
#         # [31, 64, 64], [31, 512, 512], [3, 512, 512]
#         return lr_hsi, hr_msi, hr_hsi # Yh, Ym, X