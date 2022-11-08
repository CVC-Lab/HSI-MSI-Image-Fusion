from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize


from scipy.ndimage.interpolation import rotate
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from .utils import psf2otf, to_torch_sparse
import random
import scipy
import cv2
import pdb
import os
from models.hip.fusion.getLaplacian import getLaplacian
import pypeln as pl

save_laplacians_flag = False

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
        laplacian_img = None
        # print(c)
        for im in os.listdir(c_path):
            if "_ms_" in im:
                ms_path.append(os.path.join(c_path, im))
            elif im.endswith(".bmp"):
                rgb_path = os.path.join(c_path, im)
            # elif im.endswith(".npy"):
            #     laplacian_path = os.path.join(c_path, im)
            #     os.remove(laplacian_path)
            elif im.endswith(".npz"):
                laplacian_path = os.path.join(c_path, im)
                # os.remove(laplacian_path)
                laplacian_img = scipy.sparse.load_npz(laplacian_path)
                
        # print(ms_path)
        ms_img = load_ms_img(ms_path=ms_path)
        rgb_img = np.array(Image.open(rgb_path))
        data.append((c, ms_path, ms_img, rgb_img, laplacian_img))
    return classes, class2id, data

class CAVEDataset(Dataset):
    def __init__(self, dataset_dir, cl, mode="train", sf=8, transform=ToTensor()) -> None:
        super().__init__()
        
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
        c, _, HR_HSI, HR_MSI, lz = self.data[idx]
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
        hr_hsi = hr_hsi.squeeze(0).float()/255
        hr_msi = hr_msi.squeeze(0).float()/255
        lr_hsi = lr_hsi.squeeze(0).float()/255
        # print(f'lr_hsi.shape: {lr_hsi.shape}, hr_hsi.shape: {hr_hsi.shape}, hr_msi.shape: {hr_msi.shape}')
        # [31, 64, 64], [31, 512, 512], [3, 512, 512]
        # to_torch_sparse(lz.tocoo())
        # c, y, z, x, lz
        return c, lr_hsi, hr_msi, hr_hsi, to_torch_sparse(lz.tocoo())#torch.from_numpy(lz.diagonal())# Yh, Ym, X

# dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
# c, lr_hsi, hr_msi, hr_hsi, lz = dataset[0]
# from inside hmi_fusion run - python -m datasets.cave_dataset and set get_laplacian = True initially

def save_laplacians(dataset_dir, cl, sf, mode="train"):
    sf=8
    # preprocess should be run first
    classes, class2id, data = load_data(dataset_dir, mode, cl=cl)
    # nparams = torch.load("./datasets/data/CAVE/normalize_params.pt")
    sizeI = 512
    factor = sf
    def get_laplacian(x):
        # transform = Normalize(mean, std)
        c, im_paths, HR_HSI, HR_MSI, _ = x
        # t = torch.from_numpy(HR_MSI).float().permute(2, 0, 1)
        # HR_MSI = transform(t).permute(1, 2, 0).numpy()
        # normalize HR_MSI first
        lz = getLaplacian(HR_MSI/255, HR_MSI.shape[-1])
        # save file
        folder_path = Path(im_paths[0]).parents[0]
        fp = os.path.join(folder_path, "laplacian.npz")
        print("saving:", fp)
        scipy.sparse.save_npz(fp, lz)
    
    stage = pl.process.map(get_laplacian, data, workers=8, maxsize=8)
    list(stage)
        

if save_laplacians_flag:
    save_laplacians("./datasets/data/CAVE", cl=None, sf=8, mode="train")
    save_laplacians("./datasets/data/CAVE", cl=None, sf=8, mode="test")
