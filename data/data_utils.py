import os
import numpy as np
import scipy
import scipy.io as sio
import cv2
import torch
from pathlib import Path
import pypeln as pl

from img_utils import psf2otf, to_torch_sparse
from BGLRF import getLaplacian

### CONSTANTS
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
R = torch.from_numpy(R)

T = np.array([ [0.005,0.007,0.012,0.015,0.023,0.025,0.030,0.026,0.024,0.019,\
0.010,0.004,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0.000,0.000,0.000,0.000,0.000,0.001,0.002,0.003,0.005,0.007,\
 0.012,0.013,0.015,0.016,0.017,0.02,0.013,0.011,0.009,0.005,\
 0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.002,0.002,\
 0.003],
 [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,\
 0.000,0.000,0.000,0.000,0.000,0.000,0.001,0.003,0.010,0.012,\
 0.013,0.022,0.020,0.020,0.018,0.017,0.016,0.016,0.014,0.014,\
 0.013] ])
T[0] = T[0] / T[0].sum() * T.shape[1]
T[1] = T[1] / T[2].sum() * T.shape[1]
T[2] = T[2] / T[2].sum() * T.shape[1]
T = T.T

################
def load_data(dataset_dir, mode="train", sf=8, train_split=0.6):
    data =  []
    HSdir = os.path.join(dataset_dir, "HS")
    MSdir = os.path.join(dataset_dir, "MS")
    GTdir = os.path.join(dataset_dir, "GT")
    
    for im in os.listdir(GTdir):
        if im.endswith(".mat"):
            data.append(im)

    images = list(os.listdir(GTdir))
    if mode == "train":
        data + images[:int(0.6*len(images))]
    elif mode == "test":
        data + images[int(0.6*len(images)):]
    
    loaded_data = []
    for file in data:
        HS = sio.loadmat(os.path.join(HSdir, str(sf), file))
        MS = sio.loadmat(os.path.join(MSdir, file))
        GT = sio.loadmat(os.path.join(GTdir, file))
        loaded_data.append([HS['hsi'], MS['msi'], GT['ref'], 
        os.path.splitext(file)[0]])

    return loaded_data

def load_ms_img(ms_path):
    ms_img = []
    for p in ms_path:
        # im = Image.open(p)
        im = cv2.imread(p, -1)
        ms_img.append(np.array(im, dtype=np.float32).squeeze())
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
        if c == "watercolors_ms": continue
        
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
                
        
        ms_img = load_ms_img(ms_path=ms_path)
        rgb_img = cv2.imread(rgb_path, -1)
        # rgb_img = np.array(Image.open(rgb_path))
        data.append((c, ms_path, ms_img, rgb_img, laplacian_img))
    return classes, class2id, data

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

def save_dataset(dataset, save_path):
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
        sio.savemat(f"{save_path}cave_{cl}.mat", final_dataset)

'''
dataset = CAVEDataset("../datasets/data/CAVE", cl=None, sf=32, mode="train")
dataset = CAVEDataset("../datasets/data/CAVE", cl=None, sf=32, mode="test")
dataset = HarvardDataset("./datasets/data/Harvard", sf=8, mode="train")
dataset = HarvardDataset("./datasets/data/Harvard", sf=8, mode="test")
'''
