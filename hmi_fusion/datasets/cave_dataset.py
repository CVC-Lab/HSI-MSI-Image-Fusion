
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pdb
import os


def load_ms_img(ms_path):
    ms_img = []
    for p in ms_path:
        im = Image.open(p)
        ms_img.append(np.array(im))
    return np.moveaxis(np.array(ms_img), 0, -1)

# def create_hsi():
    # Shifted blur kernel
    k0 = matlab_style_gauss2D()
    radius = 6
    k = np.zeros((13,13))
    k[0:9,0:9] = k0
    
    center = np.zeros((2,1))
    center[0]=N1/2+1
    center[1]=N2/2+1
    
    I = np.linspace(center[0]-radius,center[1]+radius,13)
    I2 = np.linspace(center[0]-radius,center[1]+radius,13)
    for _ in range(12):
        I2 = np.append(I2,I)
    
    I2 = np.reshape(I2,(169,1))
    J = np.linspace(center[1]-radius,center[1]+radius,13)
    J = np.repeat(J,13,axis = 0)
    
    arr = np.array([I2,J])
    arr = np.reshape(arr,(2,169))
    arr = arr.astype(int)
    I2 = I2.astype(int)
    J = J.astype(int)
    inds = np.ravel_multi_index(arr,(N1-1,N2-1))
    a=0
    kk = np.zeros((N1,N2))
    for i in range(13):
        for j in range(13):
            
            kk[I2[a],J[a]] = k[j,i]
            a=a+1
        
    #Shift Kernel
    kk = np.roll(kk,-int((N1+2)/2), axis=0)
    kk = np.roll(kk,-int((N2+2)/2), axis=1)
    fft = np.fft.fft2(kk)
    
    hsi = np.zeros((n1,n2,N3))
    
    for band in range(N3):
        x = sri[:,:,band]
        Fx = np.fft.fft2(x)
        x = np.multiply(Fx,fft)
        x = np.real(np.fft.ifft2(x))
        hsi[:,:,band] = x[1:-1:4,1:-1:4]
    
    
    hsi,noise_ten2 = add_noise(hsi,30)

test_classes = ["balloons_ms", "cd_ms", "cloth_ms", "photo_and_face_ms", "thread_spools_ms"]


def load_data(dataset_dir, mode):
    data =  []
    dataset_dir = os.path.join(dataset_dir)
    classes = os.listdir(dataset_dir)
    class2id = {c: idx for idx, c in enumerate(classes)}
    if mode == "test":
        classes = test_classes
    else:
        classes = [c for c in classes if c not in test_classes]
    
    for c in os.listdir(dataset_dir):
        if c not in classes: continue
        
        c_path = os.path.join(dataset_dir, c, c)
        if not os.path.isdir(c_path): continue
        ms_path = []
        rgb_path = None
        for im in os.listdir(c_path):
            if "_ms_" in im:
                ms_path.append(os.path.join(c_path, im))
            elif im.endswith(".bmp"):
                rgb_path = os.path.join(c_path, im)
        
        ms_img = load_ms_img(ms_path=ms_path)
        rgb_img = np.array(Image.open(rgb_path))
        data.append((c, ms_img, rgb_img))
    return classes, class2id, data
        

class CAVEDataset(Dataset):
    def __init__(self, dataset_dir, mode="train") -> None:
        super().__init__()
        self.classes, self.class2id, self.data = load_data(dataset_dir, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # need to convert to Tensor
        return self.data[idx]


# dataset = CAVEDataset("./data/CAVE")
# pdb.set_trace()