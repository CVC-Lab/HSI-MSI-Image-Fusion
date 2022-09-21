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


# dataset = CAVEDataset("./data/CAVE", mode="train")
# pdb.set_trace()
