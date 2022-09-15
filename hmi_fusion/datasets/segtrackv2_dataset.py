from torch.utils.data import Dataset
import os
import pdb

# from datasets import load_dataset
# dataset = load_dataset("cvc-lab/SegTrackV2", use_auth_token=True)

# need to add a dataloader script to the dataset




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
       

# git clone segtrackv2.zip -> unzip-> folder_path
dataset = SegTrackV2Dataset("/Users/shubham1.bhardwaj/Documents/masters_coursework/3D_Prof_Chandrajit/HSI-MSI-Image-Fusion/hmi_fusion/datasets/data/SegTrackv2_small")
pdb.set_trace()
