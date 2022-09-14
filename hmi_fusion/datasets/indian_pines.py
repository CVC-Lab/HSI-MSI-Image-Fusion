from torch.utils.data import Dataset
from ..models.hip.simulation.generation import generate_indian_pines_data
# from datasets import load_dataset
# dataset = load_dataset("cvc-lab/IndianPines")

# download an unzip dataset from huggingface

# need to write local dataset dataloader

class IndianPinesDataset(Dataset):
    def __init__(self, dataset_dir) -> None:
        super().__init__()
        hsi, msi, sri = generate_indian_pines_data(dataset_dir)
        self.data = [(hsi, msi, sri)]

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # need to convert to Tensor
        return self.data[idx]
