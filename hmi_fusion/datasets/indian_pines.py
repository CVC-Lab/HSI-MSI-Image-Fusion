from torch.utils.data import Dataset
# from datasets import load_dataset
# dataset = load_dataset("cvc-lab/IndianPines")

# download an unzip dataset from huggingface

# need to write local dataset dataloader

class IndianPinesDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass