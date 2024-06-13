import os
import random
import numpy as np
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from neural_nets import model_factory, model_args
from datasets import dataset_factory
from train_utils import main_training_loop, test
from transforms import apply_transforms, apply_augmentation


# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Run deep learning experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    torch.cuda.set_device(config['device'])
    model_name = config['model']['name']
    save_path = f'models/trained_{model_name}_final_noisy.pth'
    train_dataset = dataset_factory[config['dataset']['type']](
                    **config['dataset']['kwargs'], mode="train", 
                    transforms=apply_augmentation)
    test_dataset = dataset_factory[config['dataset']['type']](
                    **config['dataset']['kwargs'], mode="test", 
                    transforms=apply_augmentation)
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['dataset']['batch_size'], 
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['dataset']['batch_size'], 
                             shuffle=True)
    DEVICE = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    net = model_factory[model_name](*config['model']['args']).to(torch.double).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', factor=0.5, patience=3)
    main_training_loop(train_loader, net, optimizer, scheduler, save_path=save_path,
                    num_epochs=40, device=DEVICE, log_interval=2)

    mIOU, gdice = test(test_loader, net, save_path=save_path, num_classes=4)
    print(f"mIOU: {mIOU}, gdice: {gdice}")
main()