import os
import random
import numpy as np
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from neural_nets import model_factory
from datasets import dataset_factory
from .train_utils import main_training_loop, test, parse_args
from .transforms import apply_augmentation
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pdb
import itertools
from collections import defaultdict
from tqdm import tqdm

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


def run_experiment(config, experiment_config):
    # Update the config with the current experiment configuration
    config['model']['name'] = experiment_config['model']
    config['dataset']['kwargs']['conductivity'] = experiment_config['conductivity']
    config['model']['kwargs']['A'] = experiment_config['A']

    # Run the experiment
    net, mIOU, gdice = main(config)

    return {
        'model': experiment_config['model'],
        'conductivity': experiment_config['conductivity'],
        'A': experiment_config['A'],
        'mIOU': mIOU,
        'gDice': gdice
    }


def main(config, seed=42):
    torch.cuda.set_device(config['device'])
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    save_path = f'models/sweep_models/trained_{model_name}_{dataset_name}_final_noisy.pth'
    train_dataset = dataset_factory[config['dataset']['name']](
                    **config['dataset']['kwargs'], mode="train", 
                    transforms=apply_augmentation)
    test_dataset = dataset_factory[config['dataset']['name']](
                    **config['dataset']['kwargs'], mode="test", 
                    transforms=apply_augmentation)
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['dataset']['batch_size'], 
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['dataset']['batch_size'], 
                             shuffle=True)
    print('total batches:', len(train_loader))
    DEVICE = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    net = model_factory[model_name](**config['model']['kwargs']).to(torch.double).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', factor=0.5, patience=3)
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    final_ep_loss = main_training_loop(train_loader, net, optimizer, scheduler, 
                       writer=writer, save_path=save_path,
                    num_epochs=config["num_epochs"], device=DEVICE, log_interval=2, config=config)
    # Close the writer
    writer.close()
    mIOU, gdice = test(test_loader, net, save_path=save_path, 
                    num_classes=config['model']['kwargs']['output_channels'])
    return net, mIOU, gdice


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    sweep_config = {
        "conductivity": [0.5, 0.9, 1.4],
        "A": [0.1, 0.3, 0.75],
        "model": ["ca_siamese", "unet", "cnn"]
    }
    results = []
    # Generate all combinations
    keys, values = zip(*sweep_config.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for combination in tqdm(combinations):
        result = run_experiment(config, combination)
        results.append(result)
    # Convert results to a DataFrame for easy analysis
    df = pd.DataFrame(results)
    # Save results to CSV
    df.to_csv('experiment_results.csv', index=False)
    print("Experiments completed. Results saved to 'experiment_results.csv'")