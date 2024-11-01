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
from train_utils.train_utils import main_training_loop, test, parse_args
from adversity.transforms import apply_augmentation
from train_utils.config_generator import ConfigGenerator
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


def run_experiment(experiment_config):
    dataset_name = experiment_config['dataset']['name']
    with open(f'configs/{dataset_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_name = experiment_config['model']['name']
    # Update the config with the current experiment configuration
    config['model']['kwargs'].update(experiment_config['model']['kwargs'])
    config['model']['name'] = model_name
    config['dataset']['name'] =  dataset_name
    config['dataset']['kwargs'].update(experiment_config['dataset']['kwargs'])
    # Run the experiment
    net, mIOU, gdice = main(config)
    result = {}
    result['model'] = experiment_config['model']['name']
    result['dataset'] = experiment_config['dataset']['name']
    result.update(experiment_config['model']['kwargs'])
    result.update(experiment_config['dataset']['kwargs'])
    result['mIOU'] = mIOU
    result['gDice'] = gdice
    return result


def main(config, seed=42):
    torch.cuda.set_device(config['device'])
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    A = config["dataset"]["kwargs"]["A"]
    gamma = config["dataset"]["kwargs"]["gamma"]
    save_path = f'models/sweep_models/trained_{model_name}_{dataset_name}.pth'
    if model_name.startswith("pixel"):
        config["dataset"]["kwargs"]["data_type"] = "pixel"
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
                    num_epochs=config["num_epochs"], device=DEVICE, log_interval=1, config=config)
    # Close the writer
    writer.close()
    mIOU, gdice = test(test_loader, net, save_path=save_path, 
                    num_classes=config['model']['kwargs']['output_channels'])
    print(mIOU, gdice)
    config["dataset"]["kwargs"]["data_type"] = None
    return net, mIOU, gdice


if __name__ == "__main__":
    # args = parse_args()
    # with open(args.config, 'r') as file:
    #     config = yaml.safe_load(file)
    sweep_config = {
        "model": 
            {
                "name":["pixel_mlp"],
                "kwargs": {
                    },
            },
        "dataset":
            {
                "name": ["urban"],
                "kwargs":{
                    # "A": [0.1, 0.3, 0.75],
                    # "gamma": [0.4, 0.7, 1.0, 2.5]
                    "split_ratio": [0.25, 0.50, 0.75, 0.95]
                }
            }
    }
    results = []
    # Generate all combinations
    config_generator = ConfigGenerator(sweep_config)
    combinations = config_generator.get_all_configs()
    for combination in tqdm(combinations):
        result = run_experiment(combination)
        results.append(result)
    # Convert results to a DataFrame for easy analysis
    df = pd.DataFrame(results)
    # Save results to CSV
    os.makedirs('artifacts', exist_ok=True)
    filename = 'artifacts/experiments_pixelmlp_final_compare.csv'
    df.to_csv(filename, index=False)
    print(f"Experiments completed. Results saved to '{filename}'")