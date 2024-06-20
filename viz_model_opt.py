import os
import random
import numpy as np
import cv2
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from neural_nets import model_factory, model_args
from datasets import dataset_factory
from einops import rearrange
from PIL import Image
import pdb
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

# Define the color mapping for each class
color_mapping = {
    0: [128, 64, 128],  # Road - Purple
    1: [244, 164, 96],  # Soil - SandyBrown
    2: [0, 0, 255],     # Water - Blue
    3: [34, 139, 34],   # Tree - ForestGreen
}

def predictions_to_colored_image(predictions):
    # Get the dimensions of the predictions array
    height, width = predictions.shape

    # Create an empty array for the colored image with 3 channels (RGB)
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each class index to the corresponding color
    for class_index, color in color_mapping.items():
        colored_image[predictions == class_index] = color

    return colored_image

def save_colored_image(image, file_path):
    # Convert the numpy array to a PIL Image
    pil_image = Image.fromarray(image)
    # Save the image
    pil_image.save(file_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run deep learning experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()


def main():
    os.makedirs('images', exist_ok=True)
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    torch.cuda.set_device(config['device'])
    model_name = config['model']['name']
    save_path = f'models/trained_{model_name}_final_noisy.pth'
    train_dataset = dataset_factory[config['dataset']['type']](
                    **config['dataset']['kwargs'], mode="train", 
                    transforms=None)
    
    DEVICE = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    net = model_factory[model_name](**config['model']['args']).to(torch.double).to(DEVICE)
    net.load_state_dict(torch.load(save_path))
    net.to(DEVICE)
    num_classes = 4
    miou = MeanIoU(num_classes=num_classes, per_class=True)
    gdice = GeneralizedDiceScore(num_classes=num_classes, include_background=False)
    
    sri, msi, gt = train_dataset.img_sri, train_dataset.img_rgb, train_dataset.gt
    if not os.path.exists('images/gt.png'):
        preds = np.argmax(gt, axis=-1)
        pred_img = predictions_to_colored_image(preds)
        save_colored_image(pred_img, f'images/gt.png')
        
    sub_hsi = train_dataset.downsample(sri)
    sub_hsi = np.moveaxis(sub_hsi, 2, 0)[None, :, :, :]
    msi = np.moveaxis(msi, 2, 0)[None, :, :, :]
    sub_hsi = torch.from_numpy(sub_hsi)
    msi = torch.from_numpy(msi)
    outputs = net(sub_hsi.to(DEVICE), msi.to(DEVICE))
    predictions = torch.argmax(outputs.cpu(), axis=1).squeeze()
    miou_score = miou(predictions, torch.from_numpy(np.argmax(gt, axis=-1))).numpy()
    gdice_score = gdice(predictions[None, :, :], torch.from_numpy(np.argmax(gt, axis=-1))[None, :, :]).numpy()
    print('miou:', miou_score)
    print('gDice:', gdice_score)
    pred_img = predictions_to_colored_image(predictions)
    save_colored_image(pred_img, f'images/{model_name}.png')
    
main()