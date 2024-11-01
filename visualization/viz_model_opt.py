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
import matplotlib.pyplot as plt
from ..scripts.losses import calculate_psnr
from ..train_utils.train_utils import parse_args


# Define the color mapping for each class
color_mappings = {
    "jasper_ridge": {
        0: [128, 64, 128],  # Road - Purple
        1: [244, 164, 96],  # Soil - SandyBrown
        2: [0, 0, 255],     # Water - Blue
        3: [34, 139, 34],   # Tree - ForestGreen
    }
}



def predictions_to_colored_image(predictions, color_mapping):
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

    

def psnr_blocks(image1, image2, block_size, num_classes):
    h, w = image1.shape[:2]
    psnr_values = np.zeros((h // block_size, w // block_size))

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block1 = image1[i:i+block_size, j:j+block_size]
            block2 = image2[i:i+block_size, j:j+block_size]
            # print(f"miou ({i}, {j}): {calculate_miou(block1, block2)}" )
            psnr_values[i // block_size, j // block_size] = calculate_psnr(block1, 
                                                                           block2,
                                                                           num_classes)
    
    return psnr_values

def plot_heatmap(values, output_path):
    plt.imshow(values, cmap='YlOrRd', interpolation='nearest')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def main():
    os.makedirs('images', exist_ok=True)
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    torch.cuda.set_device(config['device'])
    model_name = config['model']['name']
    dataset_name = config['dataset']['args']['type']
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
    # pred_img = predictions_to_colored_image(predictions, dataset_name)
    # save_colored_image(pred_img, f'images/{model_name}_{dataset_name}.png')
    
    # # Calculate and save MIOU heatmap
    block_size = 10
    gt_colored = predictions_to_colored_image(np.argmax(gt, axis=-1), dataset_name)
    pred_colored = predictions_to_colored_image(predictions.numpy(), dataset_name)
    psnr_values = psnr_blocks(gt_colored, pred_colored, block_size)
    plot_heatmap(psnr_values, f'images/{model_name}_{dataset_name}_psnr_heatmap.png')

    
main()