# HSI-MSI-Image-Fusion
Hyperspectral-Multispectral Image Fusion

#### Installation
1. ```pip install requirements.txt```

#### Directory structure

```
├── artifacts (contains all the intermediate output files from your experiments)
├── adversity (low-light noisy transformations to input image)
├── motion_code (Contains code for Motion Code based Multi Output Spectral Kernel GP)
├── configs (single place to control all knobs of our experiments)
├── datasets (contains all dataloaders. downloaded dataset is kept in datasets/data)
├── neural_nets (contains code for all our neural networks)
├── train_utls (contains code for all utility scripts for training)
├── noise_sweep.py (file to find best hyperparameters using Bayesian Optimization)
├── train.py
├── train_motioncode.py 
└── notebooks (contains experiments and visualization scripts, useful for tutorial and debugging)
```


## Run experiments

1. Adjust config in configs/
2. Train motion code - 
```sh
python -m train_motioncode.py --config configs/{dataset name}.yaml
```
2. Train main segmentation model

```sh
python -m train.py --config configs/{dataset name}.yaml
```
