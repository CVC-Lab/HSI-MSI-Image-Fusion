# HSI-MSI-Image-Fusion
Hyperspectral-Multispectral Image Fusion

#### Installation
1. ```conda env create -f environment.yaml```
2. ```conda activate hsi```

#### Directory structure
`hmi_fusion` is our library that houses all our benchmarks and our own proposed model
```
├── artifacts (contains all the intermediate output files from your experiments)
├── datasets (contains all dataloaders. downloaded dataset is kept in datasets/data)
├── models (contains all benchmark codes and our proposed solution)
└── notebooks (contains experiments and visualization scripts, useful for tutorial and debugging)
```

#### Replicating our results
Make sure you have recursively cloned all submodules of this repository. 
1. Navigate to stick_breaking_vae - `cd models/stick_breaking_vae`
2. To run model training do - `python -m experiments.run_experiments_pytorch`


## Replicating benchmarks

TBD