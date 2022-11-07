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
1. Download cave_dataset
2. In cave_dataset.py set ``save_laplacians_flag = True``. 
3. From hmi_fusion run ``python -m datasets.cave_dataset``. After completion set ``save_laplacians_flag = False``
2. Navigate to hmi_fusion - ``python -m models.dbglrf.train``


## Replicating benchmarks

TBD