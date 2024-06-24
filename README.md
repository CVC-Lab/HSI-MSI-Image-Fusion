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

### Downloading datasets
Follow instructions to setup huggingface-cli as per this wiki
Navigate to ``hmi_fusion/data/CAVE/` and run the following for downloading CAVE -
```python
from huggingface_hub import login
from huggingface_hub import hf_hub_download
import os
login(token="<your_token>")

os.environ["HF_DATASETS_CACHE"] = "hmi_fusion/data"
hf_hub_download(repo_id="cvc-lab/CAVE", filename="complete_ms_data.zip", repo_type="dataset")

```
### Download pre-trained SAM model
Navigate to models/ and run
```sh
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Run experiments

```sh
python -m scripts.train --config configs/jasper_ridge_experiment.yaml
```


#### Replicating our results
Make sure you have recursively cloned all submodules of this repository. 
1. Download cave_dataset
2. In cave_dataset.py set ``save_laplacians_flag = True``. 
3. From hmi_fusion run ``python -m datasets.cave_dataset``. After completion set ``save_laplacians_flag = False``
2. Navigate to hmi_fusion - ``python -m models.dbglrf.train_new``


## Replicating benchmarks

TBD