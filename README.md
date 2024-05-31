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
Follow instructions to setup huggingface-cli as per this wiki. Go to `https://huggingface.co/cvc-lab` to checkout datasets and files in each dataset.
Available datasets for this repository -
1. Pavia - `cvc-lab/PaviaUniversity`
2. Chikusei - `cvc-lab/chikusei`
3. Urban - `cvc-lab/urban`
4. DC Mall - `cvc-lab/dc_mall`
5. Jasper Ridge - `cvc-lab/jasper_ridge`
6. CAVE - `cvc-lab/cave`
7. Harvard - `cvc-lab/harvard`

Example script to download a particular file for CAVE dataset - 

Navigate to `hmi_fusion/data/CAVE/` and run the following for downloading CAVE -
```python
from huggingface_hub import login
from huggingface_hub import hf_hub_download
import os
login(token="<your_token>")

os.environ["HF_DATASETS_CACHE"] = "hmi_fusion/data"
hf_hub_download(repo_id="cvc-lab/CAVE", filename="complete_ms_data.zip", repo_type="dataset")

```






#### Replicating our results
Make sure you have recursively cloned all submodules of this repository. 
1. Download cave_dataset
2. In cave_dataset.py set ``save_laplacians_flag = True``. 
3. From hmi_fusion run ``python -m datasets.cave_dataset``. After completion set ``save_laplacians_flag = False``
2. Navigate to hmi_fusion - ``python -m models.dbglrf.train_new``


## Replicating benchmarks

TBD


## Data sources -
1. https://rslab.ut.ac.ir/data
2. https://naotoyokoya.com/Download.html