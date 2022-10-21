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

#### Running any experiment
Make sure you have recursively cloned all submodules of this repository. 
1. Navigate to stick_breaking_vae - `cd models/stick_breaking_vae`
2. To run model training do - `python -m experiments.run_experiments_pytorch`


#### Hyperspectral_Image_Processing_python
Python code for image fusion, with same structure as the matlab fusion codes. SVM codes still have to be sorted. Please refer to Python Fusion Demo file for steps to rn the fusion code. You may have to alter the path at some places. 

#### Streamlit App
To open the streamlit app, run the following commands
```
cd App
streamlit run streamlit_app.py
```

You need streamlit installed, which you can do with the following command
```
pip install streamlit
```

For running the code on any square dataset, use the the code in HSI-MSI_Fusion2.