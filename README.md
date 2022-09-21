# HSI-MSI-Image-Fusion
Hyperspectral-Multispectral Image Fusion

#### Installation
1. ```conda env create -f environment.yaml```
2. ```conda activate hsi```


#### CVC_Tianming
Original MATLAB codes, description given in overview.pdf. Fusion code in the folder hyperspectral_image_processing.m

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