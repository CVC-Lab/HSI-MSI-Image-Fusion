# HSI-MSI-Image-Fusion
Hyperspectral-Multispectral Image Fusion

#### CVC_Tianming
Original MATLAB codes, description given in overview.pdf. Fusion code in the folder hyperspectral_image_processing.m

#### Hyperspectral_Image_Processing_python
Python code for image fusion, with same structure as the matlab fusion codes. I have created demo for denoising, band selection, simulation (HSI-MSI generation) and fusion. 
The demo for denoising and band selection is working, however denoising takes much longer than the corresponding matlab code, and the result doesn't seem to be as good.
The simulation code has been giving bugs especially while implementing the spline. I couldn't try running the fusion demo yet cause the simulation code still has bugs. 

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