# Contents of ~/my_app/streamlit_app.py
import streamlit as st
from PIL import Image
import pandas as pd

def main_page():
    st.markdown("# HSI-MSI-Image-Fusion")
    st.sidebar.markdown("# HSI-MSI-Image-Fusion")
    st.markdown(
        """
        The problem is super resolution for enhanced identification of target regions of interest (TROI). The solution to this problem is to combine two low resolution multispectral and hyperspectral video streams into a single super-resolution stream.
        """
    )
    st.markdown(
        """
        ## Relevant work

        ### Blind Hyperspectral-Multispectral Image Fusion via Graph Laplacian Regularization
       [HackMD summary](https://hackmd.io/@_k8fpxG0SxqYZhe6TwKiVA/SyavZEmFq)

       The paper presents a method for fusing a hyperspectral image (HSI) of low resolution with a multispectral image (MSI) of high resolution to produce a super-resolution image (SRI). The process is similar to the process of hyperspectral pan sharpening, but unlike previous work, it assumes no prior knowledge of the spatial or spectral degradation from the SRI to the HSI. It also doesn’t assume a perfectly aligned HSI and MSI pair. The SRI is assumed to be aligned with the MSI. The image fusion is performed iteratively by fusing the HSI with a graph Laplacian of the MSI. The proposed approach searches for the blur kernel and uses the graph Laplacian defined on the MSI to guide the super-resolution of all the bands of the HSI simultaneously. This approach is able to achieve super-resolution without prior knowledge of the spatial nor spectral degradation.
        """        
    )
    st.markdown(
        """
        #### Experiments

        ##### Paper

        The experiments for this paper were conducted from simulated HSI and MSI pairs generated from the Indian Pines, Salinas, Pavia University and the Western Sichuan dataset.
        
        The graph Laplacian chosen was one that  defines the affinity of pixel vectors using correlations between the overlapping windows, using both spectral and spatial information. Conjugate gradient was used to update the kernel and the HSI, which was implemented by the authors' own solvers. 
        
        Further experiments were done to prove the usefulness of the blind kernel estimation, and graph laplacian regularization, by comparing BGLRF with bicubic interpolation, no GLR and non-blind image fusions. BGLRF was also compared to other related algorithms (dTV, HySure, STEREO) on several metrics for aligned and misaligned blur kernels over several datasets. 
        
        ##### Our implementation

        We implemented the paper on the Indian Pines dataset. We chose the corrected image of shape (145, 145, 200) as our initial HSI. We then denoised it to obtain the SRI (ground truth).  The spatial degradation from SRI to HSI was modeled by a Gaussian blur, followed by downsampling (downsampling ratio=4) and then adding noise of SNR = 30. We performed band selection to select the 16 most informative bands of the SRI, based on the Minimum Noise Band Selection algorithm ([MNBS](https://ieeexplore.ieee.org/document/6812179)). The MSI was generated from the spectral responses obtained from the Landsat_TM5 sensor. In the simulation process, we calculated the wavelength of the selected 16 bands. For each of the 6 spectral responses (blue, green, red, near infrared, short wave infrared 1 and 2) we perform a spline interpolation from the band with the highest wavelength below the given band, and the band with the lowest wavelength above the given band, which gives us the resultant band. The 6 bands together gave the denoised MSI. We added noise of signal to noise ratio (SNR) 40 to produce the final MSI.  The Gaussian blur used in updating the HSI was chosen had 3 settings:
        1. (mis-registration: 0) We take the blur kernel of size 9x9, centered at the origin (ground truth blur kernel).
        2. (mis-registration: 2) We take the blur kernel of radius 6. The center of the blur kernel is shifted horizontally and vertically to the top left by 2 pixels. 
        3. (mis-registration: 4) The center of the blur kernel is shifted horizontally and vertically to the bottom right by 4 pixels (not there in code currently). 

        We denoise the HSI thus generated again before applying the BGLRF fusion algorithm. We choose alpha = 10 in all settings, and beta = 0 for case (1) and beta = 10 for the misregistration cases, and the radius as specified in the different settings. 
        """        
    )
    st.markdown(
        """
        #### Evaluation metrics

        The fused result was evaluated based on the following metrics:

        Metrics based on spatial measures:
        
        1. **ERGAS** (relative dimensionless global error in synthesis)
        2. **UIQI** (universal image quality index)

        Metrics based on spectral measures: 
        1. **SAM** (spectral angle mapper)
        2. **OA** (overall accuracy of classification)
        """        
    )

def page2():
    st.markdown("# Results")
    st.sidebar.markdown("# Results")

    st.markdown("## BGLRF")

    #st.markdown("Original Indian Pines image with 200 bands")

    st.image("./Images/denoise before after.png", caption="(left) Original Indian Pines image with 200 bands (2nd band), (right) SRI generated after denoising the original image (2nd band)")
    st.image("./Images/denoise plot.png", caption="SNR vs time of input Indian Pines image during the denoising process")

    st.markdown("### Mis-registration = 0")
    st.image("./Images/misreg0 hsi.png", caption="Selected bands (every 8th band, starting from 2nd band) of the generated HSI")
    st.image("./Images/MSI bands.png", caption="Generated MSI of 6 bands")
    st.image("./Images/misreg0 sri.png", caption="(left) Original SRI (ground truth) (2nd band), (right) SRI generated by fusing MSI and HSI using BGLRF (2nd band)")

    st.markdown("### Mis-registration = 2")
    st.image("./Images/HSI selected bands.png", caption="Selected bands (every 8th band, starting from 2nd band) of the generated HSI")
    st.image("./Images/MSI bands.png", caption="Generated MSI of 6 bands")
    st.image("./Images/SRI original vs fused.png", caption="(left) Original SRI (ground truth) (2nd band), (right) SRI generated by fusing MSI and HSI using BGLRF (2nd band)")

    st.markdown("### Mis-registration = 4")
    st.image("./Images/misreg4 hsi.png", caption="Selected bands (every 8th band, starting from 2nd band) of the generated HSI")
    st.image("./Images/MSI bands.png", caption="Generated MSI of 6 bands")
    st.image("./Images/misreg4 sri.png", caption="(left) Original SRI (ground truth) (2nd band), (right) SRI generated by fusing MSI and HSI using BGLRF (2nd band)")

    st.markdown("### Result in terms of error metrics")
    df = pd.DataFrame({'Mis-registration': [0,2,4],
    'PSNR': [93.8680,94.2306,93.9089],
    'RMSE': [0.0083,0.0080,0.0082],
    'ERGAS': [0.6511,0.6377,0.6514],
    'SAM': [1.3084,1.2960,1.3190],
    'UIQI': [0.8876,0.8941,0.8957],
    'SSIM': [1.0000,1.0000,1.0000],
    'DD': [0.0048,0.0046,0.0048],
    'CCS': [0.9296,0.9341,0.9336]}
    )
    st.table(df)


page_names_to_funcs = {
    "Main Page": main_page,
    "Results": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()