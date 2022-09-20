import numpy as np
from codes.hyperspectral_image_processing.quality_metrics.SpectAngMapper import SpectAngMapper
from codes.hyperspectral_image_processing.quality_metrics.img_qi import img_qi
from codes.hyperspectral_image_processing.quality_metrics.cal_ssim import cal_ssim
from codes.hyperspectral_image_processing.quality_metrics.CC import CC
from codes.hyperspectral_image_processing.quality_metrics.csnr import csnr

def quality_assessment(ground_truth, estimated, ignore_edges, ratio_ergas):
    # quality_assessment - Computes a number of quality indices from the remote sensing literature,
    #    namely the RMSE, ERGAS, SAM and UIQI indices. UIQI is computed using
    #    the code by Wang from "A Universal Image Quality Index" (Wang, Bovik).
    #
    # ground_truth - the original image (3D image),
    # estimated - the estimated image,
    # ignore_edges - when using circular convolution (FFTs), the borders will
    #    probably be wrong, thus we ignore them,
    # ratio_ergas - parameter required to compute ERGAS = h/l, where h - linear
    #    spatial resolution of pixels of the high resolution image, l - linear
    #    spatial resolution of pixels of the low resolution image (e.g., 1/4)
    #
    #   For more details on this, see Section V.B. of
    #
    #   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot,
    #        �A convex formulation for hyperspectral image superresolution
    #        via subspace-based regularization,?IEEE Trans. Geosci. Remote
    #        Sens., to be publised.

    #
    # Author: Miguel Simoes
    #
    # Version: 1
    #
    # Can be obtained online from: https://github.com/alfaiate/HySure
    #
    # % % % % % % % % % % % %
    #
    # Copyright (C) 2015 Miguel Simoes
    #
    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    ##
    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <http://www.gnu.org/licenses/>.

    # Ignore borders
    y = ground_truth[ignore_edges:-ignore_edges, ignore_edges:-ignore_edges, :]
    x = estimated[ignore_edges:-ignore_edges, ignore_edges:-ignore_edges, :]

    # Size, bands, samples
    sz_x = x.shape
    n_bands = sz_x[2]
    n_samples = sz_x[0]*sz_x[1]

    # RMSE
    aux = np.sum(np.sum((x - y)**2, 0), 1)/n_samples
    rmse_per_band = np.sqrt(aux)
    rmse = np.sqrt(np.sum(aux, 2)/n_bands)

    # ERGAS
    mean_y = np.sum(np.sum(y, 0), 1)/n_samples
    ergas = 100*ratio_ergas*np.sqrt(np.sum((rmse_per_band / mean_y)**2)/n_bands)

    # SAM
    # spectral loss
    nom_top = np.sum(np.multiply(x, y),0)
    nom_pred = np.sqrt(np.sum(np.power(x, 2),0))
    nom_true = np.sqrt(np.sum(np.power(y, 2),0))
    nom_base = np.multiply(nom_pred, nom_true)
    angle = np.arccos(np.divide(nom_top, (nom_base)))
    angle = np.nan_to_num(angle)
    sam = np.mean(angle)*180.0/3.14159
    # num = np.sum(x * y, 2)
    # den = np.sqrt(sum(x**2, 2) * np.sum(y**2, 2))
    # sam = np.sum(np.acosd(num / den))/(n_samples)

    # UIQI - calls the method described in "A Universal Image Quality Index"
    # by Zhou Wang and Alan C. Bovik
    q_band = np.zeros(n_bands)
    for idx1 in range(n_bands):
        q_band[idx1]=img_qi(ground_truth[:,:,idx1], estimated[:,:,idx1], 32)
    uiqi = np.mean(q_band)
    ssim = cal_ssim(ground_truth, estimated,0,0)
    DD = np.linalg.norm(ground_truth[:]-estimated[:],1)/ground_truth.size
    CCS = CC(ground_truth,estimated)
    CCS = np.mean(CCS)
    psnr=csnr(ground_truth, estimated,0,0)

    return psnr, rmse, ergas, sam, uiqi, ssim, DD, CCS