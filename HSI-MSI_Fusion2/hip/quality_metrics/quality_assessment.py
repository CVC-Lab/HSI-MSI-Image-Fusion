import numpy as np
from sklearn.metrics import mean_squared_error
import sys
from math import acos
from .img_qi import img_qi
import sewar.full_ref as sf
from . import cal_ssim
from . import SpectAngMapper
# from codes.hyperspectral_image_processing.quality_metrics.SpectAngMapper import SpectAngMapper
# from codes.hyperspectral_image_processing.quality_metrics.img_qi import img_qi
# from codes.hyperspectral_image_processing.quality_metrics.cal_ssim import cal_ssim
# from codes.hyperspectral_image_processing.quality_metrics.CC import CC
# from codes.hyperspectral_image_processing.quality_metrics.csnr import csnr


# credits to https://github.com/up42/image-similarity-measures/blob/0e9227e2bfa0fe3c250f75fcfeda657ff1857158/image_similarity_measures/quality_metrics.py#L14
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
    # y = ground_truth[ignore_edges:-(ignore_edges+1), ignore_edges:-(ignore_edges+1), :]
    # x = estimated[ignore_edges:-ignore_edges, ignore_edges:-ignore_edges, :]

    y = ground_truth[:, :, :]
    x = estimated[:, :, :]

    # Size, bands, samples
    sz_x = x.shape
    n_bands = sz_x[2]
    n_samples = sz_x[0]*sz_x[1]

    print(x.shape)
    print(y.shape)

    # RMSE
    # aux = np.sum(np.sum((x - y)**2, axis=0), axis=1)/n_samples
    # print(aux.shape)

    # aux = np.sum((x - y)**2, axis=(0,1))/n_samples
    # print(aux.shape)

    # # print(np.sum((x - y)**2, 0).shape)
    # rmse_per_band = np.sqrt(aux)
    # rmse = np.sqrt(np.sum(aux, 2)/n_bands)

    rmse_per_band = []
    mse_per_band = []
    for i in range(n_bands):
        mse = mean_squared_error(x[:,:,i],y[:,:,i].T)
        mse_per_band.append((mse))
        rmse_per_band.append(np.sqrt(mse))
    rmse = np.sum(rmse_per_band)/n_bands
    mse = np.sum(mse_per_band)/n_bands
    print("RMSE", rmse)

    # ERGAS
    # mean_y = np.sum(y, axis =(0,1))/n_samples
    mean_y = np.sum(np.sum(y, 0), 0)/n_samples
    # print(mean_y.shape)
    ergas = 100*ratio_ergas*np.sqrt(np.sum((rmse_per_band / mean_y)**2)/(n_bands))
    # ergas = sf.ergas(ground_truth, estimated, ratio_ergas)
    print("ERGAS", ergas)

    # # # SAM
    # # sam= SpectAngMapper( ground_truth, estimated )
    # # sam=sam*180/np.pi

    # eps = sys.float_info.epsilon
    # tmp = (np.sum(ground_truth * estimated,axis=2) + eps) / (np.sqrt(np.sum(ground_truth**2,axis=2)) + eps) / (np.sqrt(np.sum(estimated**2,axis=2)) + eps)

    # # print(tmp.shape)
    # tmp = np.reshape(tmp, (tmp.shape[0]*tmp.shape[1]))
    # arccos= []
    # for i in range(tmp.size):
    #     arccos.append(acos(tmp[i]))
    
    # sam = np.real(arccos)
    # sam = np.mean(sam)
    # sam=sam*180/np.pi
    # # sam = sf.sam(ground_truth, estimated)
    # # sam = sam*180/np.pi

    sam = SpectAngMapper.sam(ground_truth, estimated)
    print("SAM", sam)


    # num = np.sum(x * y, 2)
    # den = np.sqrt(sum(x**2, 2) * np.sum(y**2, 2))
    # sam = np.sum(np.acosd(num / den))/(n_samples)

    # UIQI - calls the method described in "A Universal Image Quality Index"
    # by Zhou Wang and Alan C. Bovik
    # q_band = np.zeros(n_bands)
    # for idx1 in range(n_bands):
    #     q_band[idx1]=img_qi(ground_truth[:,:,idx1], estimated[:,:,idx1].T, 32)
    # uiqi = np.mean(q_band)
    uiqi = sf.uqi(ground_truth, estimated)
    print("UIQI", uiqi)

    # ssim = sf.ssim(ground_truth, estimated)
    ssim = cal_ssim.ssim(ground_truth, estimated, 9604)
    print("SSIM", ssim)
    # DD = np.linalg.norm(ground_truth[:]-estimated[:],1)/ground_truth.size
    # CCS = CC(ground_truth,estimated)
    # CCS = np.mean(CCS)
    # psnr=csnr(ground_truth, estimated,0,0)
    s = 10 * np.log10(255**2/mse)
    print("PSNR ", s)

    # s = sf.psnr(ground_truth, estimated)
    # print("PSNR ", s)

    # return psnr, rmse, ergas, sam, uiqi, ssim, DD, CCS
    return