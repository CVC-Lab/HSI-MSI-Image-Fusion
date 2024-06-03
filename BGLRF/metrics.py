from skimage.measure import block_reduce
import numpy as np
from skimage.metrics import mean_squared_error
from matplotlib import pyplot as plt
import scipy
import sys
from math import acos
import torch

def ErrRelGlobAdimSyn(imagery1, imagery2):
    #==========================================================================
    # Evaluates Erreur Relative Globale Adimensionnelle de Synth®®se(ERGAS,
    # dimensionless global relative error of synthesis)[1] for two MSIs.
    #==========================================================================
    m, n, k = imagery1.shape
    mm, nn, kk = imagery2.shape
    m = min(m,mm)
    n = min(n,nn)
    k = min(k,kk)
    imagery1 = imagery1[:m,:n,:k]
    imagery2 = imagery2[:m,:n,:k]

    ergas = 0
    for i in range(k):
        ergas = ergas + np.sum(imagery1[:,:, i] - imagery2[:,:, i])**2 / (np.mean(imagery1[:,:, i])*k)
    return 100 * np.sqrt(ergas)

def csnr(A,B,row,col):
    n, m, ch = A.shape
    summa = 0
    e = A-B
    if ch == 1:
        e = e[row+1:n-row,col+1:m-col]
        me = np.mean(e**2)
        s = 10 * np.log10(255**2/me)
    else:
        for i in range(ch):
            e1 = e[row+1:n-row,col+1:m-col,i]
            mse = np.mean(e1**2)
            s = 10 * np.log10(255**2/mse)
            summa += s
        s = summa/ch

    return s

def CC(ref,tar,mask=None):
    # Cross Correlation
    rows, cols, bands = tar.shape
    out = np.zeros(bands)

    if mask is None:
        for i in range(bands):
            tar_tmp = tar[:,:,i]
            ref_tmp = ref[:,:,i]
            cc = np.corrcoef(tar_tmp, ref_tmp)
            out[i] = cc[1,2]
    else:
        mask, _ = np.where(not mask == 0)
        for i in range(bands):
            tar_tmp = tar[:, :, i]
            ref_tmp = ref[:, :, i]
            cc = np.corrcoef(tar_tmp[mask], ref_tmp[mask])
            out[i] = cc[1,2]

    return np.mean(out)

def getrmse(x, y):
    wid_x, hi_x, n_bands = x.shape
    n_samples = wid_x * hi_x

    aux = np.sum((x-y)**2)/(n_samples * n_bands)
    rmse_total = np.sqrt(aux)

    return rmse_total

def img_qi(img1, img2, block_size=8):
    # %This is an efficient implementation of the algorithm for calculating
    # %the universal image quality index proposed by Zhou Wang and Alan C.
    # %Bovik. Please refer to the paper "A Universal Image Quality Index"
    # %by Zhou Wang and Alan C. Bovik, published in IEEE Signal Processing
    # %Letters, 2001. In order to run this function, you must have Matlab's
    # %Image Processing Toobox.
    # %Input : an original image and a test image of the same size
    # %Output: (1) an overall quality index of the test image, with a value
    # %            range of [-1, 1].
    # %        (2) a quality map of the test image. The map has a smaller
    # %            size than the input images. The actual size is
    # %            img_size - BLOCK_SIZE + 1.
    # %========================================================================

    if not img1.shape == img2.shape:
        raise ValueError('Images must be the same size')

    N = block_size**2

    sum2_filter = np.ones(block_size)

    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img12 = img1 * img2

    img1_sum = scipy.signal.convolve(sum2_filter, img1, mode='valid')
    img2_sum = scipy.signal.convolve(sum2_filter, img2, mode='valid')
    img1_sq_sum = scipy.signal.convolve(sum2_filter, img1_sq, mode='valid')
    img2_sq_sum = scipy.signal.convolve(sum2_filter, img2_sq, mode='valid')
    img12_sum = scipy.signal.convolve(sum2_filter, img12, mode='valid')

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    quality_map = np.ones(denominator.shape)
    index = (denominator1 == 0) and (not img12_sq_sum_mul == 0)
    quality_map[index] = 2*img12_sum_mul[index]/img12_sq_sum_mul[index]

    index = not (denominator == 0)
    quality_map[index] = numerator[index]/denominator[index]

    quality = np.mean(quality_map)

    return quality, quality_map

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def ssim_index(img1, img2, K=[0.01, 0.03], window=None, L=255):
    #========================================================================
    # This is an implementation of the algorithm for calculating the
    # StructuraSIMilarity(SSIM) index between two images. Please refer
    # to the following paper:
    # Z.Wang, A.C.Bovik, H.R.Sheikh, and E.P.Simoncelli, "Image
    # quality assessment: From error measurement to structural similarity "
    # IEEE Transactios on Image Processing, vol. 13, no. 4, Apr. 2004.
    if not img1.shape == img2.shape:
        raise ValueError('Images must be the same shape')

    M, N = img1.shape

    if window is None:
        window = matlab_style_gauss2D([11, 11], 1.5)

    else:
        if M < window.shape[0] or N < window.shape[1]:
            raise ValueError('Image is smaller than the filter')
        H, W = window.shape
        if H*W < 4:
            raise ValueError('Filter is too small')

    if len(K) == 2:
        if K[0]<0 or K[1]<0:
            raise ValueError('K cannot be negative')
    else:
        raise ValueError('K must have two elements')

    
    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2

    window = window/np.sum(window)
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')

    mu1 = scipy.signal.convolve(window, img1, 'valid')
    mu2 = scipy.signal.convolve(window, img2, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = scipy.signal.convolve(window, img1 * img1, 'valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve(window, img2 * img2, 'valid') - mu2_sq
    sigma12 = scipy.signal.convolve(window, img1 * img2, 'valid') - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones(mu1.shape)
        index = []
        for i in range(len(denominator1)):
            if (denominator1[i] * denominator2[i]) > 0:
                index.append(i)
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = []
        for i in range(len(denominator1)):
            if (not denominator1[i]==0) and (not denominator2[i]==0):
                index.append(i)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)

    return mssim

def cal_ssim( im1, im2, b_row, b_col):
    h, w, ch = im1.shape
    ssim = 0
    if ch==1:
        ssim = ssim_index(im1[b_row + 1:h - b_row, b_col + 1: w - b_col], im2[b_row + 1: h - b_row, b_col + 1: w - b_col])
    else:
        for i in range(ch):
            ssim = ssim + ssim_index(im1[b_row+1:h-b_row, b_col+1:w-b_col, i], im2[ b_row+1:h-b_row, b_col+1:w-b_col, i])
        ssim = ssim / ch

    return ssim

def SpectAngMapper(imagery1, imagery2):
    eps = sys.float_info.epsilon
    tmp = (np.sum(imagery1 * imagery2,axis=2) + eps) / (np.sqrt(np.sum(imagery1**2,axis=2)) + eps) / (np.sqrt(np.sum(imagery2**2,axis=2)) + eps)
    sam = np.mean(np.real(acos(tmp)))

    return sam

def quality_assessment(ground_truth, estimated, ignore_edges, ratio_ergas):
    # quality_assessment - Computes a number of quality indices from the remote sensing literature,
    #    namely the RMSE, ERGAS, SAM and UIQI indices. UIQI is computed using
    #    the code by Wang from "A Universal Image Quality Index" (Wang, Bovik).

    # Ignore borders
    y = ground_truth[ignore_edges:-ignore_edges, ignore_edges:-ignore_edges, :]
    x = estimated[ignore_edges:-ignore_edges, ignore_edges:-ignore_edges, :]
    if ignore_edges == 0:
        y = ground_truth
        x = estimated

    # Size, bands, samples
    sz_x = x.shape
    n_bands = sz_x[2]
    n_samples = sz_x[0]*sz_x[1]

    # RMSE
    aux = np.sum(np.sum((x - y)**2, 0), 1)/n_samples
    rmse_per_band = np.sqrt(aux)
    rmse = np.sqrt(np.sum(aux)/n_bands)

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
    #for idx1 in range(n_bands):
    #    q_band[idx1]=img_qi(ground_truth[:,:,idx1], estimated[:,:,idx1], 32)
    uiqi = np.mean(q_band)
    ssim = cal_ssim(ground_truth, estimated,0,0)
    DD = np.linalg.norm(ground_truth.reshape(-1)-estimated.reshape(-1), 1)/ground_truth.size
    CCS = CC(ground_truth,estimated)
    CCS = np.mean(CCS)
    psnr=csnr(ground_truth, estimated,0,0)

    return psnr, rmse, ergas, sam, uiqi, ssim, DD, CCS

### WHICH ONE IS TRUE VERSION????

def compare_mpsnr(x_true, x_pred, mse):
    total_psnr = 10 * np.log10(255**2/mse)

    # x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    # channels = x_true.shape[2]
    # total_psnr = [peak_signal_noise_ratio(x_true[:, :, k], x_pred[:, :, k], data_range=np.max(x_true[:,:,k]) - np.min(x_true[:,:,k]))
    #               for k in range(channels)]

    return np.mean(total_psnr)

def compare_mssim(x_true, x_pred, multichannel=True):

    # channels = x_true.shape[2]
    # x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    # mssim = [structural_similarity(x_true[:, :, i], x_pred[:, :, i], multichannel=multichannel)
    #         for i in range(channels)]
    return ssim_index(x_true, x_pred, 19000)
    # return np.mean(mssim)

def find_rmse(img_true, img_pred):
    rmse_per_band = []
    mse_per_band = []
    n_bands = img_true.shape[-1]
    for i in range(n_bands):
        mse = mean_squared_error(img_true[:,:,i],img_pred[:,:,i].T)
        mse_per_band.append((mse))
        rmse_per_band.append(np.sqrt(mse))
    rmse = np.sum(rmse_per_band)/n_bands
    mse = np.sum(mse_per_band)/n_bands
    return rmse, mse, rmse_per_band

    # ref = img_tar * 255.0
    # tar = img_hr * 255.0
    # lr_flags = tar < 0
    # tar[lr_flags] = 0
    # hr_flags = tar > 255.0
    # tar[hr_flags] = 255.0

    # diff = ref - tar
    # size = ref.shape
    # rmse = np.sqrt(np.sum(np.sum(np.power(diff, 2))) / (size[0] * size[1]*size[2]))

    # return rmse

def compare_sam(x_true, x_pred):
    return SpectAngMapper(x_true, x_pred)
    # num = 0
    # sum_sam = 0
    # x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    # for x in range(x_true.shape[0]):
    #     for y in range(x_true.shape[1]):
    #         tmp_pred = x_pred[x, y].ravel()
    #         tmp_true = x_true[x, y].ravel()
    #         if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
    #             sum_sam += np.arccos(
    #                 np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
    #             num += 1
    # sam_deg = (sum_sam / num) * 180 / np.pi

def compare_ergas(x_true, x_pred, ratio, rmse_per_band):
    # n_s, n_s, n_bands = x_true.shape
    # n_samples = n_s * n_s
    # mean_y = np.sum(np.sum(x_true, 0), 0)/n_samples
    # ergas = 100*ratio*np.sqrt(np.sum((rmse_per_band / mean_y)**2)/(n_bands))
    # return ergas
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp

    return (100 / ratio) * (np.sqrt(sum_ergas / x_true.shape[0]))

def img_2d_mat(x_true, x_pred):

    h, w, c = x_true.shape
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    x_mat = np.zeros((c, h * w), dtype=np.float32)
    y_mat = np.zeros((c, h * w), dtype=np.float32)
    for i in range(c):
        x_mat[i] = x_true[:, :, i].reshape((1, -1),order='F')
        y_mat[i] = x_pred[:, :, i].reshape((1, -1), order='F')

    return x_mat, y_mat

def evaluate_metrics(inX, x_solver, sf):
    av_psnr = compare_mpsnr(inX, x_solver)
    av_ssim = compare_mssim(inX, x_solver, multichannel=True)
    av_sam = compare_sam(inX, x_solver)
    av_rmse = find_rmse(inX, x_solver)
    av_ergas = compare_ergas(inX, x_solver, sf)
    return av_psnr, av_ssim, av_sam, av_rmse, av_ergas
