import skimage.transform
import os
from skimage.measure import block_reduce
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from matplotlib import pyplot as plt

def compare_mpsnr(x_true, x_pred):

    x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    channels = x_true.shape[2]
    total_psnr = [peak_signal_noise_ratio(x_true[:, :, k], x_pred[:, :, k], data_range=np.max(x_true[:,:,k]) - np.min(x_true[:,:,k]))
                  for k in range(channels)]

    return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, multichannel=True):

    channels = x_true.shape[2]
    x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    mssim = [structural_similarity(x_true[:, :, i], x_pred[:, :, i], multichannel=multichannel)
            for i in range(channels)]

    return np.mean(mssim)


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
    return rmse

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

    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi

    return sam_deg


def compare_ergas(x_true, x_pred, ratio):

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
