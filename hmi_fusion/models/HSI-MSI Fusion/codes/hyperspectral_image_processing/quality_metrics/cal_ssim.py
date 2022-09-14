import numpy as np
import scipy as scp

def cal_ssim( im1, im2, b_row, b_col):

    h, w, ch = im1.shape
    ssim = 0
    if ch==1:
        ssim = ssim_index(im1[b_row + 1:h - b_row, b_col + 1: w - b_col], im2[b_row + 1: h - b_row, b_col + 1: w - b_col] )
    else:
        for i in range(ch):
            ssim = ssim + ssim_index( im1[b_row+1:h-b_row, b_col+1:w-b_col, i], im2[ b_row+1:h-b_row, b_col+1:w-b_col, i] )
        ssim = ssim / ch

    return ssim


def ssim_index(img1, img2, K=[0.01, 0.03], window=None, L=255):
    #========================================================================
    # SSIM Index, Version 1.0
    # Copyright(c) 2003 Zhou Wang
    # All Rights Reserved.
    #
    # The author was with Howard Hughes Medical Institute, and Laboratory
    # for Computational Vision at Center for Neural Science and Courant
    # Institute of Mathematical Sciences, New York University, USA.He is
    # currently with Department of Electrical and Computer Engineering,
    # University of Waterloo, Canada.
    #
    # ----------------------------------------------------------------------
    # Permission to use, copy, or modify this software and its documentation
    # for educational and research purposes only and without fee is hereby
    # granted, provided that this copyright notice and the original authors'
    # names appear on all copies and supporting documentation. This program
    # shall not be used, rewritten, or adapted as the basis of a commercial
    # software or hardware product without first obtaining permission of the
    # authors.The authors make no representations about the suitability of
    # this software for any purpose.It is provided "as is" without express
    # or implied warranty.
    # ----------------------------------------------------------------------
    #
    # This is an implementation of the algorithm for calculating the
    # StructuraSIMilarity(SSIM) index between two images. Please refer
    # to the following paper:
    #
    # Z.Wang, A.C.Bovik, H.R.Sheikh, and E.P.Simoncelli, "Image
    # quality assessment: From error measurement to structural similarity "
    # IEEE Transactios on Image Processing, vol. 13, no. 4, Apr. 2004.
    #
    # Kindly report any suggestions or corrections to zhouwang @ ieee.org
    #
    # ----------------------------------------------------------------------
    #
    # Input: (1) img1: the first image being compared
    # (2) img2: the second image being compared
    # (3) K: constants in the SSIM index formula(see the above
    # reference).defualt value: K = [0.01 0.03]
    # (4) window: local window for statistics(see the above
    # reference).default widnow is Gaussian given by
    # window = fspecial('gaussian', 11, 1.5);
    # (5) L: dynamic range of the images.default: L = 255
    #
    # Output: (1) mssim: the mean SSIM index value between 2 images.
    # If one of the images being compared is regarded as
    # perfect quality, then mssim can be considered as the
    # quality measure of the other image.
    # If img1 = img2, then mssim = 1.
    # (2) ssim_map: the SSIM index map of the test image.The map
    # has a smaller size than the input images.The actual size:
    # size(img1) - size(window) + 1.
    #
    # Default Usage:
    # Given 2 test images img1 and img2, whose dynamic range is 0 - 255
    #
    # [mssim ssim_map] = ssim_index(img1, img2);
    #
    # Advanced Usage:
    # User defined parameters.For example
    #
    # K = [0.05 0.05];
    # window = ones(8);
    # L = 100;
    # [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
    #
    # See the results:
    #
    # mssim
    # Gives the mssim value
    # imshow(max(0, ssim_map). ^ 4) % Shows the SSIM index map
    #
    #========================================================================


    if not img1.shape == img2.shape:
        raise ValueError('Images must be the same shape')

    M, N = img1.shape

    if M<window.shape[0] or N<window.shape[1]:
        raise ValueError('Image is smaller than the filter')

    if len(K) == 2:
        if K[0]<0 or K[1]<0:
            raise ValueError('K cannot be negative')
    else:
        raise ValueError('K must have two elements')

    if window == None:
        window = matlab_style_gauss2D([11, 11], 1.5)
    else:
        if M < window.shape[0] or N < window.shape[1]:
            raise ValueError('Image is smaller than the filter')
        H, W = window.shape
        if H*W < 4:
            raise ValueError('Filter is too small')


    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2

    window = window/np.sum(window)
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')

    mu1 = scp.signal.convolve(window, img1, 'valid')
    mu2 = scp.signal.convolve(window, img2, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = scp.signal.convolve(window, img1 * img1, 'valid') - mu1_sq
    sigma2_sq = scp.signal.convolve(window, img2 * img2, 'valid') - mu2_sq
    sigma12 = scp.signal.convolve(window, img1 * img2, 'valid') - mu1_mu2

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


if __name__=='__main__':
    h = matlab_style_gauss2D([11,11],1.5)
    print('pause')