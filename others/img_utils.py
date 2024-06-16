#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015 IAS / CNRS / Univ. Paris-Sud
# BSD License - see attached LICENSE file
# Author: Alexandre Boucaud <alexandre.boucaud@ias.u-psud.fr>

"""
PyPHER - Python-based PSF Homogenization kERnels
================================================
Compute the homogenization kernel between two PSFs
Usage:
  pypher psf_source psf_target output
         [-s ANGLE_SOURCE] [-t ANGLE_TARGET] [-r REG_FACT]
  pypher (-h | --help)
Example:
  pypher psf_a.fits psf_b.fits kernel_a_to_b.fits -r 1.e-5
"""
from __future__ import absolute_import, print_function, division

import os
import sys
import logging
import logging.handlers
import argparse
import numpy as np

from scipy.ndimage import rotate, zoom
import torch
import numpy as np

################
# IMAGE METHODS
################
def imrotate(image, angle, interp_order=1):
    """
    Rotate an image from North to East given an angle in degrees
    Parameters
    """
    return rotate(image, -1.0 * angle,
                  order=interp_order, reshape=False, prefilter=False)


def imresample(image, source_pscale, target_pscale, interp_order=1):
    """
    Resample data array from one pixel scale to another
    The resampling ensures the parity of the image is conserved
    to preserve the centering.
    """
    old_size = image.shape[0]
    new_size_raw = old_size * source_pscale / target_pscale
    new_size = int(np.ceil(new_size_raw))

    if new_size > 10000:
        raise MemoryError("The resampling will yield a too large image. "
                          "Please resize the input PSF image.")

    # Chech for parity
    if (old_size - new_size) % 2 == 1:
        new_size += 1

    ratio = new_size / old_size

    return zoom(image, ratio, order=interp_order) / ratio**2


def trim(image, shape):
    """
    Trim image to a given shape
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("TRIM: null or negative shape given")

    dshape = imshape - shape
    if np.any(dshape < 0):
        raise ValueError("TRIM: target size bigger than source one")

    if np.any(dshape % 2 != 0):
        raise ValueError("TRIM: source and target shapes "
                         "have different parity")

    idx, idy = np.indices(shape)
    offx, offy = dshape // 2

    return image[idx + offx, idy + offy]


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


##########
# FOURIER
##########


def udft2(image):
    """Unitary fft2"""
    norm = np.sqrt(image.size)
    return np.fft.fft2(image) / norm


def uidft2(image):
    """Unitary ifft2"""
    norm = np.sqrt(image.size)
    return np.fft.ifft2(image) * norm


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


################
# DECONVOLUTION
################

LAPLACIAN = np.array([[ 0, -1,  0],
                      [-1,  4, -1],
                      [ 0, -1,  0]])


def deconv_wiener(psf, reg_fact):
    r"""
    Create a Wiener filter using a PSF image
    """
    # Optical transfer functions
    trans_func = psf2otf(psf, psf.shape)
    reg_op = psf2otf(LAPLACIAN, psf.shape)

    wiener = np.conj(trans_func) / (np.abs(trans_func)**2 +
                                    reg_fact * np.abs(reg_op)**2)

    return wiener


def homogenization_kernel(psf_target, psf_source, reg_fact=1e-4, clip=True):
    r"""
    Compute the homogenization kernel to match two PSFs
    """
    wiener = deconv_wiener(psf_source, reg_fact)

    kernel_fourier = wiener * udft2(psf_target)
    kernel_image = np.real(uidft2(kernel_fourier))

    if clip:
        kernel_image.clip(-1, 1)

    return kernel_image, kernel_fourier


########
# DEBUG
########


def setup_logger(log_filename='pypher.log'):  # pragma: no cover
    """
    Set up and return a logger
    The logger records the time, modulename, method and message
    """
    # create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    # Add the log message handler to the logger
    handler = logging.handlers.RotatingFileHandler(log_filename)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - '
                                  '%(module)s - '
                                  '%(levelname)s - '
                                  '%(message)s')
    handler.setFormatter(formatter)
    # add handler to logger
    logger.addHandler(handler)

    return logger


def to_torch_sparse(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
