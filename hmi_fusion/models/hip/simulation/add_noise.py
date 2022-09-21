import numpy as np
# from skimage.util import random_noise

def add_noise(M, SNR):

    N1, N2, N3 = M.shape

    signal_power = np.linalg.norm(M[:])**2 / (N1 * N2 * N3)
    noise_power = signal_power/(10**(SNR/10))
    noise_tenM = np.sqrt(noise_power) * np.random.randn(N1,N2,N3)

    M = M+noise_tenM

    return M, noise_tenM


def add_noise_gaussian(M, SNR):

    N1, N2, N3 = M.shape
    # random_noise(M, mode='gaussian', mean=0, var=0.05, clip=True)
    signal_power = np.linalg.norm(M[:])**2 / (N1 * N2 * N3)
    noise_power = signal_power/(10**(SNR/10))
    noise_tenM = np.sqrt(noise_power) * np.random.standard_normal(N1,N2,N3)
    M = M+noise_tenM
    return M, noise_tenM
