import numpy as np

def add_noise(M, SNR):

    N1, N2, N3 = M.shape

    signal_power = np.linalg.norm(M[:])**2 / (N1 * N2 * N3)
    noise_power = signal_power/(10**(SNR/10))
    noise_tenM = np.sqrt(noise_power) * np.random.randn(N1,N2,N3)

    M = M+noise_tenM

    return M, noise_tenM