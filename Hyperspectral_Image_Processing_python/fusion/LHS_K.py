import numpy as np

#Latin hypercube sampling for kernel
def LHS_K(K0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3):

    p, q = K0.shape

    K = D.T @ D @ K0[:] #multiplying D transpose with D and then with K0
    K = np.reshape(K, [p, q])
    K = mu * K + (tau + mu) * K0

    temp = np.zeros(N1, N2)
    temp[ind] = K0[:]
    Fk = np.fft.fft2(temp) #computes the n-dimensional discrete fourier transform using fft

    for band in range(N3):

        Fx = FX[:, band]
        Fx = np.reshape(Fx, [N1, N2])

        temp = np.real(np.fft.ifft2(Fx * Fk)) #inverse discrete Fourier transform

        y = np.zeros(N1, N2)
        y[IND] = temp[IND]
        Fy = np.fft.fft2(y)

        temp = np.real(np.fft.ifft2(np.conj(Fx) * Fy)) #real part of the dft of the product of conjugate of Fx with Fy
        temp = temp[ind]

        temp = np.reshape(temp, [p, q])
        K = K + temp

    return K
