import numpy as np

def LHS_K(K0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3):

    p, q = K0.shape
    K0 = K0.T
    K0 = np.reshape(K0,(169,1))
    K = D.T @ D @ K0[:]
    K0 = np.reshape(K0,(13,13))
    K = np.reshape(K, [p, q])
    K = mu * K + (tau + mu) * K0
    K = K.T
    K0 = np.reshape(K0,(169,1))
    temp = np.zeros((N1, N2))
    temp = np.reshape(temp,(N1*N2,1))
    temp[ind[:]] = K0[:]
    temp = np.reshape(temp,(N1,N2))
    Fk = np.fft.fft2(temp)
    Fk = Fk.T
    K0 = np.reshape(K0,(13,13))
    for band in range(N3):

        Fx = FX[:, band]
        Fx = np.reshape(Fx, [N1, N2])
        Fx=Fx.T
        temp = np.real(np.fft.ifft2(np.multiply(Fx,Fk)))

        y = np.zeros((N1, N2))
        temp = temp.flatten()
        y = y.flatten()
        y[IND[:]] = temp[IND[:]]
        y = np.reshape(y,(N1,N2))
        Fy = np.fft.fft2(y)

        Fx=Fx.T
        temp = np.real(np.fft.ifft2(np.multiply(np.conj(Fx),Fy)))
        #temp = temp.T
        temp = temp.flatten()
        temp = temp[ind[:]]
        
        temp = np.reshape(temp, [p, q])
        K = K + temp

    return K
