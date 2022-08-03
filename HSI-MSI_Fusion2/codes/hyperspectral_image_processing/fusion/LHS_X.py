import numpy as np
import scipy
import matplotlib.pyplot as plt

def LHS_X(X0,Fk,L,IND,N1,N2,N3,tau):

    X = X0.copy()

    for band in range(N3):

        x = X0[:,band]

        x = np.reshape(x,[N1, N2])
        x = x.T
        Fx = np.fft.fft2(x)

        x = np.real(np.fft.ifft2(Fk*Fx))
        x = x.T
        x = x.flatten()
        temp = np.zeros((N1*N2,1))
        temp = temp.flatten()
        temp[IND[:]] = x[IND[:]]
        temp = temp.reshape((N1,N2))
        temp = temp.T
        temp = np.fft.fft2(temp)
        x = np.real(np.fft.ifft2(np.conj(Fk)*temp))
        x = x.T
        x = x.flatten()
        X[:,band] = x[:]


    
    X = X + L @ X0
    X = X+tau*X0

    # print("LHS_X output")
    # plt.imshow(X.reshape(N1,N2,N3))
    # plt.show()
    return X

   