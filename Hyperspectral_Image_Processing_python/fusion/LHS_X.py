import numpy as np

def LHS_X(X0,Fk,L,IND,N1,N2,N3,tau):

    X = X0

    for band in range(N3):

        x = X0[:,band]

        x = np.reshape(x,[N1, N2])
        Fx = np.fft.fft2(x)

        x = np.real(np.fft.ifft2(Fk*Fx))

        temp = np.zeros(N1,N2)
        temp[IND] = x[IND]

        temp = np.fft.fft2(temp)
        x = np.real(np.fft.ifft2(np.conj(Fk)*temp))

        X[:,band] = x[:]


    X = X+L*X0
    X = X+tau*X0

    return X