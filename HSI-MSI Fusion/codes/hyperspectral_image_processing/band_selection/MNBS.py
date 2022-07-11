import numpy as np


def MNBS(Y, YN,n):
    #An Implementation of "A New Band Selection Method for Hyperspectral Image
    # Based on Data Quality"

    ##NOTE FROM CHASE: YN is your best guess on what the noise looks like
    # Independent Component analysis is a good way to get this guess

    S = Y - YN
    Cov_S = S.T @ S
    Cov_N = YN.T @ YN
    N = S.shape[1]
    k = N
    IND = list(range(N))

    while k > n:
        Q = np.zeros(k)

        for i in range(k):

            Temp_S = np.delete(arr=Cov_S,obj=IND[i],axis=0)
            Temp_S = np.delete(arr=Temp_S, obj=IND[i], axis=1)

            # _, TEMP = np.linalg.eig(Temp_S)
            #
            # d_S = np.diag(TEMP)

            TEMP = np.cov(Temp_S)

            d_S = np.linalg.det(TEMP)

            Temp_N = np.delete(arr=Cov_N,obj=IND[i],axis=0)
            Temp_N = np.delete(arr=Temp_N, obj=IND[i], axis=1)

            # _, TEMP = np.linalg.eig(Temp_N)
            #
            # d_N = np.diag(TEMP)

            TEMP = np.cov(Temp_N)

            d_N = np.linalg.det(TEMP)

            ratio = np.sum(np.log10(d_S/d_N))

            Q[i] = ratio

        idx = np.argmax(Q)

        print('Among the {} bands, removed band is {}'.format(k,IND[idx]))

        IND = np.delete(arr=IND, obj=idx,axis=0)

        k = k-1

    return IND