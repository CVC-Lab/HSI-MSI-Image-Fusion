import numpy as np
from .CG_K import CG_K
from .soft_isotropic import soft_isotropic

def ADMM_K(RHS_1 ,K0 ,D ,FX ,ind ,IND ,beta ,tau ,N1 ,N2 ,N3):

    mu = 50 * beta # for TV

    p ,q = K0.shape

    K = K0
    K0 = np.reshape(K0,(169,1))
    gg = D @ K0 # vector
    l1 = np.zeros(gg.shape) # vector
    K0 = np.reshape(K0,(13,13))
    kK = K0
    L2 = np.zeros((p, q))

    err = np.zeros((500, 1))
    # print(err)

    for i in range(500):

        rHS_2 = mu * D.T @ (gg+l1)
        rHS_2 = np.reshape(rHS_2, [p, q])
        
        rHS_2 = rHS_2 + mu * (kK + L2)

        rHS = RHS_1 + rHS_2

        # update K
        K_pre = K
        K = CG_K(rHS, K, D, FX, ind, IND, mu, tau, N1, N2, N3);

        # update g
        
        g = D @ np.reshape(K,(169,1))
        gg = soft_isotropic(g - l1, beta / (2 * mu))

        # update KK
        k = K[:]-L2[:]
        k=k.T
        k =k.flatten()
        u = np.flip(np.sort(k)) # projection onto the probability simplex
        for j in range(p * q):
            s = np.sum(u[:j+1])
            #print('summing array', u[:j+1])
            temp = u[j] + 1 / (j+1) * (1 - s)
            #print(temp)
            if temp > 0:
                lam = temp-u[j]
            else:
                break
        zer = np.zeros((169,1))
        k = np.reshape(k,(169,1))
        k = np.maximum(k+ lam, zer)

        kK = np.reshape(k, [p, q])
        kK =kK.T

        err[i] = np.linalg.norm(K - K_pre) / np.linalg.norm(K)
        # print(err[i])
        if err[i] < 1e-4:
            err = err[:i]
            print('ADMM_K is successful in {} iterations \n'.format(i))
            break
        elif i!=0 and (err[i-1]-err[i])==0:
            break
        else:
            # update multipliers
            l1 = l1 + gg - g
            L2 = L2 + kK - K

    return K, err