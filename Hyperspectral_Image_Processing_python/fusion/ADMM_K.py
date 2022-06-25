import numpy as np
# from codes.hyperspectral_image_processing.fusion.CG_K import CG_K
# from codes.hyperspectral_image_processing.fusion.soft_isotropic import soft_isotropic

import CG_K
import soft_isotropic

def ADMM_K(RHS_1 ,K0 ,D ,FX ,ind ,IND ,beta ,tau ,N1 ,N2 ,N3):

    mu = 50 * beta # for TV

    p ,q = K0.shape

    K = K0

    gg = D @ K0[:] # vector
    l1 = np.zeros(gg.shape) # vector

    KK = K0
    L2 = np.zeros(p, q)

    err = np.zeros(100, 1)

    for i in range(999):

        RHS_2 = mu * D.T @ (gg+l1)
        RHS_2 = np.reshape(RHS_2, [p, q])
        RHS_2 = RHS_2 + mu * (KK + L2)

        RHS = RHS_1 + RHS_2

        # update K
        K_pre = K
        K = CG_K(RHS, K, D, FX, ind, IND, mu, tau, N1, N2, N3);

        # update g
        g = D @ K[:]
        gg = soft_isotropic(g - l1, beta / (2 * mu))

        # update KK
        k = K[:]-L2[:]

        u = np.flip(np.sort(k)) # projection onto the probability simplex
        for j in range(p * q):
            s = np.sum(u[:j])
            temp = u[j] + 1 / j * (1 - s)
            if temp > 0:
                lam = temp-u(j)
            else:
                break
        k = max(k+ lam, 0)

        KK = np.reshape(k, [p, q])

        err[i] = np.linalg.norm(K - K_pre, ord='fro') / np.linalg.norm(K, ord='fro')

        if err(i) < 1e-4:
            err = err[:i]
            print('ADMM_K is successful in {} iterations \n'.format(i))
            break
        else:
            # update multipliers
            l1 = l1 + gg - g
            L2 = L2 + KK - K

    return K, err