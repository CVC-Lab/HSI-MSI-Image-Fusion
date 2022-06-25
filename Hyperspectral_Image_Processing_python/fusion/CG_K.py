import numpy as np
import LHS_K

def CG_K(H ,K0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3):

    normH = np.norm(H[:]) #Frobenius norm
    tolH = 1e-4 *normH

    r0 = H - LHS_K(K0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3)
    p0 = r0

    for i in range(999):

        pp = LHS_K(p0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3)
        pp1 = p0[:].T @ pp[:]
        a = (r0[:].T @ r0[:])/pp1 # compute alpha_k

        K = K0 + a *p0;
        r1 = r0 - a *pp;

        res = np.norm(r1[:])
        if not res < tolH:
            b1 = res**2 /(r0[:].T @ r0[:]) # compute beta_k
            p1 = r1 + b1 *p0

            p0 = p1
            r0 = r1
            K0 = K

    return K