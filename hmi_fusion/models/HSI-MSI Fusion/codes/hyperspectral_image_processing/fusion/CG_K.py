import numpy as np
from codes.hyperspectral_image_processing.fusion.LHS_K import LHS_K

def CG_K(H ,K0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3):

    normH = np.linalg.norm(H[:])
    tolH = 1e-4 *normH

    r0 = H - LHS_K(K0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3)
    p0 = r0.copy()

    for i in range(999):

        pp = LHS_K(p0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3)
        pp = pp.flatten()
        p0 = p0.flatten()
        r0 = r0.flatten()
        pp1 = p0[:].T @ pp[:]
        a = (r0[:].T @ r0[:])/pp1 # compute alpha_k
        p0 = np.reshape(p0,(13,13))
        K = K0 + a *p0;
        pp = np.reshape(pp,(13,13))
        r0 = np.reshape(r0,(13,13))
        r1 = r0 - a *pp;
        
        res = np.linalg.norm(r1[:])
        
        if res < tolH:
            break
        else:
            r0 = r0.flatten()
            b1 = res**2 /(r0[:].T @ r0[:]) # compute beta_k
            r0 = np.reshape(r0,(13,13))
            p1 = r1 + b1 *p0

            p0 = p1
            r0 = r1
            K0 = K
            

    return K