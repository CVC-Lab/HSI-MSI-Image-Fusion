import numpy as np
from LHS_X import LHS_X
import matplotlib.pyplot as plt

def CG_X(H,X0,Fk,L,IND,N1,N2,N3,tau):

    normH = np.linalg.norm(H[:])
    tolH = 1e-4 *normH

    r0 = H - LHS_X(X0 ,Fk, L, IND ,N1 ,N2 ,N3,tau)
    p0 = r0.copy()

    for i in range(999):

        pp = LHS_X(p0,Fk,L,IND,N1,N2,N3,tau)
        pp = pp.flatten()
        p0 = p0.flatten()
        r0 = r0.flatten()
        pp1 = p0[:].T @ pp[:]
        a = (r0[:].T @ r0[:])/pp1 # compute alpha_k
        p0 = np.reshape(p0,(N1*N2,N3))
        #p0=p0.T
        X = X0 + a * p0;
        pp = np.reshape(pp,(N1*N2,N3))
        r0 = np.reshape(r0,(N1*N2,N3))
        r1 = r0 - a *pp;
        
        res = np.linalg.norm(r1[:])
        
        if res < tolH:
            print('CG_X is successful in {} iterations \n'.format(i))
            break
        else:
            r0 = r0.flatten()
            b1 = res**2 /(r0[:].T @ r0[:]) # compute beta_k
            r0 = np.reshape(r0,(N1*N2,N3))
            p1 = r1 + b1 *p0

            p0 = p1
            r0 = r1
            X0 = X

    # print("CG_X output")
    # plt.imshow(X.reshape(N1,N2,N3))
    # plt.show()
    return X






















    