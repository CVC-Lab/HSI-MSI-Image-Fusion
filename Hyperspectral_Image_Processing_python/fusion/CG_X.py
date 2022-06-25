import numpy as np
import LHS_X

def CG_X(H,X0,Fk,L,IND,N1,N2,N3,tau):

    normH = np.linalg.norm(H[:])
    tolH = 1e-3*normH

    r0 = H - LHS_X(X0,Fk,L,IND,N1,N2,N3,tau)
    p0 = r0

    for i in range(999):

        pp = LHS_X(p0,Fk,L,IND,N1,N2,N3,tau)
        pp1 = p0[:].H @ pp[:]
        a = (r0[:].H @ r0[:])/pp1; # compute alpha_k

        X = X0+a*p0
        r1 = r0-a*pp

        res = np.linalg.norm(r1[:])
        if res < tolH:
            print('CG_X is successful in {} iterations \n'.format(i))
            break
        else:
            b1 = res**2/(r0[:].H @ r0[:]) # compute beta_k
            p1 = r1+b1*p0

            p0 = p1
            r0 = r1
            X0 = X

    return X