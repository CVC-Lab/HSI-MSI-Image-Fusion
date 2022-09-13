import numpy as np
import scipy as scp

def nabla(p,q):
    # calculate the gradient in x and y directions, by defining a (sparse) matrix

    l = p*q
    Dx = np.zeros(l,l)
    Dy = np.zeros(l,l)

    for i in range(p):
        for j in range(q):
            ind1 = (j-1)*p+i
            Dx[ind1,ind1] = 1
            if j > 1:
                ind2 = ind1-p
                Dx[ind1,ind2] = -1

    for i in range(p):
        for j in range(q):
            ind1 = (j-1)*p+i
            Dy[ind1,ind1] = 1
            if i > 1 :
                ind2 = ind1-1
                Dy[ind1,ind2] = -1


    D = [Dx,Dy]
    D = scp.sparse.csr_matrix(D)

    return D