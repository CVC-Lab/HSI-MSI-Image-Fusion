import numpy as np
import scipy as scp
import cv2
from ADMM_K import ADMM_K
from getLaplacian import getLaplacian
from nabla import nabla
from CG_X import CG_X

def BGLRF_main(Y,Z,alpha,beta,radius):
    # X is initialized first using bicubic interpolation
    #
    # Y: HSI in tensor format, between 0 and 1
    # Z: MSI in tensor format, between 0 and 1
    # alpha: regularization parameter for graph Laplacian
    # beta: TV regularization parameter for the kernel
    # radius: radius for assumed blur kernel
    #
    # X: recovered SRI
    # K: estimated blur kernel
    # err: relative changes in X and K
    #
    # Blind Hyperspectral-Multispectral Image Fusion via Graph Laplacian
    # Regularization

    ## Define parameters, etc.
    n1,n2,N3 = Y.shape
    temp = np.zeros((n2, n1, N3))
    for i in range(N3):
        temp[:,:,i] = np.transpose(Y[:,:,i])
    Y = np.reshape(temp,[n1*n2,N3])

    N1,N2,n3 = Z.shape

    # graph Laplacian
    GL = getLaplacian(Z, n3)
    print(GL)
    print(GL.shape)

    tau = 0 # proximal weight
    L = alpha*GL+tau*scp.sparse.identity(N1*N2) # for updating X
    test = scp.sparse.csr_matrix.toarray(L)
    kernel = [2*radius+1,2*radius+1] # search space for blur kernel
    D = nabla(13,13) # operator for partial derivatives
   
    center = np.zeros((2,1))
    if N1%2:
        center[0] = (N1+1)/2
    else:
        center[0] = N1/2+1
        if N2%2:
            center[1] = (N2+1)/2
        else:
            center[1] = N2/2+1

    I = np.linspace(int((center[0]-radius-1)),int((center[0]+radius-1)),2*radius+1)
    I = np.reshape(I,(13,1))
    I = I.T
    I = np.repeat(I,2*radius+1,axis=0)
    I = np.reshape(I,(169,1))
    I = np.ravel(I)
    J = np.linspace(int((center[1]-radius-1)),int((center[1]+radius-1)),2*radius+1)
    J = np.reshape(J,(13,1))
    J = np.repeat(J,2*radius+1, axis=0)
    J = np.ravel(J)
    I = I.astype(int)
    J = J.astype(int)
    arr = np.array([J,I])
    ind = np.ravel_multi_index(arr,(N1,N2)) # indices for blur kernel

    ratio = round(N1/n1) # downsampling ratio

    I2 = np.array((range(1,N1,ratio)))
    
    I2 = I2.T
    I2 = np.reshape(I2,(1,n2))
    I2 = np.repeat(I2,n2,axis=0)
    I2 = np.reshape(I2,(n1*n2,1))
    J2 = np.array(range(1,N2,ratio))
    J2 = np.reshape(J2,(n1,1))
    J2 = np.repeat(J2,n1,axis=0)
    I2 = I2.astype(int)
    J2 = J2.astype(int)
    I2 = np.ravel(I2)
    J2 = np.ravel(J2)
    arr = np.array([J2,I2])
    ind2 = np.ravel_multi_index(arr,(N1,N2)) # indices for HSI
    # print(ind2.shape)
    # print(ind2)

    ## Initialization
    fY = np.zeros((N1 * N2,N3),dtype='complex') # used in all the iterations
    X = np.zeros((N1*N2,N3))

    for band in range(N3):
        y = Y[:,band]
        y = np.reshape(y,[n1, n2])
        y = y.T
        temp = np.zeros((N1,N2))
        temp = np.reshape(temp,(N1*N2,1))
        y = np.reshape(y,(n1*n2,1))
        temp[ind2[:]] = y[:]
        temp = np.reshape(temp,(N1,N2))
        y = np.reshape(y,(n1,n2))
        temp = np.fft.fft2(temp)
        temp = temp.T
        temp = np.reshape(temp,(N1*N2,1))
        temp = np.ravel(temp)
        fY[:,band] = temp[:]
        x = cv2.resize(y,(N1,N2),interpolation=cv2.INTER_CUBIC)
        x=x.T
        x = x.flatten()
        
        X[:,band] = x[:]

    K = np.zeros((13,13))

    maxiters = 100 # maximum number of iterations
    err = np.zeros((maxiters,2))



    ## Proximal alternating minization
    for iter in range(maxiters):
        
        # update K
        FX = np.zeros((N1*N2,N3),dtype='complex')
        rhs = np.ones((169,1))
        sum=0
        for band in range(N3):
            x = X[:,band]
            x = np.reshape(x,(N1, N2))
            x = x.T
            temp = np.roll(x,(int(1-center[0]),int(1-center[0])),axis = (1,0))
            Fx = np.fft.fft2(np.roll(x,(int(1-center[0]),int(1-center[0])),axis = (1,0)))
            Fx = Fx.T
            Fx = Fx.flatten()
            FX[:,band] = Fx[:]
            Fy = fY[:,band]
            Fy = np.reshape(Fy,[N1, N2])
            Fx = np.reshape(Fx,(N1,N2))
            Fy=Fy.T
            Fx=Fx.T
            y = np.real(np.fft.ifft2(np.conj(Fx)*Fy))
            y = np.reshape(y,(N1*N2,1))
            test = y[ind[:],0]
            test = test.T
            test = np.reshape(test,(169,1))
            rhs = rhs+test

        rhs = np.reshape(rhs,(13,13))
        rhs = rhs+tau*K
        
        K_pre = K.copy()
        K,err2= ADMM_K(rhs,K,D,FX,ind,ind2,beta,tau,N1,N2,N3)

        # update X
        KK = np.zeros((N1*N2,1))
        for i in range(13):
            for j in range(13):
                KK[ind[j+i*13]] = K[i,j]
        #KK[ind[:],0] = K[:]
        KK = np.reshape(KK,(N1,N2))
        Fk = np.fft.fft2(np.roll(KK,(int(1-center[0]),int(1-center[0])),axis = (1,0)))

        rhs = np.zeros((N1*N2,N3))
        for band in range(N3):
            temp = fY[:,band]
            temp = np.reshape(temp,[N1, N2])
            temp =temp.T
            temp = np.real(np.fft.ifft2(np.conj(Fk)*temp))
            temp = temp.T
            temp = temp.flatten()
            rhs[:,band] = temp[:]
        rhs = rhs+tau*X

        X_pre = X.copy()
        X = CG_X(rhs,X,Fk,L,ind2,N1,N2,N3,tau)

        err[iter,0] = np.linalg.norm(K-K_pre,'fro')/np.linalg.norm(K,'fro')
        err[iter,1] = np.linalg.norm(X-X_pre,'fro')/np.linalg.norm(X,'fro')

        print('GLRF - iteration {}: \n'.format(iter,err[iter,0],err[iter,1]))
        print(err[iter,1])
        if err[iter,1] < 2e-2:
            err = err[:iter,:]
            X = np.reshape(X,[N1,N2,N3])
            break

    return X, K