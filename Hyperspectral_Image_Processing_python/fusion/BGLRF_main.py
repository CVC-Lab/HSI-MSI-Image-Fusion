import numpy as np
import scipy as scp
# from codes.hyperspectral_image_processing.fusion.ADMM_K import ADMM_K
# from codes.hyperspectral_image_processing.fusion.getLaplacian import getLaplacian
# from codes.hyperspectral_image_processing.fusion.nabla import nabla
# from codes.hyperspectral_image_processing.fusion.CG_X import CG_X
import ADMM_K
import getLaplacian
import nabla
import CG_X

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
    Y = np.reshape(Y,[n1*n2,N3])

    N1,N2,_ = Z.shape

    # graph Laplacian
    GL = getLaplacian.getLaplacian(Z)

    tau = 0 # proximal weight
    L = alpha*GL+tau*scp.sparse.identity(N1*N2,N1*N2) # for updating X

    kernel = (2*radius+1)*[1, 1] # search space for blur kernel
    D = nabla(kernel(1),kernel(2)) # operator for partial derivatives

    center = np.zeros((1,2))
    if N1%2:
        center[0] = (N1+1)/2
    else:
        center[0] = N1/2+1
        if N2%2:
            center[1] = (N2+1)/2
        else:
            center[1] = N2/2+1

    I = range((center[0]-radius),(center[0]+radius))
    I = I.T
    I = np.matlib.repmat(I,[1, 2*radius+1])
    I = I[:]
    J = range((center[1]-radius),(center[1]+radius))
    J = np.matlib.repmat(J,[2*radius+1, 1])
    J = J[:]
    ind = np.ravel_multi_index(J,I,(N1,N2)) # indices for blur kernel

    ratio = round(N1/n1) # downsampling ratio

    I = list(range(1,N1,ratio))
    I = I.T
    I = np.matlib.repmat(I,[1, n2])
    I = I[:]
    J = list(range(1,N2,ratio))
    J = np.matlib.repmat(J,[n1, 1])
    J = J[:]
    IND = np.ravel_multi_index(J,I,(N1,N2)) # indices for HSI

    ## Initialization
    FY = np.zeros((N1 * N2,N3)) # used in all the iterations
    X = np.zeros((N1*N2,N3))

    for band in range(N3):
        y = Y[:,band]
        y = np.reshape(y,[n1, n2])
        temp = np.zeros((N1,N2))
        temp[IND] = y
        temp = np.fft.fft2(temp)
        FY[:,band] = temp[:]
        x = scp.misc.imresize(y,[N1, N2]) #degraded in scipy 10, use pillow instead
        X[:,band] = x[:]

    K = np.zeros(kernel)

    maxiters = 100 # maximum number of iterations
    err = np.zeros((maxiters,2))

    ## Proximal alternating minization
    for iter in range(maxiters):

        # update K
        FX = np.zeros((N1*N2,N3))
        RHS = (kernel(1)*kernel(2)).shape[0]

        for band in range(N3):
            x = X[:,band]
            x = np.reshape(x,[N1, N2])
            Fx = np.fft.fft2(np.roll(x,1-center))
            FX[:,band] = Fx[:]
            Fy = FY[:,band]
            Fy = np.reshape(Fy,[N1, N2])
            y = np.real(np.fft.ifft2(np.conj(Fx)*Fy))
            RHS = RHS+y[ind]

        RHS = np.reshape(RHS,[kernel[0],kernel[1]])
        RHS = RHS+tau*K

        K_pre = K
        K = ADMM_K(RHS,K,D,FX,ind,IND,beta,tau,N1,N2,N3)

        # update X
        KK = np.zeros((N1,N2))
        KK[ind] = K
        Fk = np.fft.fft2(np.roll(KK,1-center))

        RHS = np.zeros((N1*N2,N3))
        for band in range(N3):
            temp = FY[:,band]
            temp = np.reshape(temp,[N1, N2])
            temp = np.real(np.fft.ifft2(np.conj(Fk)*temp))
            RHS[:,band] = temp[:]
        RHS = RHS+tau*X

        X_pre = X
        X = CG_X(RHS,X,Fk,L,IND,N1,N2,N3,tau)

        err[iter,0] = np.linalg.norm(K-K_pre,'fro')/np.linalg.norm(K,'fro')
        err[iter,1] = np.linalg.norm(X-X_pre,'fro')/np.linalg.norm(X,'fro')

        print('GLRF - iteration {}: change.K = {.4f}, change.X = {.4f} \n'.format(iter,err(iter,1),err(iter,2)))

        if err[iter,1] < 1e-3:
            err = err[:iter,:]
            X = np.reshape(X,[N1,N2,N3])
            break

    return X, K, err