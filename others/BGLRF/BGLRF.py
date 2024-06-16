import sys
import numpy as np
import scipy as scp
import cv2

def LHS_K(K0 ,D ,FX ,ind ,IND ,mu ,tau ,N1 ,N2 ,N3):
    p, q = K0.shape
    K0 = K0.T
    K0 = np.reshape(K0,(169,1))
    K = D.T @ D @ K0[:]
    K0 = np.reshape(K0,(13,13))
    K = np.reshape(K, [p, q])
    K = mu * K + (tau + mu) * K0
    K = K.T
    K0 = np.reshape(K0,(169,1))
    temp = np.zeros((N1, N2))
    temp = np.reshape(temp,(20736,1))
    temp[ind[:]] = K0[:]
    temp = np.reshape(temp,(144,144))
    Fk = np.fft.fft2(temp)
    Fk = Fk.T
    K0 = np.reshape(K0,(13,13))
    for band in range(N3):
        Fx = FX[:, band]
        Fx = np.reshape(Fx, [N1, N2])
        Fx=Fx.T
        temp = np.real(np.fft.ifft2(Fx * Fk))
        y = np.zeros((N1, N2))
        temp = temp.flatten()
        y = y.flatten()
        y[IND[:]] = temp[IND[:]]
        y = np.reshape(y,(144,144))
        Fy = np.fft.fft2(y)
        temp = np.real(np.fft.ifft2(np.conj(Fx) * Fy))
        #temp = temp.T
        temp = temp.flatten()
        temp = temp[ind[:]]
        temp = np.reshape(temp, [p, q])
        K = K + temp
    return K

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
        K = K0 + a *p0
        pp = np.reshape(pp,(13,13))
        r0 = np.reshape(r0,(13,13))
        r1 = r0 - a *pp
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

def LHS_X(X0,Fk,L,IND,N1,N2,N3,tau):
    X = X0.copy()
    for band in range(N3):
        x = X0[:,band]
        x = np.reshape(x,[N1, N2])
        x = x.T
        Fx = np.fft.fft2(x)
        x = np.real(np.fft.ifft2(Fk*Fx))
        x = x.T
        x = x.flatten()
        temp = np.zeros((N1*N2,1))
        temp = temp.flatten()
        temp[IND[:]] = x[IND[:]]
        temp = temp.reshape((N1,N2))
        temp = temp.T
        temp = np.fft.fft2(temp)
        x = np.real(np.fft.ifft2(np.conj(Fk)*temp))
        x = x.T
        x = x.flatten()
        X[:,band] = x[:] 
    X = X + L @ X0
    X = X+tau*X0
    return X

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
        p0 = np.reshape(p0,(20736,200))
        #p0=p0.T
        X = X0 + a * p0
        pp = np.reshape(pp,(20736,200))
        r0 = np.reshape(r0,(20736,200))
        r1 = r0 - a *pp  
        res = np.linalg.norm(r1[:])
        if res < tolH:
            print('CG_X is successful in {} iterations \n'.format(i))
            break
        else:
            r0 = r0.flatten()
            b1 = res**2 /(r0[:].T @ r0[:]) # compute beta_k
            r0 = np.reshape(r0,(20736,200))
            p1 = r1 + b1 *p0

            p0 = p1
            r0 = r1
            X0 = X
    return X

def nabla(p,q):
    # calculate the gradient in x and y directions, by defining a (sparse) matrix
    l = p*q
    Dx = np.zeros((l,l))
    Dy = np.zeros((l,l))
    for i in range(p):
        for j in range(q):
            ind1 = (j-1)*p+i
            Dx[ind1,ind1] = 1
            if j >= 1:
                ind2 = ind1-p
                Dx[ind1,ind2] = -1
    for i in range(p):
        for j in range(q):
            ind1 = (j-1)*p+i
            Dy[ind1,ind1] = 1
            if i >= 1 :
                ind2 = ind1-1
                Dy[ind1,ind2] = -1
    D = np.concatenate((Dx,Dy))
    D = scp.sparse.csr_matrix(D)
    return D

def getLaplacian(I):
    I=I.astype('float64')
    epsilon=1e-7
    win_size=1
    neb_size=(win_size*2+1)**2
    h, w, c = I.shape
    img_size=w*h
    consts=np.zeros((h,w))
    indsM=np.reshape(list(range(img_size)),(h,w))
    indsM=np.transpose(indsM)
    tlen=np.sum(np.sum(1-consts[win_size:-win_size,win_size:-win_size]))*(neb_size**2)
    tlen = round(tlen)
    row_inds=np.zeros((tlen,1))
    col_inds=np.zeros((tlen,1))
    vals=np.zeros((tlen,1))
    len=0
    for j in range(win_size,w-win_size):
        for i in range(win_size,h-win_size):
            win_inds=indsM[i-win_size:i+win_size+1,j-win_size:j+win_size+1]
            win_inds=np.sort(win_inds, axis=None)
            winI=I[i-win_size:i+win_size+1,j-win_size:j+win_size+1,:].copy()
            for i in range(6):
                winI[:,:,i] = winI[:,:,i].T
            winI=np.reshape(winI,(neb_size,c))  
            win_mu=np.mean(winI,0).T
            win_mu = np.reshape(win_mu,(6,1))  
            win_var=np.linalg.inv(winI.T @ winI/neb_size - win_mu @ win_mu.T + epsilon/neb_size*np.eye(c))
            winI=winI-np.repeat(win_mu.T,neb_size, axis = 0)
            tvals=(1 + winI @ win_var @ winI.T)/neb_size
            win_inds = np.reshape(win_inds,(9,1))
            row_inds[len:neb_size**2+len]=np.reshape(np.repeat(win_inds.T,neb_size,axis = 0),(neb_size**2,1))
            col_inds[len:neb_size**2+len]=np.reshape(np.repeat(win_inds.T,neb_size,axis = 1),(neb_size**2,1))
            #row_inds[len:neb_size**2+len]=np.reshape(np.matlib.repmat(win_inds,1,neb_size),[neb_size**2,1])
            #col_inds[len:neb_size^2+len]=np.reshape(np.matlib.repmat(win_inds.H,[neb_size,1]),[neb_size**2,1])
            temp = tvals.T
            temp = np.reshape(temp,(81,1))
            vals[len:neb_size**2+len]=temp[:]
            len=len+neb_size**2
    row_inds = np.ravel(row_inds)
    col_inds = np.ravel(col_inds)
    vals = np.ravel(vals)
    A = scp.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))
    sumA=np.sum(A,1)
    sumA = np.ravel(sumA)
    A=scp.sparse.spdiags(sumA[:],0,img_size,img_size)-A
    test = scp.sparse.coo_matrix.todense(A)
    return A

def soft_isotropic(x, tau):
    l = int(len(x) / 2)
    y = x[:l]
    z = x[l:]
    yz = np.sqrt(y**2 + z**2 + sys.float_info.epsilon)
    test = np.zeros((169,1))
    test3 = yz[:]-tau
    test2 = np.concatenate((test3,test), axis=1)
    s = np.maximum(test2[:,0],test2[:,1])
    s = np.reshape(s,(169,1))
    y = (y / yz) * s
    z = (z / yz) * s
    tot = np.concatenate((y,z),axis=0)
    return tot

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
    err = np.zeros((200, 1))
    for i in range(999):
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
            temp = u[j] + 1 / (j+1) * (1 - s)
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
        print('ADMM_K', err[i])
        if err[i] < 1e-2:
            err = err[:i]
            print('ADMM_K is successful in {} iterations \n'.format(i))
            break
        else:
            # update multipliers
            l1 = l1 + gg - g
            L2 = L2 + kK - K

    return K, err

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
    for i in range(N3):
        Y[:,:,i] = Y[:,:,i].T
    Y = np.reshape(Y,[n1*n2,N3])

    N1,N2,_ = Z.shape

    # graph Laplacian
    GL = getLaplacian(Z)

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
    I2 = np.reshape(I2,(1,36))
    I2 = np.repeat(I2,n2,axis=0)
    I2 = np.reshape(I2,(1296,1))
    J2 = np.array(range(1,N2,ratio))
    J2 = np.reshape(J2,(36,1))
    J2 = np.repeat(J2,n1,axis=0)
    I2 = I2.astype(int)
    J2 = J2.astype(int)
    I2 = np.ravel(I2)
    J2 = np.ravel(J2)
    arr = np.array([J2,I2])
    ind2 = np.ravel_multi_index(arr,(N1,N2)) # indices for HSI

    ## Initialization
    fY = np.zeros((N1 * N2,N3),dtype='complex') # used in all the iterations
    X = np.zeros((N1*N2,N3))

    for band in range(N3):
        y = Y[:,band]
        y = np.reshape(y,[n1, n2])
        y = y.T
        temp = np.zeros((N1,N2))
        temp = np.reshape(temp,(20736,1))
        y = np.reshape(y,(1296,1))
        temp[ind2[:]] = y[:]
        temp = np.reshape(temp,(144,144))
        y = np.reshape(y,(36,36))
        temp = np.fft.fft2(temp)
        temp = temp.T
        temp = np.reshape(temp,(20736,1))
        temp = np.ravel(temp)
        fY[:,band] = temp[:]
        x = cv2.resize(y,(N1,N2),interpolation=cv2.INTER_CUBIC)
        x=x.T
        x = x.flatten()
        X[:,band] = x[:]

    K = np.zeros((13,13))

    maxiters = 1 # maximum number of iterations
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
            Fx = np.reshape(Fx,(144,144))
            Fy=Fy.T
            Fx=Fx.T
            y = np.real(np.fft.ifft2(np.conj(Fx)*Fy))
            y = np.reshape(y,(20736,1))
            test = y[ind[:],0]
            test = test.T
            test = np.reshape(test,(169,1))
            rhs = rhs+test

        rhs = np.reshape(rhs,(13,13))
        rhs = rhs+tau*K
        
        K_pre = K.copy()
        K,err2= ADMM_K(rhs,K,D,FX,ind,ind2,beta,tau,N1,N2,N3)
        #print(err2)

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
        #print('HELLO')

        err[iter,0] = np.linalg.norm(K-K_pre,'fro')/np.linalg.norm(K,'fro')
        err[iter,1] = np.linalg.norm(X-X_pre,'fro')/np.linalg.norm(X,'fro')

        print('GLRF - iteration {}: \n'.format(iter,err[iter,0],err[iter,1]))
        print(err[iter,1])
        if err[iter,1] < 1e-3:
            err = err[:iter,:]
            X = np.reshape(X,[N1,N2,N3])
            break

    return X, K