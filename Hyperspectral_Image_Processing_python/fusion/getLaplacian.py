import numpy as np
import scipy as scp

def getLaplacian(I):

    I=I.astype('float64')
    epsilon=1e-7
    win_size=1
    #neb_size=(win_size*2+1)**2
    neb_size=(win_size*2)**2
    h, w, c = I.shape
    img_size=w*h
    consts=np.zeros((h,w))
    indsM=np.reshape(list(range(img_size)),(h,w))
    tlen=int(np.sum(np.sum(1-consts[win_size:-win_size-1,win_size:-win_size-1]))*(neb_size**2))
    row_inds=np.zeros((tlen,1))
    col_inds=np.zeros((tlen,1))
    vals=np.zeros((tlen,1))
    len=0

    for j in range(win_size,w-win_size-1):
        for i in range(win_size,h-win_size-1):
            win_inds=indsM[i-win_size:i+win_size,j-win_size:j+win_size]
            win_inds=win_inds[:]
            winI=I[i-win_size:i+win_size,j-win_size:j+win_size,:]
            winI=np.reshape(winI,[neb_size,c])
            win_mu=np.mean(winI,0).T
            win_var=np.linalg.inv(winI.conj().T @ winI/neb_size-win_mu @ win_mu.conj().T + epsilon/neb_size*np.eye(c))
            winI=winI-np.matlib.repmat(win_mu.conj().T,neb_size,1)
            tvals=(1 + winI @ win_var @ winI.conj().T)/neb_size
            row_inds[len:neb_size**2+len]=np.reshape(np.matlib.repmat(win_inds,1,neb_size),[neb_size**2,1])
            col_inds[len:neb_size^2+len]=np.reshape(np.matlib.repmat(win_inds.H,[neb_size,1]),[neb_size**2,1])
            vals[len:neb_size^2+len]=tvals[:]
            len=len+neb_size**2

    vals=vals[:len]
    row_inds=row_inds[1:len]
    col_inds=col_inds[1:len]
    # A=sparse(row_inds,col_inds,vals,img_size,img_size)
    A = scp.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))
    sumA=np.sum(A,1)
    A=scp.sparse.spdiags(sumA[:],0,img_size,img_size)-A

    return A