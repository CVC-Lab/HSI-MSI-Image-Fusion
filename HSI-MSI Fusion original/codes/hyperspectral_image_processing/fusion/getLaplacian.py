import numpy as np
import scipy as scp


def getLaplacian(I, n3):

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
            for i in range(n3):
                winI[:,:,i] = winI[:,:,i].T
            winI=np.reshape(winI,(neb_size,c))
            
            win_mu=np.mean(winI,0).T
            win_mu = np.reshape(win_mu,(n3,1))
            
           
            
            win_var=np.linalg.inv(winI.T @ winI/neb_size - win_mu @ win_mu.T + epsilon/neb_size*np.eye(c))
            winI=winI-np.repeat(win_mu.T,neb_size, axis = 0)
            tvals=(1 + winI @ win_var @ winI.T)/neb_size
            print(win_inds.shape)
            win_inds = np.reshape(win_inds,(win_inds.size,1))
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
