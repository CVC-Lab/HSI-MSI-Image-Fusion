import numpy as np

def denoising(X,mask=None):
    #--------------------------------------------------------------
    # Denoising of HSI
    #
    # USAGE
    #     [Y,SNR_dB] = denoising(X)
    # INPUT
    #     X      : hyperspectral data (rows,cols,bands)
    #     mask   : list of indices to include
    # OUTPUT
    #     Y      : denoised hyperspectral data (rows,cols,bands)
    #     SNR_dB : estimated SNR in dB
    #
    # REFERENCE
    #     R. Roger, "Principal components transform with simple automatic noise
    #     adjustment," International Journal of Remote Sensing, vol. 17, pp.
    #     2719-2727, 1996
    #
    # Author: Naoto YOKOYA
    # Email : yokoya@sal.rcast.u-tokyo.ac.jp
    #--------------------------------------------------------------
    [rows,cols,bands] = X.shape
    SNR_dB = np.zeros(bands)
    Y = X

    if bands == 1:
        x0 = np.reshape(X[:,:,0],[-1,1])
        if mask == None:
            mask = list(range(x0.shape[0]))
        x = x0[mask,0]
        Y[:,:,0] = np.reshape(x,[rows,cols])
        SNR_dB[0] = 20*np.log10(np.mean(x0)/np.mean((x0-np.reshape(Y[:,:,0],[-1,1]))**2)**0.5)

    else: 
        for i in range(bands):
            # print(str(i) + 'th band')
            x0 = np.reshape(X[:,:,i],[-1,1])
            if mask == None:
                mask = list(range(x0.shape[0]))
            if i == 0:
                A0 = np.reshape(X[:,:,i+1:],[-1,bands-1])
            elif i == bands-1:
                A0 = np.reshape(X[:,:,0:bands-1],[-1,bands-1])
            else:
                A0 = np.reshape(X[:, :, list(range(0,i)) +  list(range(i+1,X.shape[2]))],[-1,bands-1])

            x = x0[mask,0]
            A = A0[mask,:]
            invAtA = np.linalg.pinv(A.T @ A)

            c = (invAtA) @ (A.T @ x)
            temp = np.matmul(A0,c)
            Y[:,:,i] = np.reshape(temp,[rows,cols])
            SNR_dB[i] = 20*np.log10(np.mean(x0)/np.mean((x0-np.reshape(Y[:,:,i],[-1,1]))**2)**0.5)
    
    return Y,SNR_dB