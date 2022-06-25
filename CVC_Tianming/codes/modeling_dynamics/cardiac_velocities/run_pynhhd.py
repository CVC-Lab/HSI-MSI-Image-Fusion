import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
from pynhhd import nHHD

data = loadmat('Velocities.mat')
Vx = data['Vx']
Vy = data['Vy']

def Cardiac_OF_hhd(Vx,Vy):

    dims = (np.size(Vx,0),np.size(Vx,1))
    pixel = (1,1) # change here if spatial ratio is not 1
    
    vx = Vx[:,:,0]
    vy = Vy[:,:,0]
    V = np.dstack((vx,vy))
    nhhd = nHHD(grid=dims,spacings=pixel)
    nhhd.decompose(V)
    C1 = nhhd.r
    C2 = nhhd.d
    C3 = nhhd.h
    
    i = 1
    while i < np.size(Vx,2):
        vx = Vx[:,:,i]
        vy = Vy[:,:,i]
        V = np.dstack((vx,vy))
        nhhd = nHHD(grid=dims,spacings=pixel)
        nhhd.decompose(V)
        c1 = nhhd.r
        c2 = nhhd.d
        c3 = nhhd.h
        C1 = np.dstack((C1,c1))
        C2 = np.dstack((C2,c2))
        C3 = np.dstack((C3,c3))
        i += 1
    print('done')
    return C1, C2, C3

C1, C2, C3 = Cardiac_OF_hhd(Vx,Vy)

savemat("C1.mat",mdict={"C1":C1}) # divergence-free
savemat("C2.mat",mdict={"C2":C2}) # curl-free
savemat("C3.mat",mdict={"C3":C3}) # harmonic