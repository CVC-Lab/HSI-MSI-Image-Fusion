import numpy as np
import sys

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
    y = (y / yz) * s;
    z = (z / yz) * s;
    tot = np.concatenate((y,z),axis=0)
    return tot