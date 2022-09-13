import numpy as np
import sys

def soft_isotropic(x, tau):
    l = len(x) / 2

    y = x[:l]
    z = x[l+1:]

    yz = np.sqrt(y**2 + z**2 + sys.float_info.epsilon)
    s = max(yz - tau, 0)

    y = (y / yz) * s;
    z = (z / yz) * s;

    return [y,z]