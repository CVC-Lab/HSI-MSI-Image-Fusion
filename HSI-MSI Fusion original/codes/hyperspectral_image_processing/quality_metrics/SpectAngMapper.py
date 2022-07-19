import numpy as np
import sys
from math import acos
def SpectAngMapper(imagery1, imagery2):
    # Syntax:
    # [psnr, ssim, fsim, ergas, msam] = MSIQA(imagery1, imagery2)
    # Input:
    # imagery1 - the reference MSI data array
    # imagery2 - the target MSI data array
    # NOTE: MSI data array is a M * N * K array for imagery with M * N spatial
    # pixels, K bands and DYNAMIC RANGE[0, 255]. If imagery1 and imagery2
    # have different size, the larger one will be truncated to fit the
    # smaller one.
    # [1] R.YUHAS, J.BOARDMAN, and A.GOETZ, "Determination of semi-arid
    # landscape endmembers and seasonal trends using convex geometry
    # spectral unmixing techniques ", JPL, Summaries of the 4 th Annual JPL
    # Airborne Geoscience Workshop. 1993.
    # See also StructureSIM, FeatureSIM and ErrRelGlobAdimSyn
    #
    # by Yi Peng

    eps = sys.float_info.epsilon
    tmp = (np.sum(imagery1 * imagery2,axis=2) + eps) / (np.sqrt(np.sum(imagery1**2,axis=2)) + eps) / (np.sqrt(np.sum(imagery2**2,axis=2)) + eps)
    sam = np.mean(np.real(acos(tmp)))

    return sam