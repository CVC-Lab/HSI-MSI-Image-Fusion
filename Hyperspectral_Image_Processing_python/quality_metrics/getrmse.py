import numpy as np

def getrmse(x, y):
    wid_x, hi_x, n_bands = x.shape
    n_samples = wid_x * hi_x

    aux = np.sum((x-y)**2)/(n_samples * n_bands)
    rmse_total = np.sqrt(aux)

    return rmse_total