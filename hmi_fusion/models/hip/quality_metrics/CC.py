import numpy as np

def CC(ref,tar,mask=None):
    # % --------------------------------------------------------------------------
    # % Cross
    # Correlation
    # %
    # % USAGE
    # % out = CC(ref, tar, mask)
    # %
    # % INPUT
    # % ref: reference
    # HS
    # data(rows, cols, bands)
    # % tar: target
    # HS
    # data(rows, cols, bands)
    # % mask: binary
    # mask(rows, cols)(optional)
    # %
    # % OUTPUT
    # % out: cross
    # correlations(bands)
    # %
    # % --------------------------------------------------------------------------

    rows, cols, bands = tar.shape
    out = np.zeros(bands)

    if mask is None:


        for i in range(bands):
            tar_tmp = tar[:,:,i]
            ref_tmp = ref[:,:,i]
            cc = np.corrcoef(tar_tmp, ref_tmp)
            out[i] = cc[1,2]
    else:
        mask, _ = np.where(not mask == 0)
        for i in range(bands):
            tar_tmp = tar[:, :, i]
            ref_tmp = ref[:, :, i]
            cc = np.corrcoef(tar_tmp[mask], ref_tmp[mask])
            out[i] = cc[1,2]

    return np.mean(out)