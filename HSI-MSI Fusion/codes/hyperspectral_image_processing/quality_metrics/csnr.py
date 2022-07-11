import numpy as np

def csnr(A,B,row,col):

    n, m, ch = A.shape
    summa = 0
    e = A-B
    if ch == 1:
        e = e[row+1:n-row,col+1:m-col]
        me = np.mean(e**2)
        s = 10 * np.log10(255**2/me)
    else:
        for i in range(ch):
            e = e[row+1:n-row,col+1:m-col,i]
            mse = np.mean(e**2)
            s = 10 * np.log10(255**2/mse)
            summa += s
        s = summa/ch

    return s