import scipy.io
import numpy as np
from my_svm import my_svm

filename ='../test_data/indian_pines.mat'
mat = scipy.io.loadmat(filename)

indian_pines_c = mat["indian_pines_corrected"]
scaling = mat["scaling"]
indian_pines_gt = mat["indian_pines_gt"]

X = indian_pines_c/scaling
n3 = X.shape[2]
n1 = X.shape[0]
n2 = X.shape[1]
X = np.reshape(X,(n1*n2,n3))
Y = indian_pines_gt
Y = Y[:]
Lbls = [2,3,5,6,8,10,11,12,14] # 9 out of 16 classes are chosen

[OA,OA_overall] = my_svm(X,Y,Lbls,0.1) # train on 10 percent samples

oa = np.mean(OA,2); # classification accuracy of each class
oa_overall = np.mean(OA_overall) # overall classification accuracy

print(oa_overall)