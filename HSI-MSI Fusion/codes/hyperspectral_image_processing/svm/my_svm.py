import numpy as np
from svm_multiclass import svm_multiclass
from svm_samples import svm_samples

def my_svm(X,Y,Lbls,num):

    rep = 10;

    OA = [];
    OA_overall = [];

    for i in range(rep):
        np.random.seed(i+1)
        class_ind_train,class_ind_test = svm_samples(Y,Lbls,num);
        oa,oa_overall = svm_multiclass(X,Y,Lbls,class_ind_train,class_ind_test);
        OA.append(oa);
        OA_overall.append(oa_overall); 
    
    return OA, OA_overall