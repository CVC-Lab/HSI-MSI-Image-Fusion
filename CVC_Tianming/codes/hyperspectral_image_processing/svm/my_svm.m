 function [OA,OA_overall] = my_svm(X,Y,Lbls,num)

rep = 10;

OA = [];
OA_overall = [];

for i = 1:rep
    rng(i)
    [class_ind_train,class_ind_test] = svm_samples(Y,Lbls,num);
    [oa,oa_overall] = svm_multiclass(X,Y,Lbls,class_ind_train,class_ind_test);
    OA = [OA oa];
    OA_overall = [OA_overall oa_overall]; 
end