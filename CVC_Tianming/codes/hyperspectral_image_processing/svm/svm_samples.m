function [class_ind_train,class_ind_test] = svm_samples(Y,Lbls,num)

Lbls_N = length(Lbls);

labels = reshape(Y,[],1);
indices = labels~=0;
labels_fin = labels(indices);

N_train_cl = zeros(Lbls_N,1);
N_test_cl = zeros(Lbls_N,1);

for i = 1:Lbls_N
    
    label_no = Lbls(i);
    
    ind_SVM = find(labels_fin==label_no);
    
    if num < 1
        N_train_cl(i) = floor(num*length(ind_SVM));)
    else
        N_train_cl(i) = num;
    end
    
    N_test_cl(i) = length(ind_SVM)-N_train_cl(i);
    
    % randomly chosen
    order = false(length(ind_SVM),1);
    order(1:N_train_cl(i)) = true;
    order = order(randperm(length(ind_SVM)));
    
    class_ind_train{i} = ind_SVM(order);
    class_ind_test{i} = ind_SVM(~order);
    
end