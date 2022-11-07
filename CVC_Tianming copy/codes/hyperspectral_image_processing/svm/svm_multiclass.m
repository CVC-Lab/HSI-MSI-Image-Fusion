function [OA,OA_overall] = svm_multiclass(X,Y,Lbls,class_ind_train,class_ind_test)
% binary SVM to multiclass SVM using one-against-all (OAA)

data = X;
labels = Y;

indices = labels~=0;
data_fin = data(indices,:);

Lbls_N = length(Lbls);

N_train_cl = zeros(Lbls_N,1);
N_test_cl = zeros(Lbls_N,1);

for i = 1:Lbls_N
    temp = class_ind_train{i};
    N_train_cl(i) = length(temp);
    temp = class_ind_test{i};
    N_test_cl(i) = length(temp);
end

% Training
for i = 1:Lbls_N
    
    ind1 = class_ind_train{i};
    
    ind2 = [];
    for j = 1:Lbls_N
        if i ~= j
            ind2 = [ind2;class_ind_train{j}];
        end
    end
    
    SVMModel = fitcsvm(data_fin([ind1;ind2],:),[ones(length(ind1),1);-ones(length(ind2),1)],...
        'Standardize',false,'KernelFunction','rbf','OptimizeHyperparameters','auto');
    
    save(num2str(i),'SVMModel')
    
    close all

end
 
% Testing
ind_test = [];
for i = 1:Lbls_N
    ind_test = [ind_test;class_ind_test{i}];
end

N_test = sum(N_test_cl);

S = zeros(N_test,Lbls_N);
for i = 1:Lbls_N
    load(num2str(i),'SVMModel')
    [~,score] = predict(SVMModel,data_fin(ind_test,:));
    S(:,i) = score(:,2);
end

labels_tot = zeros(N_test,1);
for i = 1:N_test
    [~,lbl_t] = max(S(i,:));
    labels_tot(i) = Lbls(lbl_t);
end

for i = 1:Lbls_N
    name = strcat(num2str(i),'.mat');
    delete(name)
end

% Classification via ''winner-takes-all'' decision rule
labels_test = [];
for i = 1:Lbls_N
    labels_test = [labels_test;Lbls(i)*ones(N_test_cl(i),1)];
end

OA = zeros(Lbls_N,1);
for i = 1:Lbls_N
    OA(i) = nnz((labels_tot==Lbls(i)) & (labels_test==Lbls(i)))/N_test_cl(i);
end

OA_overall = nnz(labels_tot==labels_test)/N_test;