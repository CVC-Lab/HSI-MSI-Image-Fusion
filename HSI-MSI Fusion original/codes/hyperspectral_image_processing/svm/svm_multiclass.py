import numpy as np
from sklearn.svm import SVC
import pickle

# binary SVM to multiclass SVM using one-against-all (OAA)
def svm_multiclass(X, Y, lbls, class_ind_train, class_ind_test):
    
    data = X
    labels = Y   

    indices = np.where(labels!=0)
    data_fin = data[indices,:]

    Lbls_N = len(lbls)

    N_train_cl = np.zeros((Lbls_N,1));
    N_test_cl = np.zeros((Lbls_N,1));

    for i in range(Lbls_N):
        temp = class_ind_train[i];
        N_train_cl[i] = len(temp);
        temp = class_ind_test[i];
        N_test_cl[i] = len(temp);


    # Training
    for i in range(Lbls_N):
        
        ind1 = class_ind_train[i];
        
        ind2 = [];
        for j in range(Lbls_N):
            if i != j:
                ind2.append(class_ind_train[j]);

        SVMModel = SVC(kernel='rbf')

        SVMModel.fit(data_fin[ind1:ind2,:], [np.ones((len(ind1),1)),-np.ones((len(ind2),1))])

        # save the model to disk
        filename = 'fSVMmulticlass_model.sav'
        pickle.dump(SVMModel, open(filename, 'wb'))


    # Testing
    ind_test = [];
    for i in range(Lbls_N):
        ind_test.append(class_ind_test[i]);


    N_test = sum(N_test_cl);

    S = np.zeros((N_test,Lbls_N));
    for i in range(Lbls_N):
        loaded_model = pickle.load(open(filename, 'rb'))
        output = loaded_model.predict(data_fin[ind_test,:])
        score = loaded_model.score(output, np.ones(len(ind_test),1));
        S[:,i] = score[:,1]

    labels_tot = np.zeros((N_test,1))
    for i in range(N_test):
        lbl_t = max(S[i,:])
        labels_tot[i] = lbls[lbl_t]

    # for i in range(Lbls_N):
    #     name = strcat(num2str(i),'.mat');
    #     delete(name)

    # Classification via ''winner-takes-all'' decision rule
    labels_test = [];
    for i in range(Lbls_N):
        labels_test.append(lbls[i]*np.ones(N_test_cl[i],1))

    OA = np.zeros((Lbls_N,1));
    for i in range(Lbls_N):
        OA[i] = np.count_nonzero((labels_tot==lbls[i]) & (labels_test==lbls[i]))/N_test_cl[i];

    OA_overall = np.count_nonzero(labels_tot==labels_test)/N_test;

    return OA, OA_overall
   


