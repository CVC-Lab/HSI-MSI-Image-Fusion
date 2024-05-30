import numpy as np
from sklearn.svm import SVC
import pickle
import scipy

# binary SVM to multiclass SVM using one-against-all (OAA)
def svm_multiclass(X, Y, lbls, class_ind_train, class_ind_test):
    data = X
    labels = Y   
    indices = np.where(labels!=0)
    data_fin = data[indices,:]
    Lbls_N = len(lbls)
    N_train_cl = np.zeros((Lbls_N,1))
    N_test_cl = np.zeros((Lbls_N,1))
    for i in range(Lbls_N):
        temp = class_ind_train[i]
        N_train_cl[i] = len(temp)
        temp = class_ind_test[i]
        N_test_cl[i] = len(temp)


    # Training
    for i in range(Lbls_N):
        ind1 = class_ind_train[i]
        ind2 = []
        for j in range(Lbls_N):
            if i != j:
                ind2.append(class_ind_train[j])
        SVMModel = SVC(kernel='rbf')
        SVMModel.fit(data_fin[ind1:ind2,:], [np.ones((len(ind1),1)),-np.ones((len(ind2),1))])
        # save the model to disk
        filename = 'fSVMmulticlass_model.sav'
        pickle.dump(SVMModel, open(filename, 'wb'))

    # Testing
    ind_test = []
    for i in range(Lbls_N):
        ind_test.append(class_ind_test[i])
    N_test = sum(N_test_cl)
    S = np.zeros((N_test,Lbls_N))
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
    labels_test = []
    for i in range(Lbls_N):
        labels_test.append(lbls[i]*np.ones(N_test_cl[i],1))
    OA = np.zeros((Lbls_N,1))
    for i in range(Lbls_N):
        OA[i] = np.count_nonzero((labels_tot==lbls[i]) & (labels_test==lbls[i]))/N_test_cl[i];
    OA_overall = np.count_nonzero(labels_tot==labels_test)/N_test;

    return OA, OA_overall

def svm_samples(Y,Lbls,num):
    Lbls_N = len(Lbls)
    labels = np.reshape(Y,(Y.shape[0]*Y.shape[1],1))
    indices = labels !=  0
    labels_fin = labels[indices]
    N_train_cl = np.zeros((Lbls_N,1))
    N_test_cl = np.zeros((Lbls_N,1))
    class_ind_train = []
    class_ind_test = []

    for i in range(Lbls_N):
        label_no = Lbls[i]
        ind_SVM = np.where(labels_fin==label_no)[0]
        #print(ind_SVM[0])
        
        if num < 1:
            N_train_cl[i] = np.floor(num*len(ind_SVM))
            #print(N_train_cl)
        else:
            N_train_cl[i] = int(num)

        #print(N_train_cl[i])
        
        N_test_cl[i] = len(ind_SVM)-N_train_cl[i]
        
        #print(N_train_cl[i])
        # randomly chosen
        order = np.zeros((len(ind_SVM),1))
        
        for j in range(int(N_train_cl[i])):
            order[j] = 1
        order = (np.random.permutation(order))
        
        # print(int(order))
        # print(order.shape)
        # print(ind_SVM.shape)
        # ind_SVM = np.reshape(ind_SVM, (1428,1))
        # print(ind_SVM.shape)
        # print(ind_SVM)

        # order = np.reshape(order, (1428,))
        # class_ind_train.append(ind_SVM[int(order)])

        # not_order = np.ones((len(ind_SVM),1))
        # for j in range(N_train_cl[i]):
        #     not_order[j] = 0
        
        # class_ind_test.append(ind_SVM[not_order])

        train = []
        test = []

        for ord in range(len(order)):
            if order[ord] == 0:
                test.append(ind_SVM[ord])
            else:
                train.append(ind_SVM[ord])

        train_array = np.array(train)
        test_array = np.array(test)

        class_ind_train.append(train_array)
        class_ind_test.append(test_array)
        
    return np.array(class_ind_train), np.array(class_ind_test)

def my_svm(X,Y,Lbls,num):
    rep = 10
    OA = []
    OA_overall = []
    for i in range(rep):
        np.random.seed(i+1)
        class_ind_train,class_ind_test = svm_samples(Y,Lbls,num)
        print(class_ind_train[1].shape)
        print(class_ind_test[1].shape)
        oa,oa_overall = svm_multiclass(X,Y,Lbls,class_ind_train,class_ind_test)
        OA.append(oa)
        OA_overall.append(oa_overall); 
    return OA, OA_overall


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