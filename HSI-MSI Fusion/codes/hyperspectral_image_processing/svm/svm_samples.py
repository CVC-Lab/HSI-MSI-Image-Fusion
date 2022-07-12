import numpy as np

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
        
        ind_SVM = np.where(labels_fin==label_no)
        
        if num < 1:
            N_train_cl[i] = np.floor(num*len(ind_SVM))
        else:
            N_train_cl[i] = num
        
        N_test_cl[i] = len(ind_SVM)-N_train_cl[i]
        
        #print(N_train_cl[i])
        # randomly chosen
        order = np.zeros((len(ind_SVM),1))
        
        for j in range(N_train_cl[i]):
            order[j] = 1
        order = order(np.random.permutation(range(len(ind_SVM))))
        
        class_ind_train.append(ind_SVM[order])

        not_order = np.ones((len(ind_SVM),1))
        for j in range(N_train_cl[i]):
            not_order[j] = 0
        
        class_ind_test.append(ind_SVM[not_order])
        
    return class_ind_train, class_ind_test