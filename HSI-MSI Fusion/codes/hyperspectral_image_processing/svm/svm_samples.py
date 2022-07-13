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