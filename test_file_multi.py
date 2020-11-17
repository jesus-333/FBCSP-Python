# -*- coding: utf-8 -*-
"""
Test file for the FBCSP multiclass. Used the dataset 2a of the BCI Competition IV.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
from FBCSP_support_function import cleanWorkspaec
# cleanWorkspaec()

#%%
from FBCSP_support_function import loadDatasetD2, computeTrialD2, createTrialsDictD2, loadTrueLabel
from FBCSP_Multiclass import FBCSP_Multiclass

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from scipy.io import loadmat

import time

    
#%%
fs = 250
n_w = 2
n_features = 4

labels_name = {}
labels_name[769] = 'left'
labels_name[770] = 'right'
labels_name[771] = 'foot'
labels_name[772] = 'tongue'
labels_name[783] = 'unknown'
labels_name[1] = 'left'
labels_name[2] = 'right'
labels_name[3] = 'foot'
labels_name[4] = 'tongue'

print_var = False

accuracy_list = []

idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# idx_list = [1, 2, 3, 6, 7, 8]
# idx_list = [4]

for idx in idx_list:
# for idx in range(1, 10):
    print('Subject n.', str(idx))
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Train
    
    # Path for 4 classes dataset
    path_train = 'Dataset/D2/v1/Train'
    path_train_label = 'Dataset/D2/v1/True Label/A0' + str(idx) + 'T.mat'
    
    data, event_matrix = loadDatasetD2(path_train, idx)
    
    trials, labels = computeTrialD2(data, event_matrix, fs, remove_corrupt = False)
    labels_1 = np.squeeze(loadmat(path_train_label)['classlabel'])
    labels_2 = loadTrueLabel(path_train_label)
    
    trials_dict = createTrialsDictD2(trials, labels, labels_name)
    
    FBCSP_multi_clf = FBCSP_Multiclass(trials_dict, fs, print_var = print_var)
    # FBCSP_multi_clf = FBCSP_Multiclass(trials_dict, fs, classifier = SVC(kernel = 'rbf', probability = True), print_var = print_var)
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Test set
    
    path_test = 'Dataset/D2/v1/Test'
    path_test_label = 'Dataset/D2/v1/True Label/A0' + str(idx) + 'E.mat'
    
    data_test, event_matrix_test = loadDatasetD2(path_test, idx)
    trials_test, labels_test = computeTrialD2(data_test, event_matrix_test, fs)
    data_test = -data_test
    
    labels_true_value_1 = np.squeeze(loadmat(path_test_label)['classlabel'])
    labels_predict_value = FBCSP_multi_clf.evaluateTrial(trials_test)
    
    labels_confront = np.zeros((len(labels_true_value_1), 3))
    labels_confront[:, 0] = labels_true_value_1
    labels_confront[:, 1] = labels_predict_value
    
    a1 = FBCSP_multi_clf.pred_label_array
    a2 = FBCSP_multi_clf.pred_prob_array
    a3 = FBCSP_multi_clf.pred_prob_list
    
    # Percentage of correct prediction
    correct_prediction_1 = labels_predict_value[labels_predict_value == labels_true_value_1]
    perc_correct_1 = len(correct_prediction_1)/len(labels_true_value_1)
    accuracy_list.append(perc_correct_1)
        
    
    print('\nPercentage of correct prediction: ', perc_correct_1)
    print("# # # # # # # # # # # # # # # # # # # # #\n")
    
#%%

plt.plot(data[:, 0])
plt.plot(data_test[:, 0])
