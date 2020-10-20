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

for idx in range(1, 10):
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
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Test set
    
    path_test = 'Dataset/D2/v1/Test'
    path_test_label = 'Dataset/D2/v1/True Label/A0' + str(idx) + 'E.mat'
    
    data_test, event_matrix_test = loadDatasetD2(path_test, idx)
    trials_test, labels_test = computeTrialD2(data_test, event_matrix_test, fs)
    
    labels_true_value_1 = np.squeeze(loadmat(path_test_label)['classlabel'])
    labels_true_value_2 = loadTrueLabel(path_test_label)
    labels_predict_value = FBCSP_multi_clf.evaluateTrial(trials_test)
    
    labels_confront = np.zeros((len(labels_true_value_1), 3))
    labels_confront[:, 0] = labels_true_value_1
    labels_confront[:, 1] = labels_predict_value
    labels_confront[:, 2] = labels_true_value_2
    
    a1 = FBCSP_multi_clf.pred_label_array
    a2 = FBCSP_multi_clf.pred_prob_array
    a3 = FBCSP_multi_clf.pred_prob_list
    
    # Percentage of correct prediction (1)
    correct_prediction_1 = labels_predict_value[labels_predict_value == labels_true_value_1]
    perc_correct_1 = len(correct_prediction_1)/len(labels_true_value_1)
    
    # Percentage of correct prediction (2)
    correct_prediction_2 = labels_predict_value[labels_predict_value == labels_true_value_2]
    perc_correct_2 = len(correct_prediction_2)/len(labels_true_value_2)
    
    
    print('\nPercentage of correct prediction (1): ', perc_correct_1)
    print('Percentage of correct prediction (2): ', perc_correct_2)
    print("# # # # # # # # # # # # # # # # # # # # #\n")
    

