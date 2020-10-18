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
from FBCSP_support_function import loadDatasetD2, computeTrialD2, createTrialsDictD2
from FBCSP_Multiclass import FBCSP_Multiclass

import numpy as np

from sklearn.svm import SVC
from scipy.io import loadmat

import time

    
#%%
fs = 250
labels_name = {}
labels_name[769] = 'left'
labels_name[770] = 'right'
labels_name[771] = 'foot'
labels_name[772] = 'tongue'

print_var = True

for idx in range(1, 2):
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Train
    
    # Path for 4 classes dataset
    path_train = 'Dataset/D2/v1/Train'
    # path_train_label = 'Dataset/D2/v1/Train/1_label.mat'
    
    data, event_matrix = loadDatasetD2(path_train, idx)
    
    trials, labels = computeTrialD2(data, event_matrix, fs, remove_corrupt = True)
    
    trials_dict = createTrialsDictD2(trials, labels, labels_name)
    
    FBCSP_multi_clf = FBCSP_Multiclass(trials_dict, fs, print_var = print_var)
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Test set
    
    path_test = 'Dataset/D2/v1/Test'
    path_test_label = 'Dataset/D2/True Label/A0' + str(idx) + 'E.mat'
    
    data_test, event_matrix_test = loadDatasetD2(path_test, idx)
    trials_test, labels_test = computeTrialD2(data_test, event_matrix_test, fs)
    
    labels_true_value = np.squeeze(loadmat(path_test_label)['classlabel'])
    labels_predict_value = FBCSP_multi_clf.evaluateTrial(trials_test)
    
    labels_confront = np.zeros((len(labels_true_value), 2))
    labels_confront[:, 0] = labels_true_value
    labels_confront[:, 1] = labels_predict_value
    
   
