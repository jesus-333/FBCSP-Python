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

import time

    
#%%
fs = 250
label_name = {}
label_name[769] = 'left'
label_name[770] = 'right'
label_name[771] = 'foot'
label_name[772] = 'tongue'

print_var = True

for idx in range(1, 2):
    
    # Path for 4 classes dataset
    path = 'C:/Users/albi2/OneDrive/Documenti/GitHub/Deep-Learning-For-EEG-Classification/Dataset/D2/v1/Train'
    # path = 'C:/Users/albi2/OneDrive/Documenti/GitHub/Deep-Learning-For-EEG-Classification/Dataset/D2/v1/Train/1_label.mat'
    
    data, event_matrix = loadDatasetD2(path, idx)
    
    trials, label = computeTrialD2(data, event_matrix)
    
    trials_dict = createTrialsDictD2(trials, label, label_name)
    
    FBCSP_multi_clf = FBCSP_Multiclass(trials_dict, fs, print_var = print_var)
    
    key_list = list(trials_dict.keys())
    percentage_total = 0
    for i in range(len(key_list)):
        key = key_list[i]
        tmp_trial = trials_dict[key]
        label = FBCSP_multi_clf.evaluateTrial(tmp_trial)
        
        percentage_correct_class = len(label[label == (i + 1)]) / tmp_trial.shape[0]
        percentage_total = percentage_total + percentage_correct_class * (tmp_trial.shape[0] / trials.shape[0])
    print("Percentage of total trials correctly classified: ", percentage_total)
        
    # tmp_trial = trials_dict['tongue']
    # tmp_trial = trials_dict['tongue'][0:1, :, :]
    # label = FBCSP_multi_clf.evaluateTrial(tmp_trial)
    
    # a = FBCSP_multi_clf.binary_dict_list
    # a1 = FBCSP_multi_clf.pred_label_list
    # a2 = FBCSP_multi_clf.pred_prob_list
    
    # b1 = FBCSP_multi_clf.pred_label_array
    # b2 = FBCSP_multi_clf.pred_prob_list
