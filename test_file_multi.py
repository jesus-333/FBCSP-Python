# -*- coding: utf-8 -*-
"""
Contain the implementation of the FBCSP algorithm. Developed for the train part of dataset IV-1-a of BCI competition.

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
