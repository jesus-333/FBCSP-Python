# -*- coding: utf-8 -*-
"""
Test for the binary FBCSP algorithm. I used the dataset 1 fo BCI competition IV.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
from FBCSP_support_function import cleanWorkspaec
# cleanWorkspaec()

#%%
from FBCSP_support_function import loadDatasetD1_100Hz, computeTrialD1_100Hz
from FBCSP_V3 import FBCSP_V3
from FBCSP_V4 import FBCSP_V4

import numpy as np

from sklearn.svm import SVC

import time

#%%
n_w = 2
n_features = 4

tmp_string = 'abcdefg'
tmp_string = 'e'

path = 'Dataset/D1_100Hz/Train/BCICIV_calib_ds1'
high_sampling_dataset = False

# path = 'Dataset/D1_1000Hz/Train/BCICIV_calib_ds1'
# high_sampling_dataset = True

for idx in tmp_string:
    print(idx)

    # Data load and trials extraction (Works only on dataset IV-1-a of BCI competition)
    
    # Path for 2 classes dataset
    # path = 'Dataset/D1_100Hz/Test/BCICIV_eval_ds1'
    # idx = 'a'
    
    data, labels, cue_position, other_info = loadDatasetD1_100Hz(path, idx, type_dataset = 'train', high_sampling_dataset= high_sampling_dataset)
    
    fs = other_info['sample_rate']
    trials_dict = computeTrialD1_100Hz(data, cue_position, labels, fs,other_info['class_label'])
    
    # FBCSP_clf = FBCSP_V3(trials_dict, fs, n_features = 3, classifier = SVC(kernel = 'linear'))    
    
    FBCSP_clf = FBCSP_V4(trials_dict, fs, n_w = 2, n_features = n_features, print_var = True)
    # FBCSP_clf = FBCSP_V4(trials_dict, fs, n_w = 2, n_features = n_features, classifier = SVC(kernel = 'linear'), print_var = True)
    
    # FBCSP_clf.plotFeaturesSeparateTraining()
    FBCSP_clf.plotFeaturesScatterTraining(selected_features = [0, -1])
    # FBCSP_clf.plotFeaturesScatterTraining(selected_features = [-1, 1])
    
    # for idx_2 in tmp_string:
    #     trials_test = trials_dict['left']
    #     a, b = FBCSP_clf.evaluateTrial(trials_test)
        
