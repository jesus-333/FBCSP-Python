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
tmp_string = 'abcdefg'
tmp_string = 'e'

for idx in tmp_string:
    print(idx)

    # Data load and trials extraction (Works only on dataset IV-1-a of BCI competition)
    
    # Path for 2 classes dataset
    path = 'Dataset/D1_100Hz/v1/Train/BCICIV_calib_ds1'
    # path = 'Dataset/D1_100Hz/Test/BCICIV_eval_ds1'
    # idx = 'a'
    
    data, labels, cue_position, other_info = loadDatasetD1_100Hz(path, idx, type_dataset = 'train')
    
    fs = other_info['sample_rate']
    trials_dict = computeTrialD1_100Hz(data, cue_position, labels, fs,other_info['class_label'])
    
    
    FBCSP_clf = FBCSP_V4(trials_dict, fs, n_features = 3, print_var = True)
    # FBCSP_clf = FBCSP_V3(trials_dict, fs, n_features = 3, classifier = SVC(kernel = 'linear'))
    
    # FBCSP_clf.plotFeaturesSeparateTraining()
    FBCSP_clf.plotFeaturesScatterTraining(selected_features = [0, -1])
    FBCSP_clf.plotFeaturesScatterTraining(selected_features = [0, 1])
    
    trials_test = trials_dict['left']
    a, b = FBCSP_clf.evaluateTrial(trials_test)
    
    # print("n_features x 2 = ", FBCSP_clf.n_features * 2)
    # print("Features selected = ", len(FBCSP_clf.classifier_features))
    
    # print(len(a[a == 1])/len(a))
    # print(len(a[a == 2])/len(a))