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
from FBCSP_support_function import loadDatasetD1_100Hz, computeTrialD1_100Hz, loadDatasetD2, computeTrialD2, createTrialsDictD2
from FBCSP_V3 import FBCSP_V3

import numpy as np

from sklearn.svm import SVC

import time

#%%
# tmp_string = 'abcdefg'
tmp_string = 'a'

for idx in tmp_string:

    # Data load and trials extraction (Works only on dataset IV-1-a of BCI competition)
    
    # Path for 2 classes dataset
    path = 'Dataset/D1_100Hz/Train/BCICIV_calib_ds1'
    # path = 'Dataset/D1_100Hz/Test/BCICIV_eval_ds1'
    # idx = 'a'
    
    data, labels, cue_position, other_info = loadDatasetD1_100Hz(path, idx, type_dataset = 'train')
    
    fs = other_info['sample_rate']
    trials_dict = computeTrialD1_100Hz(data, cue_position, labels, fs,other_info['class_label'])
    
    
    FBCSP_clf = FBCSP_V3(trials_dict, fs)
    
    # FBCSP_clf.plotFeaturesSeparate()
    FBCSP_clf.plotFeatuersTogetherV1()
#     CSP_clf.plotPSD(15, 12)
    
#     FBCSP_clf.trainClassifier()
#     FBCSP_clf.trainLDA()
    
#     CSP_clf.trainClassifier(classifier = SVC(kernel = 'linear'))
    
#     FBCSP_clf.plotFeaturesScatter()
    
    
#%%
# fs = 250
# label_name = {}
# label_name[769] = 'left'
# label_name[770] = 'right'
# label_name[771] = 'foot'
# label_name[772] = 'tongue'

# for idx in range(1, 2):
    
#     # Path for 4 classes dataset
#     path = 'C:/Users/albi2/OneDrive/Documenti/GitHub/Deep-Learning-For-EEG-Classification/Dataset/D2/v1/Train'
#     # path = 'C:/Users/albi2/OneDrive/Documenti/GitHub/Deep-Learning-For-EEG-Classification/Dataset/D2/v1/Train/1_label.mat'
    
#     start = time.time()
#     data, event_matrix = loadDatasetD2(path, idx)
#     print("matrix load ", time.time() - start)
    
#     start = time.time()
#     trials, label = computeTrialD2(data, event_matrix)
#     print("trial done ", time.time() - start)
    
#     start = time.time()
#     trials_dict = createTrialsDictD2(trials, label, label_name)
#     print("dict created ", time.time() - start)
    
#     start = time.time()
#     FBCSP_clf = FBCSP_V2(trials_dict, fs)
#     print("FBCSP evaluated ", time.time() - start)