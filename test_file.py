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
from FBCSP_support_function import loadDataset100Hz, computeTrial
from FBCSP import FBCSP

import numpy as np

from sklearn.svm import SVC

#%%
tmp_string = 'abcdefg'

for idx in tmp_string:

    #%% Data load
    
    path = 'Dataset/D1_100Hz/Train/BCICIV_calib_ds1'
    # path = 'Dataset/D1_100Hz/Test/BCICIV_eval_ds1'
    # idx = 'a'
    
    plot_var = False
    
    data, labels, cue_position, other_info = loadDataset100Hz(path, idx, type_dataset = 'train')
    
    #%% Extract trials from data (Works only on dataset IV-1-a of BCI competition)
    
    fs = other_info['sample_rate']
    
    trials_dict = computeTrial(data, cue_position, labels, fs,other_info['class_label'])
    
    #%%
    
    FBCSP_clf = FBCSP(trials_dict, fs)
    
    # CSP_clf.plotFeatures()
    # CSP_clf.plotPSD(15, 12)
    
    # FBCSP_clf.trainClassifier()
    # FBCSP_clf.trainLDA()
    
    # CSP_clf.trainClassifier(classifier = SVC(kernel = 'linear'))
    
    # FBCSP_clf.plotFeaturesScatter()