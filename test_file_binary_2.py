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
# from FBCSP_V3 import FBCSP_V3
from FBCSP_V4 import FBCSP_V4

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from scipy.io import loadmat

import time

#%%
n_w = 2
n_features = 4
n_trials_test = 230
print_var = True

tmp_string = 'abcdefg'
tmp_string = 'bcdefg'
# tmp_string = 'c'

# path = 'Dataset/D1_100Hz/v1/Train/BCICIV_calib_ds1'
# high_sampling_dataset = False

path = 'Dataset/D1_1000Hz/v1/Train/BCICIV_calib_ds1'
high_sampling_dataset = True

for idx in tmp_string:
    print(idx)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Data load and trials extraction (Works only on dataset IV-1-a of BCI competition)
    data, labels, cue_position, other_info = loadDatasetD1_100Hz(path, idx, type_dataset = 'train', high_sampling_dataset= high_sampling_dataset)
    
    fs = other_info['sample_rate']
    trials_dict = computeTrialD1_100Hz(data, cue_position, labels, fs,other_info['class_label'])
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Training
    
    # FBCSP_clf = FBCSP_V3(trials_dict, fs, n_features = n_features, print_var = True)
    # FBCSP_clf = FBCSP_V3(trials_dict, fs, n_w = n_w, n_features = n_features, classifier = SVC(kernel = 'linear'))    
    
    FBCSP_clf = FBCSP_V4(trials_dict, fs, n_w = n_w, n_features = n_features, print_var = print_var)
    # FBCSP_clf = FBCSP_V4(trials_dict, fs, n_w = n_w, n_features = n_features, classifier = SVC(kernel = 'linear'), print_var = True)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Features plotting
    
    # FBCSP_clf.plotFeaturesSeparateTraining()
    FBCSP_clf.plotFeaturesScatterTraining(selected_features = [0, -1])
    # FBCSP_clf.plotFeaturesScatterTraining(selected_features = [-1, 1])
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Testing
    if(high_sampling_dataset): path_testing = 'Dataset/D1_1000Hz/v2/Test/' + idx + '/'
    else: path_testing = 'Dataset/D1_100Hz/v2/Test/' + idx + '/'
    
    # Retrieve original label
    labels_test_original = np.squeeze(loadmat(path_testing + 'label.mat')['final_label'])
    labels_test_original = labels_test_original[0:n_trials_test]
    labels_test_original[labels_test_original == 1] = 2
    labels_test_original[labels_test_original == -1] = 1
    
    # Create vector for predict label
    labels_test_predict = np.zeros(n_trials_test)
    
    prob_matrix = np.zeros((labels_test_predict.shape[0], 2))
    prob_list = []
    
    for i in range(1, n_trials_test + 1):
        # if(i % 5): print('Test percentage: ', round(i/n_trials_test * 100, 2), '%')
        
        # Retrieve trial
        trial_test = loadmat(path_testing + str(i) + '_data.mat')['trial']
        trial_test = trial_test.T
        trial_test = np.expand_dims(trial_test, axis = 0)

        # Evaluate trial
        tmp_label, tmp_prob = FBCSP_clf.evaluateTrial(trial_test)
        
        # Save label
        labels_test_predict[i - 1] = tmp_label
        
        # Save probability
        prob_matrix[i - 1, :] = tmp_prob
    
    prob_list.append(tmp_prob)
        
    # Confront matrix for visual 
    labels_matrix = np.zeros((labels_test_predict.shape[0], 2))
    labels_matrix[:, 0] = labels_test_original
    labels_matrix[:, 1] = labels_test_predict
    
    # Percentage of correct prediction
    correct_prediction = labels_test_predict[labels_test_predict == labels_test_original]
    perc_correct = len(correct_prediction)/len(labels_test_original)
    
    print('Percentage of correct prediction: ', perc_correct)
    print("- - - - - - - - - - - - - - - - - - - - -")