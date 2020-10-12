# -*- coding: utf-8 -*-
"""
Contain the implementation of the CSP algorithm. Developed for the train part of dataset IV-1-a of BCI competition.
This version (V2) implement the algorithm for data with two classes.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
import numpy as np

import matplotlib.pyplot as plt

from FBCSP_V3 import FBCSP_V3

import time

#%%

class FBCSP_Multiclass():
    
    def __init__(self, data_dict, fs, freqs_band = None, filter_order = 3, n_features = 3, classifier = None, print_var = False):
        if(print_var): start = time.time()
        
        self.fs = fs
        self.n_features = n_features
        
        # List of classifier
        self.FBCSP_list = []
        
        # Cycle through different class
        for key, i in zip(data_dict.keys(), range(len(data_dict))):
            if(print_var): 
                start_cycle = time.time()
                print("Iteration number: ", i, " - ", key, "vs Others")
            
            # Create the binary dict
            tmp_binary_dict = self.createBinaryDict(data_dict, key)
            if(print_var): print("Binary Dicionary create")
            
            # Create the FBCSP object and train it
            if(classifier != None): tmp_FBCSP_clf = FBCSP_V3(tmp_binary_dict, fs, n_features = n_features, classifier = classifier, print_var = print_var)
            else: tmp_FBCSP_clf = FBCSP_V3(tmp_binary_dict, fs, n_features = n_features, print_var = print_var)
            if(print_var): print("FBCSP object and training complete")
            
            # Add the classifier to the list
            self.FBCSP_list.append(tmp_FBCSP_clf)
            
            if(print_var): 
                print("Cycle at: ", (i + 1)/len(data_dict) * 100, "%")
                print("Execution time of the cycle: {:.4f}".format(time.time() - start_cycle))
                print("- - - - - - - - - - - - - - - - - - - - - - - - \n")
                
                
        if(print_var): print("Time to execute: ", time.time() - start)
            
    
    def createBinaryDict(self, data_dict, key):
        """
        This method receive a n class dict, where n is the number of classes of EEG signal and return a binary dict
        The element of the passed key will be the first item and all other element of the other keys will be the other item.

        """
        # Retrieve trial associated with key
        trials_1 = data_dict[key]
        
        # Retrieve all other trials
        dict_with_other_trials = {k:v for k,v in data_dict.items() if k not in [key]}
        
        # Convert them in a numpy array
        tmp_list = []
        for key in dict_with_other_trials: tmp_list.append(dict_with_other_trials[key])
        
        if(len(tmp_list) == 1):
            tmp_key = list(dict_with_other_trials.keys())[0]
            trials_2 = dict_with_other_trials[tmp_key]
        else:
            for i in range(len(tmp_list) - 1):
                if(i == 0):
                    trials_2 = np.concatenate([tmp_list[0], tmp_list[1]], axis = 0)
                else: 
                    trials_2 = np.concatenate([trials_2, tmp_list[i + 1]], axis = 0)
                    
        # Create the binary dictionary
        binary_dict = {}
        binary_dict[key] = trials_1
        binary_dict['other'] = trials_2

        return binary_dict
    
    
    def evaluateTrial(self, trials_matrix):
        self.prob_list = []
        
        for clf in self.FBCSP_list:
            self.prob_predict = clf.evaluateTrial(trials_matrix)[1]
            
        
        
        

        