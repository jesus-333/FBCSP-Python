# -*- coding: utf-8 -*-
"""
Contain the multicass extension of the FBCSP algorithm.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
import numpy as np

import matplotlib.pyplot as plt

# from FBCSP_V3 import FBCSP_V3
from FBCSP_V4 import FBCSP_V4

import time

#%%

class FBCSP_Multiclass():
    
    def __init__(self, data_dict, fs, n_w = 2, n_features = 4, freqs_band = None, filter_order = 3, classifier = None, print_var = False):
        self.print_var = print_var
        
        if(print_var): start = time.time()
        
        self.fs = fs
        self.nw = n_w
        self.n_features = n_features
        
        # List of classifier
        self.FBCSP_list = []
        self.binary_dict_list = []
        
        # Cycle through different class
        for key, i in zip(data_dict.keys(), range(len(data_dict))):
            if(print_var): 
                start_cycle = time.time()
                print("Iteration number: ", i, " - ", key, "vs Others")
            
            # Create the binary dict
            tmp_binary_dict = self.createBinaryDict(data_dict, key)
            self.binary_dict_list.append(tmp_binary_dict)
            # print(tmp_binary_dict.keys())
            if(print_var): print("Binary Dicionary create")
            
            # Create the FBCSP object and train it
            if(classifier != None): tmp_FBCSP_clf = FBCSP_V4(tmp_binary_dict, fs, n_w = n_w, n_features = n_features, classifier = classifier, print_var = print_var)
            else: tmp_FBCSP_clf = FBCSP_V4(tmp_binary_dict, fs, n_w = n_w, n_features = n_features, print_var = print_var)
            if(print_var): print("FBCSP object creation and training complete")
            
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
        for tmp_key in dict_with_other_trials: tmp_list.append(dict_with_other_trials[tmp_key])
        
        for i in range(len(tmp_list) - 1):
            if(i == 0):
                trials_2 = np.concatenate([tmp_list[0], tmp_list[1]], axis = 0)
            else: 
                trials_2 = np.concatenate([trials_2, tmp_list[i + 1]], axis = 0)
                    
        # Create the binary dictionary
        binary_dict = {}
        binary_dict[key] = trials_1
        binary_dict['zzz_other'] = trials_2

        return binary_dict
    
    
    def evaluateTrial(self, trials_matrix):
        # Variable to track the predicted label for each classifier
        self.pred_label_array = np.zeros((trials_matrix.shape[0], len(self.FBCSP_list)))
        
        # Matrix wiht the probability of the prediction for each classifier
        self.pred_prob_array = np.zeros((trials_matrix.shape[0], len(self.FBCSP_list) * 2))
        
        # List with the probability of preditiction for each classifier
        self.pred_prob_list = []
        label_return = np.zeros(trials_matrix.shape[0])
        
        # Evaluate the trial(s)
        for clf, i in zip(self.FBCSP_list, range(len(self.FBCSP_list))):
            # Predict label
            # print(i)
            label, prob = clf.evaluateTrial(trials_matrix)
            
            # Save the results
            self.pred_label_array[:, i] = label
            self.pred_prob_array[:, (i*2):(i*2+2)] = prob
            self.pred_prob_list.append(prob)
            
        # Check the results (Iteration through trials)
        for i in range(trials_matrix.shape[0]):
            row = self.pred_label_array[i,:]
            
            # Check if there's a conflict between label
            if(len(row[row == 1]) > 1):
                # Case 1: The trials is classified as class 1 (specific class) in more than 1 classifier
                # Search the classification with the highest probability
                
                # Variables to track the highest probability
                max_prob = -1
                max_prob_position = -1
                
                # Cycle through the element of the classifier results
                for j in range(len(row)):
                    # If the element is classified as a specific class
                    if(row[j] == 1):
                        # Retrieve the probability that belongs to that class
                        actual_prob = self.pred_prob_list[j][i, 0]
                        
                        # Check if the probability is bigger than the probability of the last selected element
                        if(actual_prob > max_prob):
                            max_prob = actual_prob
                            max_prob_position = j
                
                # Select the most probably label
                # (+1 is added to have class 1 with label 1, class 2 with label 2 etc)
                label_return[i] = max_prob_position + 1
                
            elif(np.unique(row)[0] == 2):
                # Case 2: The trials is classified as 2 (other classes) in all the classifier
                # Search the classifier more undecided and use it as label
                
                # Variable to track the more undecided classifier
                und_prob = 1
                und_prob_position = -1
                
                # Cycle through the element of the classifier results
                for j in range(len(row)):
               
                    # Retrieve the probability that belongs to that class)
                    actual_prob = self.pred_prob_list[j][i, 0]
                    
                    # Evaluate how close is to 0.5
                    nearness = abs(0.5 - actual_prob)
                    
                    # Check if the probability is bigger than the probability of the last selected element
                    if(nearness < und_prob):
                        und_prob = nearness
                        und_prob_position = j
                        
                # Select the more undecided classifier
                # (+1 is added to have class 1 with label 1, class 2 with label 2 etc)
                label_return[i] = und_prob_position + 1
                
            else:
                # Case 3: The trials is classified as class 1 (specific class) in only 1 classifier
                
                # Note that since class 1 (specific class) is codified as 1 and class 2(all other classes) are codified as 2 with search the element with the minimum value.
                label_return[i] = np.argmin(row) + 1
    
        # Return the results
        return label_return
    
    
                    
                
                
            
        