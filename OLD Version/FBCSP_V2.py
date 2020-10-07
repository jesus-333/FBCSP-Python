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

import scipy.signal
import scipy.linalg as la

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%%

class FBCSP_V2():
    
    def __init__(self, data_dict, fs, freqs_band = None, filter_order = 3):
        self.fs = fs
        self.train_sklearn = False
        self.train_LDA = False
        self.trials_dict = data_dict
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #Filter data section
        
        # Filtered signal list
        self.filtered_band_signal_list = []
        
        # Frequencies band
        if(freqs_band == None): self.freqs = np.linspace(4, 40, 10)
        else: self.freqs =  freqs_band
        
        self.filterBankFunction(filter_order)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
        # CSP filters evaluation
        self.evaluateW()
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        
    def filterBankFunction(self, filter_order = 3):
        """
        Function that apply fhe fitlering for each pair of frequencies in the list self.freqs.
        The results are saved in a list called self.filtered_band_signal_list. Each element of the list is a diciotinary with key the label of the various class.

        Parameters
        ----------
        filter_order : int, optional
            The order of the filter. The default is 3.

        """
        
        # Cycle for the frequency bands
        for i in range(len(self.freqs) - 1):  
            # Dict for selected band that will contain the various filtered signals
            filt_trial_dict = {}
            
            # "Create" the band
            band = [self.freqs[i], self.freqs[i+1]]
            
            # Cycle for the classes
            for key in self.trials_dict.keys(): 
                # Filter the signal in each class for the selected frequency band
                filt_trial_dict[key] = self.bandFilterTrials(self.trials_dict[key], band[0], band[1], filter_order = filter_order)
            
            # Save the filtered signal in the list
            self.filtered_band_signal_list.append(filt_trial_dict)
        
    
    def bandFilterTrials(self, trials_matrix, low_f, high_f, filter_order = 3):
        """
        Applying a pass-band fitlering to the data. The filter implementation was done with scipy.signal
    
        Parameters
        ----------
        trials_matrix : numpy matrix
            Numpy matrix with the various EEG trials. The dimensions of the matrix must be n_trial x n_channel x n_samples
        fs : int/double
            Frequency sampling.
        low_f : int/double
            Low band of the pass band filter.
        high_f : int/double
            High band of the pass band filter..
        filter_order : int, optional
            Order of the filter. The default is 3.
    
        Returns
        -------
        filter_trails_matrix : numpy matrix
             Numpy matrix with the various filtered EEG trials. The dimensions of the matrix must be n_trial x n_channel x n_samples.
    
        """
        
        # Evaluate low buond and high bound in the [0, 1] range
        low_bound = low_f / (self.fs/2)
        high_bound = high_f / (self.fs/2)
        
        # Check input data
        if(low_bound < 0): low_bound = 0
        if(high_bound > 1): high_bound = 1
        if(low_bound > high_bound): low_bound, high_bound = high_bound, low_bound
        if(low_bound == high_bound): low_bound, high_bound = 0, 1
        
        b, a = scipy.signal.butter(filter_order, [low_bound, high_bound], 'bandpass')
          
        return scipy.signal.filtfilt(b, a, trials_matrix)
    
    def evaluateW(self):
        """
        Evaluate the spatial filter of the CSP algorithm for each filtered signal inside self.filtered_band_signal_list
        Results are saved inside self.W_list_band.
        So for each filter band the algorithm create n spatial filter where n is the number of classes. Each filter is used to maximize the variance between a class and all other classes.
        If n = 2 only a spatial filter is evaluated.
        
        """
        
        self.W_list_band = []
        
        for filt_trial_dict in self.filtered_band_signal_list:
            # Retrieve the key (class)
            keys = list(filt_trial_dict.keys())
            
            # List for the filter for each class
            W_list_class = []
            
            for key in keys:
                trials_1, trials_2 = self.retrieveBinaryTrials(filt_trial_dict, key)
            
                # Evaluate covariance matrix for the two classes
                cov_1 = self.trialCovariance(trials_1)
                cov_2 = self.trialCovariance(trials_2)
                R = cov_1 + cov_2
                
                # Evaluate whitening matrix
                P = self.whitening(R)
                
                # The mean covariance matrices may now be transformed
                cov_1_white = np.dot(P, np.dot(cov_1, np.transpose(P)))
                cov_2_white = np.dot(P, np.dot(cov_2, np.transpose(P)))
                
                # Since CSP requires the eigenvalues and eigenvector be sorted in descending order we find and sort the generalized eigenvalues and eigenvector
                E, U = la.eig(cov_1_white, cov_2_white)
                order = np.argsort(E)
                order = order[::-1]
                E = E[order]
                U = U[:, order]
                
                # The projection matrix (the spatial filter) may now be obtained
                W = np.dot(np.transpose(U), P)
                
                # Save the filter for each class
                W_list_class.append(W)
                
                if(len(keys) == 2): break
            
            self.W_list_band.append(W_list_class)
      
            
    def retrieveBinaryTrials(self, filt_trial_dict, key):
        """
        Function that return all the trials of a class on trials 1 and all the trials of all other classes in trials 2

        Parameters
        ----------
        filt_trial_dict : dict
            Input dicionary. The key must be the label of the classes. Each item is all the trials of the corresponding class
        key : dictionary key
            Key for trials_1.

        Returns
        -------
        trials_1 : Numpy 3D matrix
            All the trials corresponding to the key passes.
        trials_2 : Numpy 3D matrix
            All other trials.

        """
             
        # Retrieve trial associated with key
        trials_1 = filt_trial_dict[key]
        
        # Retrieve all other trials
        dict_with_other_trials = {k:v for k,v in filt_trial_dict.items() if k not in [key]}
        
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

        return trials_1, trials_2
            
    
    def trialCovariance(self, trials):
        """
        Calculate the covariance for each trial and return their average
    
        Parameters
        ----------
        trials : numpy 3D-matrix
            Trial matrix. The dimensions must be trials x channel x samples
    
        Returns
        -------
        mean_cov : Numpy matrix
            Mean of the covariance alongside channels.
    
        """
        
        n_trials, n_channels, n_samples = trials.shape
        
        covariance_matrix = np.zeros((n_trials, n_channels, n_channels))
        
        for i in range(trials.shape[0]):
            trial = trials[i, :, :]
            covariance_matrix[i, :, :] = np.cov(trial)
            
        mean_cov = np.mean(covariance_matrix, 0)
            
        return mean_cov
    
    
    def whitening(self, sigma, mode = 2):
        """
        Calculate the whitening matrix for the input matrix sigma
    
        Parameters
        ----------
        sigma : Numpy square matrix
            Input matrix.
        mode : int, optional
            Select how to evaluate the whitening matrix. The default is 1.
    
        Returns
        -------
        x : Numpy square matrix
            Whitening matrix.
        """
        [u, s, vh] = np.linalg.svd(sigma)
        
          
        if(mode != 1 and mode != 2): mode == 1
        
        if(mode == 1):
            # Whitening constant: prevents division by zero
            epsilon = 1e-5
            
            # ZCA Whitening matrix: U * Lambda * U'
            x = np.dot(u, np.dot(np.diag(1.0/np.sqrt(s + epsilon)), u.T))
        else:
            # eigenvalue decomposition of the covariance matrix
            d, V = np.linalg.eigh(sigma)
            fudge = 10E-18
         
            # A fudge factor can be used so that eigenvectors associated with small eigenvalues do not get overamplified.
            D = np.diag(1. / np.sqrt(d+fudge))
         
            # whitening matrix
            x = np.dot(np.dot(V, D), V.T)
            
        return x
    
    
    