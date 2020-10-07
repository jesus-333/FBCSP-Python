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

class FBCSP_V3():
    
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
        # CSP filters evaluation and application
        
        # CSP filter evaluation
        self.W_list_band = []
        self.evaluateW()
        
        # CSP filter application
        self.features_band_list = []
        self.spatialFilteringAndFeatureExtraction()
        
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
        """
        
        for filt_trial_dict in self.filtered_band_signal_list:
            # Retrieve the key (class)
            keys = list(filt_trial_dict.keys())
            
            
            keys = list(filt_trial_dict.keys())
            trials_1 = filt_trial_dict[keys[0]]
            trials_2 = filt_trial_dict[keys[1]]
        
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
            
            self.W_list_band.append(W)
      
    
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
    
    def spatialFilteringAndFeatureExtraction(self):
        # Cycle through frequency band and relative CSP filter
        for filt_trial_dict, W in zip(self.filtered_band_signal_list, self.W_list_band):
            # Features dict for the current frequency band
            features_dict = {}
            
            # Cycle through the classes
            for key in filt_trial_dict.keys():
                # Applying spatial filter
                tmp_trial = self.spatialFilteringW(filt_trial_dict[key], W)
                
                # Features evaluation
                features_dict[key] = self.logVarEvaluation(tmp_trial)
            
            self.features_band_list.append(features_dict)
                
    
    def spatialFilteringW(self, trials, W):
        # Allocate memory for the spatial fitlered trials
        trials_csp = np.zeros(trials.shape)
        
        # Apply spatial fitler
        for i in range(trials.shape[0]): trials_csp[i, :, :] = W.dot(trials[i, :, :])
            
        return trials_csp
    
    
    def logVarEvaluation(self, trials):
        """
        Evaluate the log (logarithm) var (variance) of the trial matrix along the samples axis.
        The sample axis is the axis number 2, counting axis as 0,1,2. 
    
        Parameters
        ----------
        trials : numpy 3D-matrix
            Trial matrix. The dimensions must be trials x channel x samples
    
        Returns
        -------
        features : Numpy 2D-matrix
            Return the features matrix. DImension will be trials x channel
    
        """
        features = np.var(trials, 2)
        features = np.log(features)
        
        return features
    
    
    def plotFeaturesSeparate(self, width = 0.3):
        fig, axs = plt.subplots(len(self.features_band_list), 1, figsize = (15, 10))
        for features_dict, ax in zip(self.features_band_list, axs):
            keys = list(features_dict.keys())
            features_1 = features_dict[keys[0]]
            features_2 = features_dict[keys[1]]
            
            x1 = np.linspace(1, features_1.shape[1], features_1.shape[1])
            x2 = x1 + 0.35
            
            y1 = np.mean(features_1, 0)
            y2 = np.mean(features_2, 0)
            
            ax.bar(x1, y1, width = width, color = 'b', align='center')
            ax.bar(x2, y2, width = width, color = 'r', align='center')
            ax.set_xlim(0.5, 59.5)
            
            
    def plotFeatuersTogetherV1(self, width = 0.3, figsize = (15, 10)):
        
        y1 = np.zeros(0)
        y2 = np.zeros(0)
        for features_dict in self.features_band_list:
            keys = list(features_dict.keys())
            features_1 = features_dict[keys[0]]
            features_2 = features_dict[keys[1]]
        
            tmp_y1 = np.mean(features_1, 0)
            tmp_y2 = np.mean(features_2, 0)
            
            y1 = np.concatenate((y1, tmp_y1))
            y2 = np.concatenate((y2, tmp_y2))
            
        y1 = np.sort(y1)
        y2 = np.flip(np.sort(y2))
        
        x1 = np.linspace(1, len(y1), len(y1))
        x2 = x1 + 0.35
        
        fig, ax = plt.subplots(figsize = figsize)
        ax.bar(x1, y1, width = width, color = 'b', align='center')
        ax.bar(x2, y2, width = width, color = 'r', align='center')
        
        
    def plotFeatuersTogetherV2(self, width = 0.3, figsize = (15, 10)):
        
        y1 = np.zeros(0)
        y2 = np.zeros(0)
        for features_dict in self.features_band_list:
            keys = list(features_dict.keys())
            features_1 = features_dict[keys[0]]
            features_2 = features_dict[keys[1]]
        
            tmp_y1 = np.mean(features_1, 0)
            tmp_y2 = np.mean(features_2, 0)
            
            y1 = np.concatenate((y1, tmp_y1))
            y2 = np.concatenate((y2, tmp_y2))
            
        y1 = np.sort(y1)
        y2 = np.flip(np.sort(y2))
        
        x1 = np.linspace(1, len(y1), len(y1))
        x2 = x1 + 0.35
        
        fig, ax = plt.subplots(figsize = figsize)
        ax.bar(x1, y1, width = width, color = 'b', align='center')
        ax.bar(x2, y2, width = width, color = 'r', align='center')
        
    
    
    