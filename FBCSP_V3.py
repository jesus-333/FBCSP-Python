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
from sklearn.feature_selection import mutual_info_classif as MIBIF


#%%

class FBCSP_V3():
    
    def __init__(self, data_dict, fs, freqs_band = None, filter_order = 3, n_features = 1, classifier = None):
        self.fs = fs
        self.trials_dict = data_dict
        self.n_features = n_features
        
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
        
        # Evaluate mutual information between features
        self.mutual_information_list = self.computeFeaturesMutualInformation()
        
        # Select features to use for classification
        for mutual_information in self.mutual_information_list:
            sorted_MI_index = np.flip(np.argsort(mutual_information))
            
                
    
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
    
         
    def trainClassifier(self, n_features = 1, train_ratio = 0.75, classifier = None):
        """
        Divide the data in train set and test set and used the data to train the classifier.

        Parameters
        ----------
        n_features : int, optional
            The number of mixture channel to use in the classifier. It must be even and at least as big as 2. The default is 2.
        train_ratio : doble, optional
            The proportion of the data to used as train dataset. The default is 0.75.
        classifier : sklearnn classifier, optional
            Classifier used for the problem. It must be a sklearn classifier. If no classfier was provided the fucntion use the LDA classifier.


        """
        self.n_features = n_features
        
        features_1, features_2 = self.extractFeatures(n_features)
        print(features_1.shape)
    
        # Save both features in a single data matrix
        data_matrix = np.zeros((features_1.shape[0] + features_2.shape[0], features_1.shape[1]))
        data_matrix[0:features_1.shape[0], :] = features_1
        data_matrix[features_1.shape[0]:, :] = features_2
        self.tmp_data_matrix = data_matrix
        
        # Create the label vector
        label = np.zeros(data_matrix.shape[0])
        label[0:features_1.shape[0]] = 1
        label[features_1.shape[0]:] = 2
        self.tmp_label = label
        
        # # Shuffle the data
        # perm = np.random.permutation(len(label))
        # label = label[perm]
        # data_matrix = data_matrix[perm, :]
        
        # # Select the portion of data used during training
        # if(train_ratio <= 0 or train_ratio >= 1): train_ratio = 0.75
        # index_training = int(data_matrix.shape[0] * train_ratio)
        # train_data = data_matrix[0:index_training, :]
        # train_label = label[0:index_training]
        # test_data = data_matrix[index_training:, :]
        # test_label = label[index_training:]
        # self.tmp_train = [train_data, train_label]
        # self.tmp_test = [test_data, test_label]
        
        # # Select classifier
        # if(classifier == None): self.classifier = LDA()
        # else: self.classifier = classifier
        
        # # Train Classifier
        # self.classifier.fit(train_data, train_label)
        # print("Accuracy on TRAIN set: ", self.classifier.score(train_data, train_label))
        
        # # Test parameters
        # print("Accuracy on TEST set: ", self.classifier.score(test_data, test_label), "\n")
        
    def extractFeatures(self, n_features = 1):
        a = 2
        
    
    def computeFeaturesMutualInformation(self):
        mutual_information_list = []
        
        idx = []
        
        for i in range(self.n_features): idx.append(i)
        for i in reversed(idx): idx.append(-(idx + 1))
        
        # Cycle through the different band
        for features_dict in self.features_band_list:
            # Retrieve features for that band
            keys = list[features_dict.keys()]
            feat_1 = features_dict[keys[0]]
            feat_2 = features_dict[keys[1]]
            
            # Save features in a single matrix
            all_features = np.zeros((feat_1.shape[0] + feat_2.shape[0], self.n_features * 2))            
            all_features[0:feat_1.shape[0], :] = feat_1
            all_features[feat_1.shape[0]:, :] = feat_2
            
            # Create label vector
            label = np.ones(all_features.shape[0])
            label[0:feat_1.shape[0]] = 2
            
            tmp_mutual_information = MIBIF(all_features, label)
            mutual_information_list.append(tmp_mutual_information)
            
        return mutual_information_list
            
    def extractFeaturesVVV(self, n_features = 1):
        """
        Extract the first n_features and the last n_features

        Parameters
        ----------
        n_features : int, optional
            Number of features to extract from each side. The default is 2.

        """
                
        # Creation of hte features matrix
        features_dict = self.features_band_list[0]
        keys = list(features_dict.keys())
        features_1_tmp = features_dict[keys[0]]
        features_2_tmp = features_dict[keys[1]]

        if(n_features < 1 or n_features > features_1_tmp.shape[1]): n_features = 2
        self.n_features = n_features

        features_1 = np.zeros((features_1_tmp.shape[0], 2 * n_features * len(self.features_band_list)))
        features_2 = np.zeros((features_2_tmp.shape[0], 2 * n_features * len(self.features_band_list)))             
        
        for features_dict, i in zip(self.features_band_list, range(len(self.features_band_list))):
            keys = list(features_dict.keys())
            features_1_tmp = features_dict[keys[0]]
            features_2_tmp = features_dict[keys[1]]
            
            # Select the first features
            for j in range(n_features):
                features_1[:, j * i] = features_1_tmp[:, j]
                features_2[:, j * i] = features_2_tmp[:, j]
                
            # Select the last features
            for j in range(1, n_features + 1):
                features_1[:, -j * i] = features_1_tmp[:, -j]
                features_2[:, -j * i] = features_2_tmp[:, -j]
        
        # features_1 = np.min(features_1, 1)
        # features_2 = np.min(features_2, 1)

        return features_1, features_2
    
    def getMinFeatures(self, features_vet, n_features, reverse):
        # Return variable of dimension "n. trials x n.features"
        features_ret = np.zeros((features_vet.shape[0], n_features))
        
        for row, i in zip(features_vet, range(len(features_vet.shape[0]))):
            tmp_row = np.sort(row)
            
            if(reverse): tmp_row = np.flip(tmp_row)
            
            features_ret[i, :] = tmp_row[0:n_features]
            
        return tmp_row
    
    def plotFeaturesSeparate(self, width = 0.3, figsize = (15, 30)):
        fig, axs = plt.subplots(len(self.features_band_list), 1, figsize = figsize)
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
            
            
    def plotFeatuersTogether(self, width = 0.3, figsize = (15, 10)):
        
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
        # print(tmp_y1.shape, y1.shape)
        
        x1 = np.linspace(1, len(y1), len(y1))
        x2 = x1 + 0.35
        
        fig, ax = plt.subplots(figsize = figsize)
        ax.bar(x1, y1, width = width, color = 'b', align='center')
        ax.bar(x2, y2, width = width, color = 'r', align='center')
        
        
        