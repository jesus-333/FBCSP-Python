# -*- coding: utf-8 -*-
"""
Contain the implementation of the FBCSP algorithm (binary version). 
This version (V3) is the correct one and the only fully functional. 

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
    
    def __init__(self, data_dict, fs, freqs_band = None, filter_order = 3, n_features = 2, classifier = None, print_var = False):
        self.fs = fs
        self.trials_dict = data_dict
        self.n_features = n_features
        self.n_trials_class_1 = data_dict[list(data_dict.keys())[0]].shape[0]
        self.n_trials_class_2 = data_dict[list(data_dict.keys())[1]].shape[0]
        self.print_var = print_var
        
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
        # Training of the classifier
        if(classifier != None): self.trainClassifier(classifier = classifier)
        else: self.trainClassifier() 
        
        
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
        self.mutual_information_vector, self.other_info_matrix = self.changeShapeMutualInformationList()
        
        # Select features to use for classification
        self.classifier_features = self.selectFeatures()
        
    def spatialFilteringW(self, trials, W):
        # Allocate memory for the spatial fitlered trials
        trials_csp = np.zeros(trials.shape)
        
        # Apply spatial fitler
        for i in range(trials.shape[0]): trials_csp[i, :, :] = W.dot(trials[i, :, :])
            
        return trials_csp
    
    
    def computeFeaturesMutualInformation(self):
        """
        Select the first and last n columns of the various features matrix and compute their mutual inforamation.
        The value of n is self.n_features

        Returns
        -------
        mutual_information_list : List of numpy matrix
            List with the mutual information of the various features.

        """
        
        mutual_information_list = []
        
        # Create index for select the first and last m column
        idx = []
        for i in range(self.n_features): idx.append(i)
        for i in reversed(idx): idx.append(-(i + 1))
        
        # Cycle through the different band
        for features_dict in self.features_band_list:
            # Retrieve features for that band
            keys = list(features_dict.keys())
            feat_1 = features_dict[keys[0]][:, idx]
            feat_2 = features_dict[keys[1]][:, idx]
            
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
    
    
    def changeShapeMutualInformationList(self):
        # 1D-Array with all the mutual information value
        mutual_information_vector = np.zeros(9 * 2 * self.n_features)
            
        # Since the CSP features are coupled (First with last etc) in this matrix I save the couple.
        # I will also save the original band and the position in the original band
        other_info_matrix = np.zeros((len(mutual_information_vector), 4))
        
        for i in range(len(self.mutual_information_list)):
            mutual_information = self.mutual_information_list[i]
            
            for j in range(self.n_features * 2):
                # Acual index for the various vector
                actual_idx = i * self.n_features * 2 + j
                
                # Save the various information
                mutual_information_vector[actual_idx] = mutual_information[j]
                other_info_matrix[actual_idx, 0] = i * self.n_features * 2 + ((self.n_features * 2) - (j + 1))
                other_info_matrix[actual_idx, 1] = actual_idx
                other_info_matrix[actual_idx, 2] = i
                other_info_matrix[actual_idx, 3] = j
                
        return mutual_information_vector, other_info_matrix
            
    
    def selectFeatures(self):
        """
        Select n features for classification. n can vary between self.n_features and (2 * self.n_features).
        The features selected are the self.n_features with the highest mutual information. 
        Since the CSP features are coupled if the original couple was not selected we add to the list of features the various couple.

        Returns
        -------
        complete_list_of_features : List of tuple
            List that contatin the band for the filter and the position inside the original band.

        """
        
        # Sort features in order of mutual information
        sorted_MI_features_index = np.flip(np.argsort(self.mutual_information_vector))
        sorted_other_info = self.other_info_matrix[sorted_MI_features_index, :]
        
        complete_list_of_features = []
        selected_features = sorted_other_info[:, 1][0:self.n_features]
        
        for i in range(self.n_features):
            # Current features (NOT USED)(added just for clarity during coding)
            # current_features = sorted_other_info[i, 1]
            
            # Twin/Couple feature of the current features
            current_features_twin = sorted_other_info[i, 0]
            
            if(current_features_twin in selected_features): 
                # If I also select its counterpart I only add the current feaures because the counterpart will be added in future iteration of the cycle
                
                # Save the features as tuple with (original band, original position in the original band)
                features_item = (int(sorted_other_info[i, 2]), int(sorted_other_info[i, 3]))
                
                # Add the element to the features vector
                complete_list_of_features.append(features_item)
            else: 
                # If I not select its counterpart I addo both the current features and it's counterpart
                
                # Select and add the current feature
                features_item = (int(sorted_other_info[i, 2]), int(sorted_other_info[i, 3]))
                complete_list_of_features.append(features_item)
                
                # Select and add the twin/couple feature
                idx = sorted_other_info[:, 1] == current_features_twin
                features_item = (int(sorted_other_info[idx, 2][0]), int(sorted_other_info[idx, 3][0]))
                complete_list_of_features.append(features_item)
                
        return sorted(complete_list_of_features)
                
    
    def extractFeaturesForTraining(self, n_features = 1):
        # Tracking variable of the band
        old_band = -1
        
        # Return matrix
        features_1 = np.zeros((self.n_trials_class_1, len(self.classifier_features)))
        features_2 = np.zeros((self.n_trials_class_2, len(self.classifier_features)))
        
        # Cycle through the different features
        for i in range(len(self.classifier_features)):
            # Retrieve the position of the features
            features_position = self.classifier_features[i]
            
            # Band of the selected feaures
            current_features_band = features_position[0]
            
            # Check if the band is the same of the previous iteration
            if(current_features_band != old_band):
                # In this case the band is not the same of the previous iteration
                
                old_band = current_features_band
                
                # Retrieve the dictionary with the features of the two classes for the current band
                current_band_features_dict = self.features_band_list[current_features_band]
                
                # Retrieve the matrix of features for the two classes
                keys = list(current_band_features_dict.keys())
                tmp_feat_1 = current_band_features_dict[keys[0]]
                tmp_feat_2 = current_band_features_dict[keys[1]]
                
                # Squeeze the features matrix
                tmp_feat_1 = self.squeezeFeatures(tmp_feat_1)
                tmp_feat_2 = self.squeezeFeatures(tmp_feat_2)
                
                # Extract the features
                features_1[:, i] = tmp_feat_1[:, features_position[1]]
                features_2[:, i] = tmp_feat_2[:, features_position[1]]
                
            else: 
                # In this case I'm in the same band of the previous iteration
                
                # Extract the features
                features_1[:, i] = tmp_feat_1[:, features_position[1]]
                features_2[:, i] = tmp_feat_2[:, features_position[1]]
        
                
        return features_1, features_2
    
    def squeezeFeatures(self, features_matrix):
        """
        Given a trial matrix with dimension "n_trials x n_features" this function select the first and last n columns.
        n is the number self.n_features.

        """
        
        # Create index for select the first and last n column
        idx = []
        for i in range(self.n_features): idx.append(i)
        for i in reversed(idx): idx.append(-(i + 1))
        
        # Create the new matrix for the feaures
        squeeze_features = np.zeros((features_matrix.shape[0], self.n_features * 2))
        
        # Select the firs and last n features
        squeeze_features[:, :] = features_matrix[:, idx]
                 
        return squeeze_features
        
    def trainClassifier(self, train_ratio = 0.75, classifier = None):
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
        
        features_1, features_2 = self.extractFeaturesForTraining()
        self.n_features_for_classification = features_1.shape[1]
        if(self.print_var): print("Features used for classification: ", self.n_features_for_classification)
    
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
        
        # Create the label dict
        self.tmp_label_dict = {}
        keys = list(self.features_band_list[0].keys())
        self.tmp_label_dict[1] = keys[0]
        self.tmp_label_dict[2] = keys[1]
        
        # Shuffle the data
        perm = np.random.permutation(len(label))
        label = label[perm]
        data_matrix = data_matrix[perm, :]
        
        # Select the portion of data used during training
        if(train_ratio <= 0 or train_ratio >= 1): train_ratio = 0.75
        index_training = int(data_matrix.shape[0] * train_ratio)
        train_data = data_matrix[0:index_training, :]
        train_label = label[0:index_training]
        test_data = data_matrix[index_training:, :]
        test_label = label[index_training:]
        self.tmp_train = [train_data, train_label]
        self.tmp_test = [test_data, test_label]
        
        # Select classifier
        if(classifier == None): self.classifier = LDA()
        else: self.classifier = classifier
        
        # Train Classifier
        self.classifier.fit(train_data, train_label)
        if(self.print_var): print("Accuracy on TRAIN set: ", self.classifier.score(train_data, train_label))
        
        # Test parameters
        if(self.print_var): print("Accuracy on TEST set: ", self.classifier.score(test_data, test_label), "\n")
        
        # print("total: ", self.classifier.score(train_data, train_label) * self.classifier.score(test_data, test_label))
        
    def evaluateTrial(self, trials_matrix, plot = True):
        """
        Evalaute trial/trials given in input

        Parameters
        ----------
        trials_matrix : Numpy 3D matrix
            Input matrix of trials. The dimension MUST BE "n. trials x n. channels x n.samples".
            Also in case of single trials the input input dimension must be "1 x n. channels x n.samples".
        plot : Boolean, optional
            If set to true will plot the features of the trial. The default is True.

        Returns
        -------
        y : Numpy vector
            Vector with the label of the respective trial. The length of the vector is the number of trials.
            The label are 1 for class 1 and 2 for class 2.
        
        y_prob : Numpy matrix
            Vector with the label of the respective trial. The length of the vector is the number of trials.
            The label are 1 for class 1 and 2 for class 2.

        """
        
        # Compute and extract the features for the training
        features_input = self.extractFeatures(trials_matrix)
           
        # Classify the trials
        y = self.classifier.predict(features_input)
        
        # Evaluate the probabilty
        y_prob = self.classifier.predict_proba(features_input)
        
        return y, y_prob
    
    
    def extractFeatures(self, trials_matrix):
        # Create index for select the first and last m column
        idx = []
        for i in range(self.n_features): idx.append(i)
        for i in reversed(idx): idx.append(-(i + 1))
        
        # List for the features
        features_list = []
        
        # Input for the classifier
        features_input = np.zeros((trials_matrix.shape[0], len(self.classifier_features)))
        
        # Frequency filtering, spatial filtering, features evaluation
        for i in range(len(self.freqs) - 1):              
            # "Create" the band
            band = [self.freqs[i], self.freqs[i+1]]
            
            # Retrieve spatial filter
            W = self.W_list_band[i]
            
            # Frequency and spatial filter
            band_filter_trials_matrix = self.bandFilterTrials(trials_matrix, band[0], band[1])
            spatial_filter_trial = self.spatialFilteringW(band_filter_trials_matrix, W)
            
            # Features evaluation
            features = self.logVarEvaluation(spatial_filter_trial)
            
            features_list.append(features[:, idx])
            
        # Features selection
        for i in range(len(self.classifier_features)):
            # Retrieve feature position
            feature_position = self.classifier_features[i]
            
            # Retrieve feature from the evaluated features
            features_input[:, i] = features_list[feature_position[0]][:, feature_position[1]]
            
        return features_input
    
    def plotFeaturesSeparateTraining(self, width = 0.3, figsize = (15, 30)):
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
            
    def plotFeaturesScatterTraining(self, selected_features = [0, -1], figsize = (15, 10)):
        """
        Plot a mean of the two selected features. 

        Parameters
        ----------
        selected_features : List, optional
            Features to plot. By default the first and the last one are selected. The default is [0, -1].
            It MUST BE a list of length 2.
        figsize : Tuple, optional
            Dimension of the figure. The default is (15, 10).

        """
        # Check the selected_featurest
        if(type(selected_features) != list): selected_features = [0, -1]
        else:
            # Check length
            if(len(selected_features) != 2): selected_features = [0, -1]
            # Check first features
            if(selected_features[0] >= self.n_features_for_classification): selected_features = [0, -1]
            if(selected_features[0] < -self.n_features_for_classification): selected_features = [0, -1]
            # Check second features
            if(selected_features[1] >= self.n_features_for_classification): selected_features = [0, -1]
            if(selected_features[1] < -self.n_features_for_classification): selected_features = [0, -1]
            
        # Plot cretion
        fig, ax = plt.subplots(figsize = figsize)
        
        # Features extraction
        features_1, features_2 = self.extractFeaturesForTraining()
        
        # Plot features
        ax.scatter(features_1[:, selected_features[0]], features_1[:, selected_features[1]], color = 'b')
        ax.scatter(features_2[:, selected_features[0]], features_2[:, selected_features[1]], color = 'r')
        
        if(self.classifier.__class__.__name__ == 'LinearDiscriminantAnalysis' or (self.classifier.__class__.__name__ == 'SVC' and self.classifier.kernel == 'linear')):
            coef = self.classifier.coef_
            bias = self.classifier.intercept_[0]
            
            min_x = min(min(features_1[:, 0]), min(features_2[:, 0]))
            max_x = max(max(features_1[:, 0]), max(features_2[:, 0]))
            x = np.linspace(min_x, max_x)
            
            y1 = - (bias + coef[0, selected_features[0]] * x) / coef[0, selected_features[1]]
            
            ax.plot(x, y1, color = 'k')
        
            
    def plotFeaturesScatter(self, trials_matrix, selected_features = [0, -1], figsize = (15, 10)):
        
        # Check the selected_featurest
        if(type(selected_features) != list): selected_features = [0, -1]
        else:
            # Check length
            if(len(selected_features) != 2): selected_features = [0, -1]
            # Check first features
            if(selected_features[0] >= self.n_features_for_classification): selected_features = [0, -1]
            if(selected_features[0] < -self.n_features_for_classification): selected_features = [0, -1]
            # Check second features
            if(selected_features[1] >= self.n_features_for_classification): selected_features = [0, -1]
            if(selected_features[1] < -self.n_features_for_classification): selected_features = [0, -1]
        
        # Features extraction
        features_input = self.extractFeatures(trials_matrix)
        
        # Plotting
        fig, ax = plt.subplots(figsize = figsize)
        ax.scatter(features_input[:, selected_features[0]], features_input[:, selected_features[1]])
            
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
        
        x1 = np.linspace(1, len(y1), len(y1))
        x2 = x1 + 0.35
        
        fig, ax = plt.subplots(figsize = figsize)
        ax.bar(x1, y1, width = width, color = 'b', align='center')
        ax.bar(x2, y2, width = width, color = 'r', align='center')
        
        

        