# -*- coding: utf-8 -*-
"""
File containing various support function.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
import scipy.signal
import scipy.linalg as la

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%% Function for 100Hz dataset (Dataset IV-1) and data handling
# This function are specific for the dataset IV-1

def loadDatasetD1_100Hz(path, idx, type_dataset, high_sampling_dataset = False):
    if(high_sampling_dataset):  tmp = loadmat(path + idx + '_1000Hz.mat');
    else:  tmp = loadmat(path + idx + '.mat');
   
    data = tmp['cnt'].T
    
    if(type_dataset == 'train'):
        b = tmp['mrk'][0,0]
        cue_position = b[0]
        labels  = b[1]
    else:
        cue_position = labels = None
    
    other_info = tmp['nfo'][0][0]
    sample_rate = other_info[0][0,0]
    channe_name = retrieveChannelName(other_info[2][0])
    class_label = [str(other_info[1][0, 0][0]), str(other_info[1][0, 1][0])]
    n_class = len(class_label)
    n_events = len(cue_position)
    n_channels = np.size(data, 0)
    n_samples = np.size(data, 1)
    
    other_info = {}
    other_info['sample_rate'] = sample_rate
    other_info['channel_name'] = channe_name
    other_info['class_label'] = class_label
    other_info['n_class'] = n_class
    other_info['n_events'] = n_events
    other_info['n_channels'] = n_channels
    other_info['n_samples'] = n_samples

    
    return data, labels, cue_position, other_info


def retrieveChannelName(channel_array):
    channel_list = []
    
    for el in channel_array: channel_list.append(str(el[0]))
    
    return channel_list


def computeTrialD1_100Hz(data, cue_position, labels, fs, class_label = None):
    """
    Transform the 2D data matrix of dimensions channels x samples in various 3D matrix of dimensions trials x channels x samples.
    The number of 3D matrix is equal to the number of class.
    Return everything inside a dicionary with the key label/name of the classes. If no labels are passed a progressive numeration is used.

    Parameters
    ----------
    data : Numpy matrix of dimensions channels x samples.
         Obtained by the loadDataset100Hz() function.
    cue_position : Numpy vector of length 1 x samples.
         Obtained by the loadDataset100Hz() function.
    labels : Numpy vector of length 1 x trials
        Obtained by the loadDataset100Hz() function.
    fs : int/double.
        Sample frequency.
    class_label : string list, optional
        List of string with the name of the class. Each string is the name of 1 class. The default is ['1', '2'].

    Returns
    -------
    trials_dict : dictionair
        Diciotionary with jey the various label of the data.

    """
    
    trials_dict = {}
    
    windows_sample = np.linspace(int(0.5 * fs), int(2.5 * fs) - 1, int(2.5 * fs) - int(0.5 * fs)).astype(int)
    n_windows_sample = len(windows_sample)
    
    n_channels = data.shape[0]
    labels_codes = np.unique(labels)
    
    if(class_label == None): class_label = np.linspace(1, len(labels_codes), len(labels_codes))
    
    for label, label_code in zip(class_label, labels_codes):
        # print(label)
        
        # Vector with the initial samples of the various trials related to that class
        class_event_sample_position = cue_position[labels == label_code]
        
        # Create the 3D matrix to contain all the trials of that class. The structure is n_trials x channel x n_samples
        trials_dict[label] = np.zeros((len(class_event_sample_position), n_channels, n_windows_sample))
        
        for i in range(len(class_event_sample_position)):
            event_start = class_event_sample_position[i]
            trials_dict[label][i, :, :] = data[:, windows_sample + event_start]
                
    return trials_dict

#%%

def loadDatasetD2(path, idx):
    """
    Function to load the dataset 2 of the BCI competition.
    N.B. This dataset is a costum dataset crated from the original gdf file using the MATLAB script 'dataset_transform.m'

    Parameters
    ----------
    path : string
        Path to the folder.
    idx : int.
        Index of the file.

    Returns
    -------
    data : Numpy 2D matrix
        Numpy matrix with the data. Dimensions are "samples x channel".
    event_matrix : Numpy matrix
        Matrix of dimension 3 x number of event. The first row is the position of the event, the second the type of the event and the third its duration

    """
    path_data = path + '/' + str(idx) + '_data.mat' 
    path_event = path + '/' + str(idx) + '_label.mat'
    
    data = loadmat(path_data)['data']
    event_matrix = loadmat(path_event)['event_matrix']
    
    return data, event_matrix
    
    
def computeTrialD2(data, event_matrix, fs, windows_length = 4, remove_corrupt = False):
    """
    Convert the data matrix obtained by loadDatasetD2() into a trials 3D matrix

    Parameters
    ----------
    data : Numpy 2D matrix
        Input data obtained by loadDatasetD2().
    event_matrix : Numpy 2D matrix
        event_matrix obtained by loadDatasetD2().
    fs: int
        frequency sampling
    windows_length: double
        Length of the trials windows in seconds. Defualt is 4.

    Returns
    -------
    trials : Numpy 3D matrix
        Matrix with dimensions "n. trials x channel x n. samples per trial".
    labels : Numpy vector
        Vector with a label for each trials. For more information read http://www.bbci.de/competition/iv/desc_2a.pdf

    """
    event_position = event_matrix[:, 0]
    event_type = event_matrix[:, 1]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Remove corrupted trials
    if(remove_corrupt):
        event_corrupt_mask_1 = event_type == 1023
        event_corrupt_mask_2 = event_type == 1023
        for i in range(len(event_corrupt_mask_2)):
            if(event_corrupt_mask_2[i] == True): 
                # Start of the trial
                event_corrupt_mask_1[i - 1] = True
                # Type of the trial
                event_corrupt_mask_1[i + 1] = True
                
        
        event_position = event_position[np.logical_not(event_corrupt_mask_1)]
        event_type = event_type[np.logical_not(event_corrupt_mask_1)]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Since trials have different length I crop them all to the minimum length
    
    # Retrieve event start
    event_start = event_position[event_type == 768]
    
    # Evaluate the samples for the trial window
    start_second = 2
    end_second = 6
    windows_sample = np.linspace(int(start_second * fs), int(end_second * fs) - 1, int(end_second * fs) - int(start_second * fs)).astype(int)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create the trials matrix
    trials = np.zeros((len(event_start), data.shape[1], len(windows_sample)))
    data = data.T
    
    for i in range(trials.shape[0]):
        trials[i, :, :] = data[:, event_start[i] + windows_sample]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create the label vector
    labels = event_type[event_type != 768]
    labels = labels[labels != 32766]
    labels = labels[labels != 1023]
    
    new_labels = np.zeros(labels.shape)
    labels_name = {}
    labels_name[769] = 1
    labels_name[770] = 2
    labels_name[771] = 3
    labels_name[772] = 4
    labels_name[783] = -1
    for i in range(len(labels)):
        new_labels[i] = labels_name[labels[i]]
    labels = new_labels
    
    return trials, labels


def createTrialsDictD2(trials, labels, label_name = None):
    """
    Converts the trials matrix and the labels vector in a dict.

    Parameters
    ----------
    trials : Numpy 2D matrix
        trials matrix obtained by computeTrialD2().
    labels : Numpy vector
        vector trials obtained by computeTrialD2().
    label_name : dicitonary, optional
        If passed must be a dictionart where the keys are the value 769, 770, 771, 772. For each key you must insert the corresponding label.
        See the table 2 at http://www.bbci.de/competition/iv/desc_2a.pdf for  more information.
        The default is None.

    Returns
    -------
    trials_dict : TYPE
        DESCRIPTION.

    """
    trials_dict = {}
    keys = np.unique(labels)
    
    for key in keys:
        if(label_name != None): trials_dict[label_name[key]] = trials[labels == key, :, :]
        else: trials_dict[key] = trials[labels == key, :, :] 
    
    return trials_dict


def loadTrueLabel(path):
    # Load the labels and create a copy
    labels = np.squeeze(loadmat(path)['classlabel'])
    labels_copy = np.copy(labels)
    
    # Invert values (this step is done because the labels in the classifier and in the saved data are inverted)
    # I.e. Label 1 in saved data corresponding to label 4 in the classfier etc
    labels[labels_copy == 1] = 4
    labels[labels_copy == 2] = 3
    labels[labels_copy == 3] = 2
    labels[labels_copy == 4] = 1
    
    return labels
      
#%% Other

def cleanWorkspaec():
    try:
        from IPython import get_ipython
        # get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass
