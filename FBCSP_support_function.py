# -*- coding: utf-8 -*-
"""
File containing various support function 

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

def loadDataset100Hz(path, idx, type_dataset):
    tmp = loadmat(path + idx + '.mat');
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


def computeTrial(data, cue_position, labels, fs, class_label = None):
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


      
#%% Other

def cleanWorkspaec():
    try:
        from IPython import get_ipython
        # get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass
