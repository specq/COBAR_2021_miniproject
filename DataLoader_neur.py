import pandas as pd
import numpy as np
import torch
import random as rd
from scipy import signal


labels = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}


# these two functions are just wrappers around the numpy functions to apply them across dimension 0 only
def reduce_mean(values):
    
    return np.mean(values, axis=0)


def reduce_std(values):
    
    return np.std(values, axis=0)


def reduce_behaviour(values):
    """
    this is just a sketch for how to reduce behavioural classes. 
    It picks whatever behaviour occurs the most.
    Try to make this more stable, for example by handling the case when two behaviours are equally likely.
    You might also want to include a certainty threshold, 
    e.g. 3/4 of the behaviour frames have to be labelled the same way, otherwise it is None and the data is excluded
    """
    unique_values, N_per_unique = np.unique(values, return_counts=True)
    i_max = np.argmax(N_per_unique)
    return unique_values[i_max]


def reduce_during_2p_frame(twop_index, values, function=reduce_behaviour):
    """
    Reduces all values occuring during the acquisition of a
    two-photon imaging frame to a single value using the `function` given by the user.
    Parameters
    ----------
    twop_index : numpy array
        1d array holding frame indices of one trial.
    values : numpy array
        Values upsampled to the frequency of ThorSync,
        i.e. 1D numpy array of the same length as
        `frame_counter` or 2D numpy array of the same length.
    function : function
        Function used to reduce the value,
        e.g. np.mean for 1D variables
    Returns
    -------
    reduced : numpy array
        Numpy array with value for each two-photon imaging frame.
    """
    
    if len(twop_index) != len(values):
        raise ValueError("twop_index and values need to have the same length.")
    if len(values.shape) == 1:
        values = np.expand_dims(values, axis=1)
        squeeze = True
    else:
        squeeze = False
    N_samples, N_variables = values.shape
    
    index_unique = np.unique(twop_index)
    index_unique = np.delete(index_unique, index_unique==-9223372036854775808)
    
    dtype = values.dtype
    if np.issubdtype(dtype, np.number):
        dtype = float
    else:
        dtype = object
    reduced = np.empty((len(index_unique), N_variables), dtype=dtype)

    for i, index in enumerate(index_unique):
        reduced[i] = function(values[twop_index == index, :])

    return np.squeeze(reduced) if squeeze else reduced


def deltaF_over_F(F):
    
    F_sorted = np.sort(F)
        
    # Calculate the baseline intensity
    q10_index = int(0.1 * len(F_sorted))
    F0 = np.mean(F_sorted[ : q10_index])
        
    # Calculate flurescence changes and filter
    deltaF_over_F = (F - F0) / F0
    
    return signal.medfilt(deltaF_over_F, kernel_size=5) # kernel = 11


def load_beh_data(path_beh, path_neur, train_ratio=.1, validation_ratio=.1, nb_events=5):
    
    beh_df = pd.read_pickle(path_beh)
    neur_df = pd.read_pickle(path_neur)
    
    indexes = [index for index in neur_df.columns if 'neuron' in index]
    
    x, y = create_data_set(beh_df, neur_df, indexes, 0, 12, nb_events)
    
    nb_samples_train = int(train_ratio * x.shape[0])
    index_samples_validation = nb_samples_train + int(validation_ratio * x.shape[0])
    
    index = np.arange(x.shape[0])
    rd.shuffle(index)
    x, y = x[index], y[index]
    
    x_train, y_train = x[ : nb_samples_train], y[ : nb_samples_train]
    x_validation, y_validation = x[nb_samples_train : index_samples_validation], y[nb_samples_train : index_samples_validation]
    x_test, y_test = x[index_samples_validation : ], y[index_samples_validation : ]
    
    index = {}
    nb_samples = []
    
    for k in range(5):
        index[k] = y_train[ : , k] >= .5
        nb_samples.append(torch.sum(index[k]))
    
    baseline = np.argmax(nb_samples)
    x_temp, y_temp = {}, {}
    
    for k in range(5):
        
        if k != baseline:
            
            nb_cats = int(nb_samples[baseline] // nb_samples[k]) - 1
            
            if nb_cats != 0:
                x_temp[k], y_temp[k] = x_train[index[k]], y_train[index[k]]
                x_temp[k] = torch.cat([x_temp[k]] * nb_cats)
                y_temp[k] = torch.cat([y_temp[k]] * nb_cats)
    
    for k in x_temp.keys():
        
        x_train = torch.cat((x_train, x_temp[k]), axis=0)
        y_train = torch.cat((y_train, y_temp[k]), axis=0)
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def create_data_set(beh_df, neur_df, indexes, trial_start, trial_stop, nb_events):
    
    x, y = [], []
    
    for k in range(trial_start, trial_stop):
        
        beh_trial = beh_df[beh_df.index.get_level_values("Trial") == k]
        neur_trial = neur_df[neur_df.index.get_level_values("Trial") == k]
        
        predictions = reduce_during_2p_frame(beh_trial['twop_index'].values, beh_trial['Manual'].values)
        neur_trial = neur_trial[indexes]
        
        for ind in indexes:
            neur_trial[ind] = deltaF_over_F(neur_trial[ind])
        
        for l in range(neur_trial.shape[0] - nb_events):
            
            new_x = neur_trial.iloc[l : l + nb_events].values
            x.append(new_x)
            
            true_pred = predictions[l : l + nb_events]
            true_pred = [labels[p] for p in true_pred]
            true_pred = np.array([true_pred.count(i) / nb_events for i in range(len(labels))])
            y.append(true_pred)
    
    return torch.Tensor(x).double(), torch.Tensor(y).double()
