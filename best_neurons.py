#%% IMPORT
import pandas as pd
import numpy as np

#%% Functions
def compute_delta_F_over_F_frame(F):
    # Sort the intensities
    F_sorted = np.sort(F)
    
    # Calculate the baseline intensity
    q10_index = int(0.1*np.size(F_sorted,1))
    F0 = np.mean(F_sorted[:,:q10_index], axis=1)
    
    # Calculate flurescence changes and filter
    delta_F_over_F = np.transpose((np.transpose(F)-F0)/F0)*100
    
    return delta_F_over_F

def compute_delta_F_over_F_neuron(F):
    # Sort the intensities
    F_sorted = np.sort(F,0)
    
    # Calculate the baseline intensity
    q10_index = int(0.1*len(F_sorted))
    F0 = np.mean(F_sorted[:q10_index], axis=0)
    
    # Calculate flurescence changes and filter
    delta_F_over_F = (F-F0)/F0*100
    
    return delta_F_over_F

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

def reduce_during_2p_frame(twop_index, values, function=reduce_mean):
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
        reduced[i] = function(values[twop_index==index, :])

    return np.squeeze(reduced) if squeeze else reduced

def down_sampling(beh_df):
    

    labels_red_tot = np.array([])
    angles_red_tot = np.empty((0,42))
    
    for i in range(12):
        # Trial indices
        trial_indices = beh_df.index.get_level_values("Trial") == i
        
        # Get the joint angles 
        beh_angles = beh_df.filter(regex = "angle")[trial_indices].to_numpy()
        
        # Get the 2-photon indices
        twop_index = beh_df[trial_indices]["twop_index"].to_numpy()
        
        # Down-sample the joint angles
        beh_angles_red = reduce_during_2p_frame(twop_index, beh_angles, function=reduce_mean)
        
        # Get the behavioural labels
        beh_labels = beh_df[trial_indices]["Manual"].to_numpy()
        
        # Down-sample the behavioural labels
        labels_red = reduce_during_2p_frame(twop_index, beh_labels, function=reduce_behaviour)
    
        angles_red_tot = np.concatenate((angles_red_tot,beh_angles_red))
        labels_red_tot = np.concatenate((labels_red_tot,labels_red))
    
    return angles_red_tot, labels_red_tot


def select_best_neurons(data, labels, ratio):
    
    F_beh = []

    # Select one of the two lines to choose between F and dF/F
    for label in labels_name:
        F_beh.append(data[labels == label][:])
    
    F_beh_mean = []
    neur_indexes_sorted_desc = []
    F_beh_mean_sum = []
    best_neurons = []
    
    for i in range(labels_name.size):
        F_beh_mean.append(np.mean(F_beh[i],0))
        F_beh_mean_sum.append(np.sum(F_beh_mean[i]))
        neur_indexes_sorted_desc.append(np.argsort(F_beh_mean[i])[::-1])
        
        neurons = np.array([])
        
        n = 0
        sum_neur = 0
        
        while sum_neur/F_beh_mean_sum[i] < ratio :
            index_neur = neur_indexes_sorted_desc[i][n]
            sum_neur+= F_beh_mean[i][index_neur]
            neurons = np.append(neurons, index_neur)
            n += 1
        
        best_neurons.append(neurons)
    
    return best_neurons, F_beh_mean, neur_indexes_sorted_desc
    

#%% Load data and downsample
beh_df = pd.read_pickle("COBAR_behaviour_incl_manual.pkl")
neural_df = pd.read_pickle("COBAR_neural.pkl")

F = neural_df.filter(regex = "neuron").to_numpy()
dF_over_F_fr = compute_delta_F_over_F_frame(F)
dF_over_F_ne = compute_delta_F_over_F_neuron(F)

_, labels = down_sampling(beh_df)
labels_name = np.unique(labels)

#%% Select best neurons

ratio = 0.25

best_neurons_F, mean_F, index_F = select_best_neurons(F, labels, ratio)
best_neurons_dF_fr, mean_dF_fr, index_df_fr = select_best_neurons(dF_over_F_fr, labels, ratio)
best_neurons_dF_ne, mean_dF_ne, index_df_ne = select_best_neurons(dF_over_F_ne, labels, ratio)

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%