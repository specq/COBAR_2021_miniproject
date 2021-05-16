import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

beh_data_dir = "COBAR_behaviour_incl_manual.pkl"
beh_df = pd.read_pickle(beh_data_dir)

neural_data_dir = "COBAR_neural.pkl"
neural_df = pd.read_pickle(neural_data_dir)

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

# Trial indices
trial_indices = beh_df.index.get_level_values("Trial") == 0

# Get the joint angles 
beh_angles = beh_df.filter(regex = "angle")[trial_indices].to_numpy()

# Get the 2-photon indices
twop_index = beh_df[trial_indices]["twop_index"].to_numpy()

# Down-sample the joint angles
beh_angles_red = reduce_during_2p_frame(twop_index, beh_angles, function=reduce_mean)

# Get the behavioural labels
labels = beh_df[trial_indices]["Manual"].to_numpy()

# Down-sample the behavioural labels
labels_red = reduce_during_2p_frame(twop_index, labels, function=reduce_behaviour)

trial_indices = neural_df.index.get_level_values("Trial") == 0

# Get the neural data 
neural_data = neural_df.filter(regex = "neuron")[trial_indices].to_numpy()

print(beh_angles_red.shape)
print(labels_red.shape)
print(neural_data.shape)


