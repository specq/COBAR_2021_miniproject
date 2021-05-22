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


def select_best_neurons(data, labels):
    
    F_beh = []

    # Select one of the two lines to choose between F and dF/F
    for label in labels_name:
        F_beh.append(data[labels == label][:])
    
    F_beh_mean = np.empty([0,123])
    F_beh_std = np.empty([0,123])
    neur_indexes_sorted_desc = np.empty([0,123])
    delta_min_tot = np.empty([0,123])
    best_neurons = np.empty([0,123])
    
    for i in range(labels_name.size):
        F_beh_mean = np.append(F_beh_mean,[np.mean(F_beh[i],0)], axis=0)
        F_beh_std = np.append(F_beh_std,[np.std(F_beh[i],0)], axis=0)
        neur_indexes_sorted_desc = np.append(neur_indexes_sorted_desc, \
                                             [np.argsort(F_beh_mean[i,:])[::-1]],axis=0)
        
        
    for i in range(labels_name.size):
        
        delta_min_neurons = np.array([])
         
        for n in range(np.size(data,1)):
            #initialize min at a high value 
            delta_min = 1000*max(F_beh_mean[0])
            
            #compare the activity of one neuron for all behaviours
            for j in range(labels_name.size):
                delta = (F_beh_mean[i,n]-2*F_beh_std[i,n])-(F_beh_mean[j,n]+2*F_beh_std[j,n])
                if(i!=j and delta < delta_min):
                    delta_min = delta
            delta_min_neurons = np.append(delta_min_neurons,delta_min)
        delta_min_tot = np.append(delta_min_tot,[np.sort(delta_min_neurons)[::-1]],axis=0)
        best_neurons = np.append(best_neurons,[np.argsort(delta_min_neurons)[::-1]],axis=0)
                    
    
    return best_neurons, delta_min_tot, F_beh_mean, F_beh_std
    

#%% Load data and downsample
beh_df = pd.read_pickle("COBAR_behaviour_incl_manual.pkl")
neural_df = pd.read_pickle("COBAR_neural.pkl")

F = neural_df.filter(regex = "neuron").to_numpy()
dF_over_F_fr = compute_delta_F_over_F_frame(F)
dF_over_F_ne = compute_delta_F_over_F_neuron(F)

_, labels = down_sampling(beh_df)
labels_name = np.unique(labels)

#%% Select best neurons

#best_neurons_F, delta_min_F, mean_F, index_F = select_best_neurons(F, labels)
#best_neurons_dF_fr, delta_min_dF_fr, mean_dF_fr, index_dF_fr = select_best_neurons(dF_over_F_fr, labels)
best_neurons_dF_ne, delta_min_dF_ne, mean_dF_ne, std_dF_ne = select_best_neurons(dF_over_F_ne, labels)

#%%
prediction = np.empty([np.size(dF_over_F_ne,0),labels_name.size+2],dtype = 'O')
prediction[:,labels_name.size+1] = labels

#best_neur = [71, 35, 24, 52, 97]
best_neur = best_neurons_dF_ne[:,0].astype(int)

for i in range(np.size(dF_over_F_ne,0)):
    neurs = dF_over_F_ne[i,:]
    maximum = 0
    max_index = -1
    for j in range(labels_name.size):

        threshold = mean_dF_ne[j,best_neur[j]]-2*std_dF_ne[j,best_neur[j]]
        if neurs[best_neur[j]] > threshold:
            prediction[i,j] = labels_name[j]
            delta = (neurs[best_neur[j]]-threshold)/threshold
            if delta > maximum :
                maximum = delta
                max_index = j
    if max_index != -1 :
        prediction[i,labels_name.size] = labels_name[max_index]
    else :
        prediction[i,labels_name.size] = 'None'  

#%%
no_prediction = sum(prediction[:,5] == 'None')
accuracy =sum(prediction[:,5] == prediction[:,6])/np.size(prediction,0)
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