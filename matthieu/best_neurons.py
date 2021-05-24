#%% IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy

#%% Functions

def compute_delta_F_over_F_neuron(F):
    # Sort the intensities
    F_sorted = np.sort(F,0)
    
    # Calculate the baseline intensity
    q10_index = int(0.1*len(F_sorted))
    F0 = np.mean(F_sorted[:q10_index], axis=0)
    
    # Calculate flurescence changes and filter
    delta_F_over_F = (F-F0)/F0*100
    
    return  delta_F_over_F #gaussian_filter1d(delta_F_over_F, 2) #

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
    
    for i in range(12):
        # Trial indices
        trial_indices = beh_df.index.get_level_values("Trial") == i
        
        # Get the 2-photon indices
        twop_index = beh_df[trial_indices]["twop_index"].to_numpy()
        
        # Get the behavioural labels
        beh_labels = beh_df[trial_indices]["Manual"].to_numpy()
        
        # Down-sample the behavioural labels
        labels_red = reduce_during_2p_frame(twop_index, beh_labels, function=reduce_behaviour)
    
        labels_red_tot = np.concatenate((labels_red_tot,labels_red))
    
    return labels_red_tot


def select_best_neurons(data, labels):
    
    df_ov_F = []

    # Select one of the two lines to choose between F and dF/F
    for label in labels_name:
        df_ov_F.append(data[labels == label][:])
    
    F_beh_mean = np.empty([0,123])
    F_beh_std = np.empty([0,123])
    neur_indexes_sorted_desc = np.empty([0,123])
    delta_min_tot = np.empty([0,123])
    best_neurons = np.empty([0,123])
    
    for i in range(labels_name.size):
        F_beh_mean = np.append(F_beh_mean,[np.mean(df_ov_F[i],0)], axis=0)
        F_beh_std = np.append(F_beh_std,[np.std(df_ov_F[i],0)], axis=0)
        neur_indexes_sorted_desc = np.append(neur_indexes_sorted_desc, \
                                             [np.argsort(F_beh_mean[i,:])[::-1]],axis=0)
        
        
    for i in range(labels_name.size):
        
        delta_min_neurons = np.array([])
         
        for n in range(np.size(data,1)):
            #initialize min at a high value 
            delta_min = 1000*max(F_beh_mean[0])
            
            #compare the activity of one neuron for all behaviours
            for j in range(labels_name.size):
                delta = (F_beh_mean[i,n]-1*F_beh_std[i,n])-(F_beh_mean[j,n]+1*F_beh_std[j,n])
                if(i!=j and delta < delta_min):
                    delta_min = delta
            delta_min_neurons = np.append(delta_min_neurons,delta_min)
        delta_min_tot = np.append(delta_min_tot,[np.sort(delta_min_neurons)[::-1]],axis=0)
        best_neurons = np.append(best_neurons,[np.argsort(delta_min_neurons)[::-1]],axis=0)
                    
    
    return best_neurons, delta_min_tot, F_beh_mean, F_beh_std
    

#%% Load data and downsample
beh_df = pd.read_pickle("COBAR_behaviour_incl_manual_corrected.pkl")
neural_df = pd.read_pickle("COBAR_neural.pkl")

time = neural_df.filter(regex = "t").to_numpy()

F = np.empty([0,123])
dF_over_F = np.empty([0,123])
for trial in range(12):
    neural_trial = neural_df[neural_df.index.get_level_values("Trial") == trial]
    F_trial = neural_trial.filter(regex = "neuron").to_numpy()
    dF_over_F_trial = compute_delta_F_over_F_neuron(F_trial)
    F = np.concatenate((F,F_trial))
    dF_over_F = np.concatenate((dF_over_F, dF_over_F_trial))
old_df = compute_delta_F_over_F_neuron(F)

arf = old_df - dF_over_F
labels = down_sampling(beh_df)
labels_name = np.unique(labels)

#%% Select best neurons

best_neurons, delta_min, mean, std = select_best_neurons(dF_over_F, labels)

#%%
prediction = np.empty([np.size(dF_over_F,0),labels_name.size+2],dtype = 'O')
prediction[:,labels_name.size+1] = labels

best_neur = best_neurons[:,0].astype(int)

for i in range(np.size(dF_over_F,0)):
    neurs = dF_over_F[i,:]
    maximum = 0
    max_index = -1
    for j in range(labels_name.size):

        threshold = mean[j,best_neur[j]]-2*std[j,best_neur[j]]
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
accuracy = np.mean(prediction[:,5] == prediction[:,6])
print(accuracy)
#%%
mean_sorted = np.sort(mean)[:,::-1]
mean_ind_sorted = np.argsort(mean)[:,::-1]

#%%
behaviour = np.empty([np.size(labels,0),np.size(labels_name,0)])
i = 0
for label in labels_name:
    behaviour[:,i] = labels == label
    i += 1

#%%
# x = np.array(['abdominal_pushing', 'anterior_grooming',  'posterior_grooming',  'walking', 'resting' ])
# y = mean_dF_ne[:,81]
# e = std_dF_ne[:,81]

# plt.errorbar(x, y, e, linestyle='None', marker='.')

# plt.show()
#%%
start= 3000
end = 5000
beh = 0
#for i in range(np.size(labels_name,0)):
plt.plot(range(start,end), scipy.stats.zscore(dF_over_F[start:end,best_neur[beh]]))
plt.plot(range(start,end), behaviour[start:end,beh]*1)
    
plt.show()
#%%
start= 8000
end = 10000
beh = 1
#for i in range(np.size(labels_name,0)):
plt.plot(range(start,end), scipy.stats.zscore(dF_over_F[start:end,best_neur[beh]]))
plt.plot(range(start,end), behaviour[start:end,beh]*1)
    
plt.show()
#%%
start= 17500
end = 20000
beh = 2
#for i in range(np.size(labels_name,0)):
plt.plot(range(start,end), scipy.stats.zscore(dF_over_F[start:end,best_neur[beh]]))
plt.plot(range(start,end), behaviour[start:end,beh]*1)
    
plt.show()
#%%
start= 14000
end = 16200
beh = 3
#for i in range(np.size(labels_name,0)):
plt.plot(range(start,end), scipy.stats.zscore(dF_over_F[start:end,best_neur[beh]]))
plt.plot(range(start,end), behaviour[start:end,beh]*1)
    
plt.show()
#%%
start= 2000
end = 3750
beh = 4
#for i in range(np.size(labels_name,0)):
plt.plot(range(start,end), scipy.stats.zscore(dF_over_F[start:end,best_neur[beh]]))
plt.plot(range(start,end), behaviour[start:end,beh]*1)
    
plt.show()
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