#%% IMPORT
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.decomposition import PCA
from sklearn import svm
from behavelet import wavelet_transform
from sklearn.ensemble import RandomForestClassifier

#%% Functions 

labels_num = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}

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

def compute_label_distribution(labels):
    nb_sample_per_label = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        nb_sample_per_label.append((labels == label).sum().item())
    return unique_labels, nb_sample_per_label

def compute_weights(y):
    unique_labels, nb_sample_per_label = compute_label_distribution(y)
    max_samples = max(nb_sample_per_label)
    weights = np.empty(np.size(y,0))
    for label in unique_labels:
        weight_label = max_samples/nb_sample_per_label[int(label)]
        indices = (y == label)
        weights[indices] = weight_label
    return weights
        

def create_data_set(data, beh_labels, val_ratio, test_ratio):

    y=np.empty(np.size(beh_labels,0))
    
    x = data
    y[:] = [labels_num[p] for p in beh_labels]
    
    #Shuffle dataset
    indices_suffled = np.random.permutation(np.size(y,0))
    x = x[indices_suffled]
    y = y[indices_suffled]
    
    # Create dataset
    test_split = math.floor(test_ratio*np.size(x,0))
    val_split = math.floor((test_ratio+val_ratio)*np.size(x,0))
    x_test = x[:test_split]
    y_test = y[:test_split]
    x_val = x[test_split:val_split]
    y_val = y[test_split:val_split]
    x_train = x[val_split:]
    y_train = y[val_split:]
    
    weights = compute_weights(y_train)
    
    return x_train, y_train, weights, x_val, y_val, x_test, y_test

#%% LOADING DATA

#Load labels and neural data
beh_df = pd.read_pickle("COBAR_behaviour_incl_manual_corrected.pkl")
neural_df = pd.read_pickle("COBAR_neural.pkl")
#down-sample the labels to match the neural data frequency
labels = down_sampling(beh_df)

#compute the fulorescence change for each trial 
F = np.empty([0,123])
dF_ov_F = np.empty([0,123])
for t in range(12):
    neural_trial = neural_df[neural_df.index.get_level_values("Trial") == t]
    F_trial = neural_trial.filter(regex = "neuron").to_numpy()
    dF_ov_F_trial = compute_delta_F_over_F_neuron(F_trial)
    F = np.concatenate((F,F_trial))
    dF_ov_F = np.concatenate((dF_ov_F, dF_ov_F_trial))
old_df = compute_delta_F_over_F_neuron(F)

arf = old_df - dF_ov_F
#%% PCA 
#compute PCA to reduce the dimensions of the dataset 
PCA_object = PCA(n_components=4)
dF_ov_F_proj = PCA_object.fit_transform(dF_ov_F)
sum_proj = sum(PCA_object.explained_variance_ratio_)

#%% WAVELET
#compute wavelet transform to add dynamic
_, _, dF_ov_F_wav = wavelet_transform(dF_ov_F_proj, n_freqs=25, fsample=100., fmin=1., fmax=50.)

#%% EXTRACT DATA

#split the dataset into test, train, validation
val_ratio = 0.2
test_ratio = 0.3
train_ratio = 0.5

x_train, y_train, weights, x_val, y_val, x_test, y_test = create_data_set(dF_ov_F_wav, labels, val_ratio, test_ratio)
#%% Save test dataset
file = open("test_neur.pkl", "wb")
pickle.dump(x_test, file)
pickle.dump(y_test, file)
file.close()

#%% Random Forest Classifier

#compute the random forest model
print("Start Random Forest")
clf_forest = RandomForestClassifier()
clf_forest.fit(x_train, y_train,sample_weight=weights)
pickle.dump(clf_forest, open('forest_model_neur_weight.sav', 'wb'))
print("Random Forest model saved")


#%% SVM Tuning

#find the best parameters for the SVM
C_svm = [1, 100, 500, 750, 1000, 1500] 
kernel = ['rbf', 'poly']

params = np.empty((0,2))
scores = np.empty(0)


for c in C_svm:
    for k in kernel:
        print('C =',c, 'kernel =',k)
        params = np.concatenate((params, [[c,k]]))
        clf_svm = svm.SVC(C = c, kernel = k) 
        clf_svm.fit(x_train, y_train,sample_weight=weights)
        score_svm = clf_svm.score(x_val, y_val)
        scores = np.concatenate((scores,[score_svm]))
        print('Score = ',score_svm)

scores_index_sorted_desc = np.argsort(scores)[::-1]
best_trial = int(scores_index_sorted_desc[0])
best_score = scores[best_trial]
best_params = params[best_trial]

#%% SVM

#use the best parameters to compute the model
clf_svm = svm.SVC(C = int(best_params[0]), kernel = best_params[1]) 
clf_svm.fit(x_train, y_train,sample_weight=weights)
pickle.dump(clf_svm, open('svm_neur_weight.sav', 'wb'))
print("SVM model saved")
