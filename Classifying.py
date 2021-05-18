# -*- coding: utf-8 -*-
"""
Created on Sun May  9 10:39:50 2021

@author: Matthieu
"""

#%% IMPORT
import pandas as pd
import numpy as np
import math
import pywt
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from behavelet import wavelet_transform
from sklearn.model_selection import GridSearchCV
#%% LOADING DATA
beh_data_dir = "COBAR_behaviour_incl_manual_corrected.pkl"
beh_df = pd.read_pickle(beh_data_dir)
angles = beh_df.filter(regex="angle").values
labels = beh_df["Manual"].values
labels_predict = beh_df["Prediction"].values

behaviours = np.unique(labels)

score_prediction = np.mean(labels == labels_predict)


#%% Down Sampling
# these two functions are just wrappers around the numpy functions to apply them across dimension 0 only
def reduce_mean(values):
    return np.mean(values, axis=0)
def reduce_std(values):
    return np.std(values, axis=0)
def reduce_behaviour(values):
    unique_values, N_per_unique = np.unique(values, return_counts=True)
    i_max = np.argmax(N_per_unique)
    return unique_values[i_max]

def reduce_during_2p_frame(twop_index, values, function=reduce_mean):
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


#%% PCA 
PCA_object = PCA(n_components=15)
angles_proj = PCA_object.fit_transform(angles_red_tot)
#angles_proj = PCA_object.fit_transform(angles)
sum_proj = sum(PCA_object.explained_variance_ratio_)

#%% WAVELET
#freqs, power, angles_wav = wavelet_transform(angles_red_tot, n_freqs=25, fsample=100., fmin=1., fmax=50.)
freqs, power, angles_wav = wavelet_transform(angles_proj, n_freqs=25, fsample=100., fmin=1., fmax=50.)

#%% EXTRACT DATA

labels = labels_red_tot

angles_train = angles_wav[0:math.floor(0.8*angles_wav.shape[0]),:]
labels_train = labels[0:math.floor(0.8*labels.size)]

angles_val = angles_wav[math.floor(0.4*angles_wav.shape[0]):math.floor(0.8*angles_wav.shape[0]),:]
labels_val = labels[math.floor(0.4*labels.size):math.floor(0.8*labels.size)]

angles_test = angles_wav[math.floor(0.8*angles_wav.shape[0]):math.floor(angles_wav.shape[0]),:]
labels_test = labels[math.floor(0.8*labels.size):labels.size]

#%% RANDOM FOREST CLASSIFICATION
clf_forest = RandomForestClassifier(max_depth=10, random_state=0)
clf_forest.fit(angles_train, labels_train)

score_forest = clf_forest.score(angles_test, labels_test)


#%% SVM
clf_svm = svm.SVC()
clf_svm.fit(angles_train, labels_train)
#%%
score_test_svm = clf_svm.score(angles_test, labels_test)

#%% SVM Grid Search

#param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
grid.fit(angles_train, labels_train)

#%%
arf = grid.cv_results_['params'][grid.best_index_]
print(arf)
#%%
score_test_svm = grid.score(angles_test, labels_test)

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
#%%
#%%
#%%