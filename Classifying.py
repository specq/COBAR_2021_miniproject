# -*- coding: utf-8 -*-
"""
Created on Sun May  9 10:39:50 2021

@author: Matthieu
"""

#%% IMPORT
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from behavelet import wavelet_transform
#%% LOADING DATA
beh_data_dir = "COBAR_behaviour_incl_manual.pkl"
beh_df = pd.read_pickle(beh_data_dir)
angles = beh_df.filter(regex="angle").values
labels = beh_df["Manual"].values
behaviours = np.unique(labels)


#%% WAVELET
freqs, power, angles_wav = wavelet_transform(angles, n_freqs=25, fsample=100., fmin=1., fmax=50.)

#%% EXTRACT DATA
angles_train = angles_wav[0:25200*5,:]
labels_train = labels[0:25200*5]

angles_val = angles_wav[25200*5:25200*10,:]
labels_val = labels[25200*5:25200*10]

angles_test = angles_wav[25200*10:25200*12,:]
labels_test = labels[25200*10:25200*12]

#%% RANDOM FOREST CLASSIFICATION
clf_forest = RandomForestClassifier(max_depth=5, random_state=0)
clf_forest.fit(angles_train, labels_train)

score_forest = clf_forest.score(angles_val, labels_val)


#%% SVM
clf_svm = svm.SVC()
clf_svm.fit(angles_train, labels_train)

score_svm = clf_svm.score(angles_val, labels_val)

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
#%%