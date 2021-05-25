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
from behavelet import wavelet_transform
#%% LOADING DATA
beh_data_dir = "COBAR_behaviour_incl_manual.pkl"
beh_df = pd.read_pickle(beh_data_dir)

#%% EXTRACT DATA
beh_df_0 = beh_df[beh_df.index.get_level_values("Trial")==0]
t_beh = beh_df_0["t"]
angles = beh_df_0.filter(regex="angle").values
manual = beh_df_0["Manual"].values
behaviours = np.unique(manual)

#%% PCA 
PCA_object = PCA(n_components=16)
angles_proj = PCA_object.fit_transform(angles)
sum_proj = sum(PCA_object.explained_variance_ratio_)

plt.figure()
for beh in behaviours:
    plt.scatter(angles_proj[manual == beh,0], angles_proj[manual == beh,1], label = beh, marker='.')
plt.legend()
plt.xlabel("$1^{st}$ Component")
plt.ylabel("$2^{nd}$ Component")
plt.title('PCA 2 main components')
plt.grid()
plt.show()


#%% WAVELET
freqs, power, angles_new = wavelet_transform(angles_proj, n_freqs=25, fsample=100., fmin=1., fmax=50.)

#%% T-SNE WITHOUT WAVELET
angles_tsne_pca = TSNE(n_components = 2).fit_transform(angles_proj)

# %% T-SNE
angles_tsne = TSNE(n_components = 2).fit_transform(angles_new)


#%% PLOT T-SNE 
plt.figure()
for beh in behaviours:
    plt.scatter(angles_tsne[manual == beh,0], angles_tsne[manual == beh,1], label = beh, marker='.')
plt.legend()
plt.xlabel("$1^{st}$ Component")
plt.ylabel("$2^{nd}$ Component")
plt.title('t-sne applied on wavelet 0')
plt.grid()
plt.show()

#%% PLOT T-SNE WITHOUT WAVELET
plt.figure()
for beh in behaviours:
    plt.scatter(angles_tsne_pca[manual == beh,0], angles_tsne_pca[manual == beh,1], label = beh, marker='.')
plt.legend()
plt.xlabel("$1^{st}$ Component")
plt.ylabel("$2^{nd}$ Component")
plt.title('t-sne applied on PCA')
plt.grid()
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