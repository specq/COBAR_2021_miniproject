#%% IMPORT

import pandas as pd
import numpy as np
import math
import pickle
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from behavelet import wavelet_transform

#%% Functions 

labels_num = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}

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
        

def create_data_set(beh_angles, beh_labels, val_ratio, test_ratio):

    y=np.empty(np.size(beh_labels,0))
    
    x = beh_angles
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
beh_df = pd.read_pickle("COBAR_behaviour_incl_manual_corrected.pkl")
angles = beh_df.filter(regex="angle").values
labels = beh_df["Manual"].values

#%% PCA 
PCA_object = PCA(n_components=17)
angles_proj = PCA_object.fit_transform(angles)
sum_proj = sum(PCA_object.explained_variance_ratio_)

#%% WAVELET
freqs, power, angles_wav = wavelet_transform(angles_proj, n_freqs=25, fsample=100., fmin=1., fmax=50.)

#%% EXTRACT DATA
val_ratio = 0.2
test_ratio = 0.3
train_ratio = 0.5

x_train, y_train, weights, x_val, y_val, x_test, y_test = create_data_set(angles_wav, labels, val_ratio, test_ratio)
#%%
x_test_2 = np.concatenate([x_val,x_test],axis=0)
y_test_2 = np.concatenate([y_val,y_test],axis=0)
#%% RANDOM FOREST CLASSIFICATION
# print("Start Random Forest")
# clf_forest = RandomForestClassifier(max_depth=10, random_state=0)
# clf_forest.fit(x_train, y_train)
# pickle.dump(clf_forest, open('forest_model05.sav', 'wb'))
# print("Random Forest model saved")

#%% SVM
print("Start SVM")
clf_svm = svm.SVC() 
clf_svm.fit(x_train, y_train,sample_weight=weights)
pickle.dump(clf_svm, open('svm_beh_weight.sav', 'wb'))
print("SVM model saved")
#%% SVM score
score_test_svm = clf_svm.score(x_test, y_test)




#%%
forest_loaded = pickle.load(open('forest_model05.sav', 'rb'))
svm_loaded = pickle.load(open('svm_model_05.sav', 'rb'))
#%%
score_svm_loaded = svm_loaded.score(x_test_2, y_test_2)
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