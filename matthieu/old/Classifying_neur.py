#%% IMPORT

import pandas as pd
import numpy as np
import math
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

def balance_dataset(x, y):
    unique_labels, nb_sample_per_label = compute_label_distribution(y)
    nb_samples_per_class = max(nb_sample_per_label)
    x_balanced = np.empty([0, np.size(x,1)])
    y_balanced = np.empty(0)
    for label in unique_labels:
        indices = (y == label)
        samples = x[indices]
        labels = y[indices]
        while 2*np.size(samples,0) < nb_samples_per_class:
            samples = np.concatenate([samples, samples], axis=0)
            labels = np.concatenate([labels, labels], axis=0)
        samples = np.concatenate([samples, samples[:(nb_samples_per_class-np.size(samples,0))]], axis=0)
        labels = np.concatenate([labels, labels[:(nb_samples_per_class-np.size(labels,0))]], axis=0)
        x_balanced = np.concatenate([x_balanced, samples], axis=0)
        y_balanced = np.concatenate([y_balanced, labels], axis=0)
    indices_suffled = np.random.permutation(np.size(y_balanced,0))
    x_balanced = x_balanced[indices_suffled]
    y_balanced = y_balanced[indices_suffled]
    return x_balanced, y_balanced
        

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
    
    x_train, y_train = balance_dataset(x_train, y_train)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

#%% LOADING DATA
neur_df = pd.read_pickle("COBAR_behaviour_incl_manual_corrected.pkl")
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

x_train, y_train, x_val, y_val, x_test, y_test = create_data_set(angles_wav, labels, val_ratio, test_ratio)

#%% RANDOM FOREST CLASSIFICATION
print("Start Random Forest")
clf_forest = RandomForestClassifier(max_depth=10, random_state=0)
clf_forest.fit(x_train, y_train)

score_forest = clf_forest.score(x_test, y_test)


#%% SVM
print("Start SVM")
clf_svm = svm.SVC() 
clf_svm.fit(x_train, y_train)
# clf_svm_26 = svm.SVC(C=10, kernel='sigmoid', gamma=1) 
# clf_svm_26.fit(x_train, y_train)
# clf_svm_41 = svm.SVC(C=100, kernel='sigmoid', gamma=0.1) 
# clf_svm_41.fit(x_train, y_train)
#%% SVM score
score_test_svm = clf_svm.score(x_test, y_test)
# score_test_svm_26 = clf_svm_26.score(x_test, y_test)
# score_test_svm_41 = clf_svm_41.score(x_test, y_test)

#%% SVM Grid Search

# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
# grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
# grid.fit(angles_train, labels_train)

#%%
# best_ind = grid.cv_results_['params'][grid.best_index_]
# print(best_ind)
# best_sc = grid.best_score_
# print(best_sc)
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