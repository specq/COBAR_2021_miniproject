import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from behavelet import wavelet_transform

###########################################################
#                    PCA neural data
###########################################################

# Load the neural data
neural_data_dir = beh_data_dir = "COBAR_neural.pkl"
neural_df = pd.read_pickle(neural_data_dir)

# Get the number of trials
nb_trials = len(np.unique(neural_df.index.get_level_values("Trial")))

# Get the time 
t = neural_df["t"][neural_df.index.get_level_values("Trial") == 10]

# Create the dataset for trial 0
F = neural_df.filter(regex = "neuron")[neural_df.index.get_level_values("Trial") == 10]

# Convert into a numpy array
F = F.to_numpy()

# Sort the intensities
F_sorted = np.sort(F, axis=0)

# Calculate the baseline intensity
q10_index = int(0.1*F_sorted.shape[0])
F0 = np.mean(F_sorted[:q10_index], axis=0)

# Calculate flurescence changes and filter
deltaF_over_F = (F-F0)/F0*100

# Filter the signal
deltaF_over_F_filtered = signal.medfilt(deltaF_over_F)

#PCA decomposition
PCA_object = PCA(n_components=4)
deltaF_over_F_proj = PCA_object.fit_transform(deltaF_over_F_filtered)
sum(PCA_object.explained_variance_ratio_)

# Plot the first principal components over time
legend_labels = []
plt.figure()
for i in range(deltaF_over_F_proj.shape[1]):
    plt.plot(t, deltaF_over_F_proj[:,i])
    legend_labels.append("Component " + str(i+1))

plt.xlabel("Time [s]")
plt.ylabel("$\Delta F/F$  projected [%]")
plt.grid()
plt.legend(legend_labels)
plt.show()

# Scatter the first 2 components
plt.figure()
plt.scatter(deltaF_over_F_proj[:,0], deltaF_over_F_proj[:,1], marker='.')
plt.xlabel("$1^{st}$ Component [%]")
plt.ylabel("$2^{nd}$ Component [%]")
plt.grid()
plt.show()

####################################################################
#             Dimensionality reduction behaviour
####################################################################

# Load the behavioural data
beh_data_dir = beh_data_dir = "COBAR_behaviour.pkl"
beh_df = pd.read_pickle(beh_data_dir)

# Get the number of trials
nb_trials = len(np.unique(beh_df.index.get_level_values("Trial")))

# Get the time 
t = beh_df["t"][beh_df.index.get_level_values("Trial") == 0]

# Create the dataset for a specific trial
beh_joint_set = beh_df.filter(regex = "joint")[beh_df.index.get_level_values("Trial") == 10].to_numpy()
beh_angle_set = beh_df.filter(regex = "angle")[beh_df.index.get_level_values("Trial") == 10].to_numpy()
beh_set = np.concatenate((beh_joint_set, beh_angle_set), axis = 1)

PCA_object = PCA(n_components=16)
beh_set_pca = PCA_object.fit_transform(beh_set)
print('Explained variance: %.3f' % sum(PCA_object.explained_variance_ratio_))

# Wavelet transform
_,_,beh_set_pca_wlet = wavelet_transform(beh_set_pca, n_freqs=25, fsample=100., fmin=1., fmax=50.)

# t-SNE
TSNE_object = TSNE(n_components=2)
beh_embedded = TSNE_object.fit_transform(beh_set_pca_wlet)

# Scatter the embedding
plt.figure()
plt.scatter(beh_embedded[:,0], beh_embedded[:,1], marker='.')
plt.xlabel("$1^{st}$ Component [%]")
plt.ylabel("$2^{nd}$ Component [%]")
plt.grid()
plt.show()

# Plot the first principal components over time
legend_labels = []
plt.figure()
for i in range(beh_set_pca.shape[1]):
    plt.plot(t, beh_set_pca[:,i])
    legend_labels.append("Component " + str(i+1))

plt.xlabel("Time [s]")
plt.ylabel("Angle projected [rad]")
plt.grid()
plt.legend(legend_labels)
plt.show()