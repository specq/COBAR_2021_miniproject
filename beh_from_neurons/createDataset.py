import torch
import pandas as pd
import numpy as np
from down_sampling import down_sampling
import pickle
import math


labels = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}

def compute_delta_F_over_F(F):
    # Sort the intensities
    F_sorted = np.sort(F)
    
    # Calculate the baseline intensity
    q10_index = int(0.1*len(F_sorted))
    F0 = np.mean(F_sorted[:q10_index], axis=0)
    
    # Calculate flurescence changes and filter
    delta_F_over_F = (F-F0)/F0*100
    
    return delta_F_over_F

def compute_label_distribution(labels):
    L = labels.size(0)
    nb_sample_per_label = []
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        nb_sample_per_label.append((labels == label).long().sum().item())
    return unique_labels, nb_sample_per_label

def balance_dataset(x, y):
    unique_labels, nb_sample_per_label = compute_label_distribution(y)
    nb_samples_per_class = max(nb_sample_per_label)
    x_balanced = torch.empty([0, x.size(1)])
    y_balanced = torch.empty(0)
    for label in unique_labels:
        indices = (y == label)
        samples = x[indices]
        labels = y[indices]
        while 2*samples.size(0) < nb_samples_per_class:
            samples = torch.cat([samples, samples], dim=0)
            labels = torch.cat([labels, labels], dim=0)
        samples = torch.cat([samples, samples[:(nb_samples_per_class-samples.size(0))]], dim=0)
        labels = torch.cat([labels, labels[:(nb_samples_per_class-labels.size(0))]], dim=0)
        x_balanced = torch.cat([x_balanced, samples], dim=0)
        y_balanced = torch.cat([y_balanced, labels], dim=0)
    indices_suffled = torch.randperm(y_balanced.size(0))
    x_balanced = x_balanced[indices_suffled]
    y_balanced = y_balanced[indices_suffled]
    return x_balanced, y_balanced
        

def create_data_set(beh_df, neural_df, val_ratio, test_ratio):
    x = np.empty([12, 4040, 123])
    y = np.empty([12, 4040])
    
    for k in range(12):
        # Get behavioural data
        beh_trial = beh_df[beh_df.index.get_level_values("Trial") == k]
        beh_joints = beh_trial.filter(regex = "joint").to_numpy()
        beh_labels = beh_trial['Manual'].to_numpy()
        twop_index = beh_trial["twop_index"].to_numpy()
        # Get neural data
        neural_trial = neural_df[neural_df.index.get_level_values("Trial") == k]
        F = neural_trial.filter(regex = "neuron").to_numpy()
        delta_F_over_F = compute_delta_F_over_F(F)
        
        # Down-sample the behavioural data
        _, beh_labels_red = down_sampling(beh_joints, beh_labels, twop_index)
        x[k] = delta_F_over_F
        y[k] = [labels[p] for p in beh_labels_red]

    
    x = torch.Tensor(x)
    x = x.view(x.size(0)*x.size(1), x.size(2))
    y = torch.Tensor(y)
    y = y.view(y.size(0)*y.size(1)).long()
    x_bal, y_bal = balance_dataset(x, y)
    return x, y, x_bal, y_bal.long()

        
beh_df = pd.read_pickle("COBAR_behaviour_incl_manual.pkl")
neural_df = pd.read_pickle("COBAR_neural.pkl")

val_ratio = 0.2
test_ratio = 0.2
train_ratio = 0.6

x, y, x_bal, y_bal = create_data_set(beh_df, neural_df, val_ratio, test_ratio)

N = x_bal.size(0)
x_train = x_bal[:math.floor(train_ratio*N)]
x_val = x_bal[math.floor(train_ratio*N):math.floor((train_ratio+val_ratio)*N)]
x_test = x_bal[math.floor((train_ratio+val_ratio)*N):N]
y_train = y_bal[:math.floor(train_ratio*N)]
y_val = y_bal[math.floor(train_ratio*N):math.floor((train_ratio+val_ratio)*N)]
y_test = y_bal[math.floor((train_ratio+val_ratio)*N):N]

file = open("train_bal.pkl", "wb")
pickle.dump(x_train, file)
pickle.dump(y_train, file)
file.close()

file = open("val_bal.pkl", "wb")
pickle.dump(x_val, file)
pickle.dump(y_val, file)
file.close()

file = open("test_bal.pkl", "wb")
pickle.dump(x_test, file)
pickle.dump(y_test, file)
file.close()

file = open("test.pkl", "wb")
pickle.dump(x, file)
pickle.dump(y, file)
file.close()