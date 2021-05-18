import torch
import pandas as pd
import numpy as np
from down_sampling import down_sampling
import pickle


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

def load_beh_data():
    beh_df = pd.read_pickle("COBAR_behaviour_incl_manual.pkl")
    neural_df = pd.read_pickle("COBAR_neural.pkl")
    
    nb_trials_train = 8
    nb_trials_val = 2
    nb_trials_test = 2
    nb_events = 10
    
    x_train, y_train = create_data_set(beh_df, neural_df, 0, nb_trials_train, nb_events, "train_set")
    breakpoint()
    x_val, y_val = create_data_set(beh_df, neural_df, nb_trials_train, nb_trials_train+nb_trials_val, nb_events, "train_set")
    x_test, y_test = create_data_set(beh_df, neural_df, nb_trials_train+nb_trials_val, 
                                     nb_trials_train+nb_trials_val+nb_trials_test, nb_events)
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def create_data_set(beh_df, neural_df, trial_start, trial_stop, nb_events, filename):
    """

    Parameters
    ----------
    beh_df : bare behavioural dataset
    neural_df : bare neural dataset
    trial_start : index of the first trial in the dataset
    trial_stop : index of the last trial in the dataset
    nb_events : length of the input sequence for the RNN

    Returns
    -------
    x : neural sequence (size: (nb_samples-nb_events) x nb_events x nb_neurons)
    y : joint position/angle sequence (size: (nb_samples-nb_events) x nb_events x nb_joint_position

    """

    nb_neurons = neural_df.filter(regex = "neuron").to_numpy().shape[1]
    nb_joint_pos = beh_df.filter(regex = "joint").to_numpy().shape[1]
    nb_samples = 4040
    x = np.empty([(trial_stop-trial_start), (nb_samples-nb_events), nb_events, nb_neurons])
    y = np.empty([(trial_stop-trial_start), (nb_samples-nb_events), nb_events, nb_joint_pos])

    for k in range(trial_start, trial_stop):
        print(x.shape)
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
        beh_joints_red, beh_labels_red = down_sampling(beh_joints, beh_labels, twop_index)
        print_every = 100
        for i in range(nb_samples - nb_events):
            x[k, i] = delta_F_over_F[i:i+nb_events]
            y[k, i] = beh_joints_red[i:i+nb_events]
            #if i % print_every == 0:
                #print("{:.1f} %".format((k*(nb_samples - nb_events)+i)/(nb_samples - nb_events)/(trial_stop- trial_start)*100))
        #breakpoint()
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    file = open(filename, "wb")
    pickle.dump(x, file)
    pickle.dump(y, file)
    file.close()
    return torch.Tensor(x), torch.Tensor(y)
        
