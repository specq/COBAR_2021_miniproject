import torch
import pandas as pd
import numpy as np


labels = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}


def load_beh_data(path):
    
    beh_df = pd.read_pickle(path)
    indexes = [index for index in beh_df.columns if ('angle' in index or 'joint' in index)] + ['t']
    
    nb_trials_train = 10
    nb_trials_validation = 1
    nb_trials_test = 1
    nb_events = 10
    
    x_train, y_train = create_data_set(beh_df, indexes, 0, nb_trials_train, nb_events)
    x_validation, y_validation = create_data_set(beh_df, indexes, nb_trials_train, nb_trials_train + nb_trials_validation, nb_events)
    x_test, y_test = create_data_set(beh_df, indexes, nb_trials_train + nb_trials_validation, nb_trials_train + nb_trials_validation + nb_trials_test, nb_events)
    
    return torch.Tensor(x_train).double(), torch.Tensor(y_train).double(), torch.Tensor(x_validation).double(), torch.Tensor(y_validation).double(), torch.Tensor(x_test).double(), torch.Tensor(y_test).double()


def create_data_set(df, indexes, trial_start, trial_stop, nb_events):
    
    x, y = [], []
    
    for k in range(trial_start, trial_stop):
        
        trial_df = df[df.index.get_level_values("Trial") == k]
        predictions = trial_df['Manual'].values
        trial_df = trial_df[indexes]
        
        for l in range(trial_df.shape[0] - nb_events):
            
            x.append(trial_df.iloc[k : k + nb_events].values)
            
            true_pred = predictions[k : k + nb_events]
            true_pred = [labels[p] for p in true_pred]
            y.append(np.array([true_pred.count(k) / nb_events for k in range(len(labels))]))
            
    print(len(x))
    print(len(y))
    
    return x, y