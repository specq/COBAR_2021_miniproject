import torch
import pandas as pd
import numpy as np
from down_sampling import down_sampling
import matplotlib.pyplot as plt
import pickle
import LSTM

def compute_delta_F_over_F(F):
    # Sort the intensities
    F_sorted = np.sort(F)
    
    # Calculate the baseline intensity
    q10_index = int(0.1*len(F_sorted))
    F0 = np.mean(F_sorted[:q10_index], axis=0)
    
    # Calculate flurescence changes and filter
    delta_F_over_F = (F-F0)/F0*100
    
    return delta_F_over_F

beh_df = pd.read_pickle("COBAR_behaviour_incl_manual.pkl")
neural_df = pd.read_pickle("COBAR_neural.pkl")

trial = 7
# Get behavioural data
beh_trial = beh_df[beh_df.index.get_level_values("Trial") == trial]
beh_joints = beh_trial.filter(regex = "joint").to_numpy()
beh_labels = beh_trial['Manual'].to_numpy()
twop_index = beh_trial["twop_index"].to_numpy()

# Get neural data
neural_trial = neural_df[neural_df.index.get_level_values("Trial") == trial]
F = torch.Tensor(neural_trial.filter(regex = "neuron").to_numpy())
delta_F_over_F = compute_delta_F_over_F(F)

# Get time
t = neural_trial['t'].to_numpy()

# Down-sample the behavioural data
beh_joints_red, beh_labels_red = down_sampling(beh_joints, beh_labels, twop_index)

model = LSTM.LSTMNet(input_size=123, output_size=90, hidden_dim=10, n_layers=2)
checkpoint = torch.load('model10_2.ckpt')
model.load_state_dict(checkpoint)

output = model.forward(delta_F_over_F.unsqueeze(0)).squeeze().detach().numpy()

joint = 21
# Scatter the first 2 components
plt.figure()
plt.plot(t, output[:,joint])
plt.plot(t, beh_joints_red[:,joint])
plt.xlabel("Time [s]")
plt.ylabel("Joint position [m]")
plt.legend(["Prediction", "Groundtruth"])
plt.grid()
plt.show()