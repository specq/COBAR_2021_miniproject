import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import LSTM
import pickle
from DataSaver import save_trained_model
from data_loader import loadDataset
 
trainloader, valloader = loadDataset(batch_size=64)


loss_dict = {}
hidden_dim_range = [5, 10, 20, 40, 80]
n_layers_range = [1, 2, 4, 8, 16]
for hidden_dim in hidden_dim_range:
    for n_layers in n_layers_range:
        model = LSTM.LSTMNet(input_size=123, output_size=90, hidden_dim=hidden_dim, n_layers=n_layers)
        descriptor = str(hidden_dim)+'_'+str(n_layers)
        loss_dict[descriptor] = LSTM.train(model, trainloader, valloader, epochs=25, lr=1e-1)
        torch.save(model.state_dict(), 'model'+descriptor+'.ckpt')
        print("ok")

