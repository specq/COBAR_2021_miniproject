import torch.nn as nn
import torch
import torch.nn.functional as F
import random as rd
import numpy as np
from torch.autograd import Variable
import time



class LSTMNet(nn.Module):
    
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dim_linear, device):
        super(LSTMNet, self).__init__()
        
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        
        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # stack of fully connected layers
        self.seq = nn.Sequential(nn.Linear(hidden_dim, dim_linear), nn.CELU(), nn.Linear(dim_linear, output_size), nn.CELU())
    
    
    def forward(self, x):
        
        batch_size = x.size(0)
        
        # Initializing hidden state for first input using method defined below
        hidden, cell = self.init_hidden_and_cell(batch_size)
        hidden, cell = hidden.double(), cell.double()
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out, (_, _) = self.lstm(x, (hidden, cell))
        #print('1 : ', out.shape, hidden.shape)
        
        # Reshaping the outputs such that it can be fit into the fully connected layers
        out = out[ : , -1]
        out = self.seq(out)
        
        return out, hidden
    
    
    def init_hidden_and_cell(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden, cell


def train(model, x_train, y_train, x_validation, y_validation, nb_epochs, learning_rate, max_batch_size, criterion=nn.MSELoss()):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    index = np.arange(x_train.shape[0])
    criterion = criterion.double()
    
    errors_train, errors_validation = [], []
    nb_batchs = int(x_train.shape[0] // max_batch_size) + 1 * (x_train.shape[0] % max_batch_size != 0)
    
    t = time.time()
    
    for n in range(1, nb_epochs + 1):
        
        rd.shuffle(index)
        x_train, y_train = x_train[index], y_train[index]
        epoch_loss = 0
        
        for k in range(nb_batchs):
                    
            # Clearing existing gradients from previous epoch
            optimizer.zero_grad()
            input, target = x_train[k * max_batch_size : (k + 1) * max_batch_size], y_train[k * max_batch_size : (k + 1) * max_batch_size]
            #print('0 : ', input.shape, target.shape)
            model = model.double()
            output, _ = model(input)
            #print('2 : ', output.shape, target.shape)
            loss = criterion(output, target).double()
            # Performing backprop
            loss.backward()
            # Updating weights
            optimizer.step()
            # Adding the weighted loss to epoch_loss
            epoch_loss += loss.item() * input.shape[0]
        
        errors_train.append(epoch_loss / x_train.shape[0])
        errors_validation.append(compute_loss(model, x_validation, y_validation, max_batch_size, criterion))
        
        if n % 10 == 0:
            
            print(f"Epoch: {n}/{nb_epochs}.............", end=' ')
            print(f"Train loss: {round(errors_train[-1], 6)}")
            print(f"              ............. Test loss: {round(errors_validation[-1], 6)}")
            t = time.time() - t
            print("Time elapsed for the last 10 epochs : ", round(t / 60, 2), 'min\n')
            t = time.time()
    
    return errors_train, errors_validation


def compute_loss(model, x, y, max_batch_size, criterion=nn.MSELoss()):
    
    nb_batchs = int(x.shape[0] // max_batch_size) + 1 * (x.shape[0] % max_batch_size != 0)        
    total_loss = 0
    
    with torch.no_grad():
        
        for k in range(nb_batchs):
            
            input, target = x[k * max_batch_size : (k + 1) * max_batch_size], y[k * max_batch_size : (k + 1) * max_batch_size]
            #print('0 : ', input.shape, target.shape)
            model = model.double()
            output, _ = model(input)
            #print('2 : ', output.shape, target.shape)
            loss = criterion(output, target).double()
            # Adding the weighted loss to total_loss
            total_loss += loss.item() * input.shape[0]
    
    return total_loss / x.shape[0]