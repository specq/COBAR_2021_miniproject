import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

class LSTMNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTMNet, self).__init__()
        
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        hidden,_ = self.lstm(x)
        output = self.fc(hidden)
        return output


def train(model, trainloader, valloader, epochs=25, lr=1e-3):
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    lr_lambda = lambda e: 0.8**e
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    steps = 0
    running_loss = 0
    print_every = 20
    for e in range(epochs):
        start = time.time()
        for x, y in iter(trainloader):
            steps += 1
            output = model.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                stop = time.time()
                loss_val = 0
                model.eval()
                with torch.no_grad():
                    for i, (x,y) in enumerate(valloader):
                        loss_val += criterion(model.forward(x),y)
                    print("Epoch: {}/{}..".format(e+1, epochs),
                          "lr: {:.4f}..".format(optimizer.state_dict()['param_groups'][0]['lr']),
                          "Loss train: {:.4f}..".format(running_loss/print_every),
                          "Loss val: {:.4f}..".format(loss_val/i),
                          "{:.4f} s/batch".format((stop - start)/print_every)
                         )
                model.train()
                running_loss = 0
                start = time.time()
        scheduler.step()
        
    # Compute validation loss
    loss_val = 0
    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(valloader):
            loss_val += criterion(model.forward(x),y)
    
    return loss_val/i
    