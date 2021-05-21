import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader import loadDataset
from trainer import train
 
trainloader, valloader = loadDataset('train_bal.pkl', 'val_bal.pkl', batch_size=64)

model = nn.Sequential(nn.Linear(123, 1000),
                      nn.ReLU(),
                      nn.Linear(1000, 4000),
                      nn.ReLU(), 
                      nn.Linear(4000, 1000),
                      nn.ReLU(),
                      nn.Linear(1000, 5))

train(model, trainloader, valloader, epochs=200, lr=1e-3)
torch.save(model.state_dict(), 'model.ckpt')
