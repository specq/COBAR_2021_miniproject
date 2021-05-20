import torch
from torch.utils.data import Dataset, DataLoader
import pickle
 

class FlyDataset(Dataset):
    def __init__(self, filename):
        file = open(filename, "rb")
        self.x = pickle.load(file)
        self.y = pickle.load(file)
        self.n_samples = self.x.size(0)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

def loadDataset(batch_size=64):
    train_set = FlyDataset("train_set.pkl")
    val_set = FlyDataset("val_set.pkl")
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader
