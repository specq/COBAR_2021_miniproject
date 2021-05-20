import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

file = open("test.pkl", "rb")
x_test = pickle.load(file)
y_test = pickle.load(file)

model = nn.Sequential(nn.Linear(123, 1000),
                    nn.ReLU(),
                    nn.Linear(1000, 4000),
                    nn.ReLU(), 
                    nn.Linear(4000, 1000),
                    nn.ReLU(),
                    nn.Linear(1000, 5))

checkpoint = torch.load('model.ckpt')
model.load_state_dict(checkpoint)

output = model.forward(x_test)
logits = F.softmax(output, dim=1)
y_pred = torch.argmax(logits, dim=1)
acc = (y_pred == y_test).float().mean()

print("{:.1f} %".format(acc*100))


