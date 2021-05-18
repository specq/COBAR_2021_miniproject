import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import LSTM
from DataLoader import load_beh_data
from DataSaver import save_trained_model

x_train, y_train, x_validation, y_validation, x_test, y_test = load_beh_data()

################################################################################

# training the model

nb_epochs = 150
learning_rate = 1e-3
max_batch_size = 5000
h_dim = 50
n_layers = 10

# checking if there is a GPU available
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. Device variable will be used later in the code.
if is_cuda:
    device = torch.device("cuda:1")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

x_train, y_train = x_train.to(device), y_train.to(device)
x_validation, y_validation = x_validation.to(device), y_validation.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)


model = LSTM.LSTMNet(input_size=len(x_train[0, 0]), output_size=5, hidden_dim=h_dim, n_layers=n_layers, device=device)
model.to(device)
errors_train, errors_test = LSTM.train(model, x_train, y_train, x_validation, y_validation, nb_epochs, learning_rate, max_batch_size)
save_trained_model(model, 'LSTM', errors_train, errors_test, n_layers, h_dim, learning_rate, nb_epochs)


################################################################################

# Evaluate model predictions

criterion = nn.MSELoss()
nb_batchs = int(x_test.shape[0] // max_batch_size) + 1 * (x_test.shape[0] % max_batch_size != 0)
total_loss = 0
test_preds = []

labels_inv = {0 : 'abdominal_pushing', 1 : 'anterior_grooming', 2 : 'posterior_grooming', 3 : 'walking', 4 : 'resting'}

with torch.no_grad():
    
    for k in range(nb_batchs):
        
        input, target = x_test[k * max_batch_size : (k + 1) * max_batch_size], y_test[k * max_batch_size : (k + 1) * max_batch_size]
        model = model.double()
        pred, _ = model.forward(input)
        loss = criterion(pred, target).double()
        total_loss += loss.item() * input.shape[0]
        test_preds += list(pred)

total_loss /= x_test.shape[0]
print("Loss on test set : ", round(total_loss, 6))

test_preds = np.array([labels_inv[torch.argmax(p).item()] for p in test_preds])
df = pd.read_pickle(beh_data_dir)
df = df[df.index.get_level_values("Trial") == 11]
true_preds = df['Manual'].values[5 : -5]

print("Accuracy on test set : ", round(100 * np.sum(test_preds == true_preds) / test_preds.shape[0], 2), " %")
