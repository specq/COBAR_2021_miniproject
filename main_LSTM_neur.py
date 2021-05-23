import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import LSTM
from DataLoader_neur import load_beh_data
from DataSaver import save_trained_model


gc.collect()

torch.cuda.empty_cache()

################################################################################

# extracting the data
beh_data_dir = "../data/COBAR_behaviour_incl_manual.pkl"
neur_data_dir = '../data/COBAR_neural.pkl'

x_train, y_train, x_validation, y_validation, x_test, y_test = load_beh_data(beh_data_dir, neur_data_dir)

################################################################################

# training the model

nb_epochs = 300
learning_rate = 1.9e-3
max_batch_size = 5000
h_dim = 50
n_layers = 2
dim_linear = 50

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


model = LSTM.LSTMNet(input_size=len(x_train[0, 0]), output_size=5, hidden_dim=h_dim, n_layers=n_layers, dim_linear=dim_linear, device=device)
model.to(device)
errors_train, errors_test = LSTM.train(model, x_train, y_train, x_validation, y_validation, nb_epochs, learning_rate, max_batch_size)
save_trained_model(model, 'LSTM_neur', errors_train, errors_test, n_layers, h_dim, learning_rate, nb_epochs)


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

test_preds = np.array([torch.argmax(p).item() for p in test_preds])
true_preds = np.array([torch.argmax(p).item() for p in y_test])

print("Accuracy on test set : ", round(100 * np.sum(test_preds == true_preds) / test_preds.shape[0], 2), " %")

for k in range(5):
    
    indexes = (true_preds == k)
    spec_preds = test_preds[indexes]
    spec_true_preds = true_preds[indexes]
    print("Number of samples for ", labels_inv[k], " in the test set : ", spec_preds.shape[0])
    print("Accuracy for ", labels_inv[k], " in the test set : ", round(100 * np.sum(spec_preds == spec_true_preds) / spec_preds.shape[0], 2), " %")


################################################################################

# plot confusion matrix on the test set

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("../Plots/Confusion_matrix.png")

cm = confusion_matrix(true_preds, test_preds)
classes = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}
plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues)


################################################################################

# Evaluate model predictions per type of behavior

total_loss = 0
preds = []

with torch.no_grad():
    
    data = [(x_train, y_train), (x_validation, y_validation), (x_test, y_test)]
    
    for d in data:
        
        x, y = d
        print("Loop")
        print(x.shape)
        print(y.shape)
        nb_batchs = int(x.shape[0] // max_batch_size) + 1 * (x.shape[0] % max_batch_size != 0)
        print(nb_batchs)
        
        for k in range(nb_batchs):
            
            input, target = x[k * max_batch_size : (k + 1) * max_batch_size], y[k * max_batch_size : (k + 1) * max_batch_size]
            model = model.double()
            pred = model.forward(input)
            loss = criterion(pred, target).double()
            total_loss += loss.item() * input.shape[0]
            preds += list(pred)

total_loss /= len(preds)
print("Total loss : ", round(total_loss, 6))

preds = np.array([labels_inv[torch.argmax(p).item()] for p in preds])
true_preds = np.array([labels_inv[torch.argmax(p).item()] for p in y_train] + [labels_inv[torch.argmax(p).item()] for p in y_validation] + [labels_inv[torch.argmax(p).item()] for p in y_test])

print("Total accuracy : ", round(100 * np.sum(preds == true_preds) / preds.shape[0], 2), " %")

for k in range(5):
    
    indexes = (true_preds == labels_inv[k])
    spec_preds = preds[indexes]
    spec_true_preds = true_preds[indexes]
    print("Number of samples for ", labels_inv[k], " : ", spec_preds.shape[0])
    print("Accuracy for ", labels_inv[k], " : ", round(100 * np.sum(spec_preds == spec_true_preds) / spec_preds.shape[0], 2), " %")
