import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

# Code from https://deeplizard.com/learn/video/0LhiS6yu2qQ
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

# Create confusion matrix      
cm = confusion_matrix(y_test, y_pred)
classes = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}
plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues)

