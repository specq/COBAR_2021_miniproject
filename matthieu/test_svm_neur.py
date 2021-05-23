#%% IMPORT
import pandas as pd
import numpy as np
import math
import pickle
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

#%% Functions
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



#%% LOADING DATA
neur = open("test_neur.pkl", "rb") 
x_test_neur = pickle.load(neur)
y_test_neur = pickle.load(neur)
svm_neurons = pickle.load(open('svm_neur_weight.sav', 'rb'))

#%% SVM prediction neurons and plot
y_pred_neur = svm_neurons.predict(x_test_neur)

score_neur = np.mean(y_pred_neur == y_test_neur)

cm_neur = confusion_matrix(y_test_neur, y_pred_neur)
classes = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}
plot_confusion_matrix(cm_neur, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues)

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%