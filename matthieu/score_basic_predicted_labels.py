#%% IMPORT
import pandas as pd
import numpy as np
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
classes = {'anterior_grooming' : 0, 'posterior_grooming' : 1, 'walking' :2, 'resting' : 3}
classes_prediction = { 'antennal_grooming' : 0, 'eye_grooming' : 0, 'foreleg_grooming' : 0, 'hindleg_grooming' : 1, 'abdominal_grooming' : 1, 'walking' : 2, 'resting' : 3}
beh_df = pd.read_pickle("COBAR_behaviour_incl_manual_corrected.pkl")
labels_manual = beh_df["Manual"].values
labels_prediction = beh_df["Prediction"].values
labels_prediction = np.delete(labels_prediction,labels_manual == 'abdominal_pushing')
labels_manual = np.delete(labels_manual,labels_manual == 'abdominal_pushing')

score = np.mean(labels_prediction == labels_manual)

y_predicted = [classes_prediction[p] for p in labels_prediction]
y_true = [classes[p] for p in labels_manual]
 
cm_angle = confusion_matrix(y_true, y_predicted)
plot_confusion_matrix(cm_angle, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues)