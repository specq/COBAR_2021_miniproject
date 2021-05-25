#%% IMPORT
import numpy as np
import pickle
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
    plt.show()

#%% LOADING DATA

#define the five behaviours encountered in the dataset
classes = {'abdominal_pushing' : 0, 'anterior_grooming' : 1, 'posterior_grooming' : 2, 'walking' : 3, 'resting' : 4}

#Load test data (x) with true labels (y)
angle = open("test_angle.pkl", "rb")
x_test_angle = pickle.load(angle)
y_test_angle = pickle.load(angle)

#Load classifiers
forest_angles = pickle.load(open('forest_model_angle_weight.sav', 'rb'))
svm_angles= pickle.load(open('svm_angle_weight.sav', 'rb'))

#%% Random Forest prediction angles and plot

#Predict the labels with the loaded model
y_pred_angle = forest_angles.predict(x_test_angle)

#Compute the score of the prediction and plot the confusion matrix
score_angle_forest = np.mean(y_pred_angle == y_test_angle)
cm_angle = confusion_matrix(y_test_angle, y_pred_angle)
plot_confusion_matrix(cm_angle, classes, normalize=True, title='Confusion matrix Random Forest - Angles', cmap=plt.cm.Blues)

#%% SVM prediction angles and plot

#Predict the labels with the loaded model
y_pred_angle = svm_angles.predict(x_test_angle)

#Compute the score of the prediction and plot the confusion matrix
score_angle_svm = np.mean(y_pred_angle == y_test_angle)  
cm_angle = confusion_matrix(y_test_angle, y_pred_angle)
plot_confusion_matrix(cm_angle, classes, normalize=True, title='Confusion matrix SVM - Angles', cmap=plt.cm.Blues)