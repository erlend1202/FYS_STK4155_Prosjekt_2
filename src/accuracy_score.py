import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy_score(Y_test, Y_pred, conf=True):
    if conf: print(confusion_matrix(Y_test, Y_pred))
    return np.sum(Y_test == Y_pred) / len(Y_test)