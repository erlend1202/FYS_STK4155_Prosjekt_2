from sklearn.model_selection import train_test_split
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from LogisticRegression import *


def test_classification():    
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size,
                                                    test_size=test_size)
    #Important to change n_categories to 2 and problem to anything else than regression
    lg = NumpyLogReg()
    lg.fit(X_train,Y_train, eta=0.1, epochs=100)
    Y_predict = lg.predict(X_test)
    #print(accuracy_score_numpy(Y_test, Y_predict))
    acc = accuracy_score_numpy(Y_test, Y_predict, conf=False)
    print(acc)


if __name__ == "__main__":
    n = 100 
    np.random.seed(4)
    x = np.random.rand(n,1)
    y = 4+3*x + x**2 +np.random.randn(n,1)

    x_exact = np.linspace(0,1,n)
    y_exact = 4+3*x_exact + x_exact**2
    #y = 2.0+3*x +4*x*x# +np.random.randn(n,1)
    test_classification()