from sklearn.model_selection import train_test_split
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from logistic_regression import *
from matplotlib.colors import LogNorm
from accuracy_score import accuracy_score
from grid_search import * 

def epochs_plot(plot_title, max_epochs, lmda=0.01, eta=0.1, increment = 5, verbose = False):
    num = int(max_epochs/increment)
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size)
    acc_values = np.zeros(num)

    for epoch in range(0,max_epochs,increment):
        #nn = FeedForwardNeuralNetwork(X, y, layers, 1, 10, epochs=epoch, eta=eta, lmbda=lmda)
        #nn.train()
        #mse_values[epoch] = MSE(y_exact, nn.predict_probabilities(X))
        lg = NumpyLogReg()
        lg.fit(X_train,Y_train, eta=eta, epochs=epoch, M = 5, lmbda=lmda)
        Y_predict = lg.predict(X_test)
        acc = accuracy_score(Y_test, Y_predict, conf=False)
        acc_values[int(epoch/increment)] = acc
        
        if verbose:
            print(f"Testing epoch {epoch}, acc is {acc}")
    
    plt.figure()
    x = np.linspace(0,max_epochs,num)
    plt.title(plot_title)
    plt.plot(x, acc_values)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"figures/{plot_title}")


if __name__ == "__main__":
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size)
    #Important to change n_categories to 2 and problem to anything else than regression
    lg = NumpyLogReg()
    lg.fit(X_train,Y_train, eta=0.1, epochs=100)
    Y_predict = lg.predict(X_test)
    #print(accuracy_score_numpy(Y_test, Y_predict))
    acc = accuracy_score(Y_test, Y_predict, conf=True)
    print(acc)
    
    grid_search_hyperparameters_log_reg(X_train, X_test, Y_train, Y_test, "Training_accuracy_Logistic", verbose=True)
    epochs_plot("Epochs_Logistic", 200, verbose=True)