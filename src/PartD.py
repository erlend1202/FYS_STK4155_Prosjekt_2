import numpy as np
from FFNN import *
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from to_categorical import *
from grid_search import * 
#For eta and lambda
#Best seems to be eta=1, lmbda = 0.0001

if __name__ == "__main__": 
    # Representing the hidden layers in the neural network
    layers = [10]

    # Loading breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    Y = data.target


    # Splitting data in train / test
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size)
    Y_train = to_categorical(Y_train)

    # Creating plots
    grid_search_hyperparameters_NN_classification(X_train, X_test, Y_train, Y_test, "Prediction accuracy (sigmoid)", func = sigmoid, verbose = True)
    grid_search_layers(X_train, X_test, Y_train, Y_test, "Prediction accuracy different layers (sigmoid)", func = sigmoid, verbose = True)