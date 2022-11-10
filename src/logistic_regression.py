# Using Autograd to calculate gradients for OLS
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from activation_functions import * 
from to_categorical import *
from accuracy_score import *

def learning_schedule(t,t0,t1):
    return t0/(t+t1)

class NumpyLogReg:

    def fit(self, X, y, eta = 0.1, epochs=10, M=5, lmbda=0.1):
        """X_train is a Nxm matrix, N data points, m features
        t_train are the targets values for training data"""
        (k, n) = X.shape
        self.weights = weights = np.zeros(n)

        m = int(k/M)
             
        change = 0
        
        #for changing learning rate
        self.t0, self.t1 = 5, 50

        for iter in range(epochs):
            for i in range(m):
                random_index = np.random.randint(m)*M
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                #Changing eta over time
                eta = learning_schedule(iter*m+i, self.t0, self.t1)

                gradients = eta / k *  xi.T @ (self.forward(xi) - yi) + lmbda*change 

                change = gradients
                weights -= change

        self.weights = weights
            
    def forward(self, X):
        return sigmoid(X @ self.weights)
    
    def score(self, x):
        z = x
        score = self.forward(z)
        return score
    
    def predict(self, x, threshold=0.5):
        z = x.copy()
        score = self.forward(z)
        return (score>threshold).astype('int')
    
    def loss(self, y, y_hat):
        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))

        return loss