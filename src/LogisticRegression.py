# Using Autograd to calculate gradients for OLS
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

def accuracy_score_numpy(Y_test, Y_pred, conf=True):
    if conf: print(confusion_matrix(Y_test, Y_pred))
    return np.sum(Y_test == Y_pred) / len(Y_test)

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector
    

def learning_schedule(t,t0,t1):
    return t0/(t+t1)

def designMatrix(x, polygrade):
    n = len(x) 
    X = np.ones((n,polygrade+1))      
    for i in range(1,polygrade+1):
        X[:,i] = (x**i).ravel()
    return X

def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return  z / (1 + z)

sigmoid = np.vectorize(sigmoid)



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

    


    