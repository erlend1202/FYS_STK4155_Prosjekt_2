# Using Autograd to calculate gradients for OLS
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

def accuracy_score_numpy(Y_test, Y_pred):
    print(confusion_matrix(Y_test, Y_pred))
    return np.sum(Y_test == Y_pred) / len(Y_test)

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector
    

def learning_schedule(t):
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

    def fit(self, X, y, eta = 0.1, epochs=10, M=5, momentum=0.1):
        """X_train is a Nxm matrix, N data points, m features
        t_train are the targets values for training data"""
        
        (k, n) = X.shape
        self.weights = weights = np.zeros(n)
        
        self.losses = np.zeros(epochs)


        m = int(k/M)
             
        change = 0


        for iter in range(epochs):
            for i in range(m):
                random_index = np.random.randint(m)*M
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                #Changing eta over time
                eta = learning_schedule(iter*m+i)

                gradients = eta / k *  xi.T @ (self.forward(xi) - yi) + momentum*change

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

    

def test_classification():    
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    Y_onehot = to_categorical_numpy(Y)
    #Important to change n_categories to 2 and problem to anything else than regression
    lg = NumpyLogReg()
    lg.fit(X,Y, eta=0.1, epochs=100)
    Y_predict = lg.predict(X)
    print(accuracy_score_numpy(Y, Y_predict))


if __name__ == "__main__":
    n = 100 
    np.random.seed(4)
    x = np.random.rand(n,1)
    y = 4+3*x + x**2 +np.random.randn(n,1)

    x_exact = np.linspace(0,1,n)
    y_exact = 4+3*x_exact + x_exact**2
    #y = 2.0+3*x +4*x*x# +np.random.randn(n,1)
    t0, t1 = 5, 50
    test_classification()

    