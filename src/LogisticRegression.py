# Using Autograd to calculate gradients for OLS
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import load_breast_cancer


def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector
    
def CostOLS(beta,y,X):
    n = len(y)
    return (1.0/n)*np.sum((y-X @ beta)**2)

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

def forward(X, theta):
    return sigmoid(X @ theta)


def LogReg(x,y,Niterations, momentum, M, plot=True):
    n = len(x)
    #X = np.c_[np.ones((n,1)), x, x**2]
    X = designMatrix(x,2)

    sh = X.shape[1]

    
    theta = np.random.randn(sh,1)
    eta = 0.1
    # Including AdaGrad parameter to avoid possible division by zero
    delta  = 1e-8
    # define the gradient
    training_gradient = grad(CostOLS)
    # Value for parameter rho
    rho = 0.99
    change = 0
    m = int(n/M)
    for iter in range(Niterations):
        Giter = np.zeros(shape=(sh,sh))
        for i in range(m):
            random_index = np.random.randint(m)*M
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            #Changing eta over time
            eta = learning_schedule(iter*m+i)

            #gradients = (1.0/M)*training_gradient(theta,yi,xi) + momentum*change
            gradients = eta / sh *  X.T @ (forward(X, theta) - y) #+ momentum*change
            # Previous value for the outer product of gradients
            Previous = Giter
            # Accumulated gradient
            Giter +=gradients @ gradients.T
            # Scaling with rho the new and the previous results
            Gnew = (rho*Previous+(1-rho)*Giter)
	        # Simpler algorithm with only diagonal elements
            
            Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Gnew)))]

            # compute update
            change = np.multiply(Ginverse,gradients)
            theta -= change
    print("theta from own gd")
    print(theta)
    print(theta.shape)

    #xnew = np.array([[0],[2]])
    xnew = np.linspace(0,2,n)
    #Xnew = np.c_[np.ones((n,1)), xnew, xnew**2]
    Xnew = designMatrix(xnew,2)

    ypredict = Xnew.dot(theta)
    #ypredict2 = Xnew.dot(theta_linreg)

    if plot:
        plt.plot(xnew, ypredict, "r-")
        #plt.plot(xnew, ypredict2, "b-")
        plt.plot(x, y ,'ro')
        plt.axis([0,2.0,0, 15.0])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'Random numbers ')
        plt.show()
    else:
        return xnew,ypredict


class NumpyLogReg:

    def fit(self, X_train, t_train, eta = 0.1, epochs=10, loss_diff = 0, val_set = None):
        """X_train is a Nxm matrix, N data points, m features
        t_train are the targets values for training data"""
        
        (k, m) = X_train.shape
        
        self.weights = weights = np.zeros(m)
        
        self.losses = np.zeros(epochs)
    
            
            
        for e in range(2):
            weights -= eta / k *  X_train.T @ (self.forward(X_train) - t_train) 
            #self.losses[e] = self.loss(t_train, X_train)  
            self.losses[e] = self.loss(t_train, self.forward(X_train))   
        
                

        for e in range(2, epochs):
            self.runs = e+1
            if (abs(self.losses[e-1] - self.losses[e-2]) > loss_diff):
                weights -= eta / k *  X_train.T @ (self.forward(X_train) - t_train)                      
                self.losses[e] = self.loss(t_train, self.forward(X_train))   
                  
            else:
                break
                
    def forward(self, X):
        return sigmoid(X @ self.weights)
    
    def score(self, x):
        z = x
        score = self.forward(z)
        return score
    
    def predict(self, x, bias = True, threshold=0.5):
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
    lg.fit(X,Y)
    Y_predict = lg.predict(X)
    print(accuracy_score_numpy(Y, Y_predict))

if __name__ == "__main__":
    n = 100 
    np.random.seed(4)
    x = np.random.rand(n,1)
    y = 4+3*x + x**2 +np.random.randn(n,1)

    x_exact = np.linspace(0,2,11)
    y_exact = 4+3*x_exact + x_exact**2
    #y = 2.0+3*x +4*x*x# +np.random.randn(n,1)
    t0, t1 = 5, 50
    #LogReg(x,y, 200, 0.1, 5)
    test_classification()

    