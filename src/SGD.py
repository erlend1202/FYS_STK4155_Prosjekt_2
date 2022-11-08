# Using Autograd to calculate gradients for OLS
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from mean_square_error import MSE

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

def GD(x,y,Niterations, momentum, plot=True):
    n = len(x)
    #X = np.c_[np.ones((n,1)), x, x**2]
    X = designMatrix(x,2)
    sh = X.shape[1]
    
    XT_X = X.T @ X

    theta = np.random.randn(sh,1)
    eta = 0.1
    # define the gradient
    training_gradient = grad(CostOLS)
    change = 0
    for iter in range(Niterations):
        gradients = training_gradient(theta,y,X) + momentum*change
        #eta = learning_schedule(iter)
        change = eta*gradients
        theta -= change
    print("theta from own gd")
    print(theta)

    #xnew = np.array([[0],[2]])
    xnew = np.linspace(0,2,n)
    #Xnew = np.c_[np.ones((n,1)), xnew, xnew**2]
    Xnew = designMatrix(xnew,2)

    ypredict = Xnew.dot(theta)

    if plot:
        plt.plot(xnew, ypredict, "r-")
        plt.plot(x, y ,'ro')
        plt.axis([0,2.0,0, 15.0])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'Random numbers ')
        plt.show()
    else:
        return xnew,ypredict


def SGD(x,y,Niterations, momentum, M, plot=True):
    n = len(x)
    #X = np.c_[np.ones((n,1)), x, x**2]
    X = designMatrix(x,2)

    XT_X = X.T @ X
    sh = X.shape[1]
    
    theta = np.random.randn(sh,1)
    eta = 0.1
    # define the gradient
    training_gradient = grad(CostOLS)
    change = 0
    m = int(n/M)
    for iter in range(Niterations):
        for i in range(m):
            random_index = np.random.randint(m)*M
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradients = (1.0/M)*training_gradient(theta,yi,xi) + momentum*change
            #eta = learning_schedule(iter*m+i)
            change = eta*gradients
            theta -= change
    print("theta from own gd")
    print(theta)

    #xnew = np.array([[0],[2]])
    xnew = np.linspace(0,2,n)
    Xnew = designMatrix(xnew,2)
    #Xnew = np.c_[np.ones((n,1)), xnew, xnew**2]
    ypredict = Xnew.dot(theta)

    if plot:
        plt.plot(xnew, ypredict, "r-")
        plt.plot(x, y ,'ro')
        plt.axis([0,2.0,0, 15.0])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'Random numbers ')
        plt.show()
    else:
        return xnew,ypredict



#Added AdaGrad
def GD_Tuned(x,y,Niterations, momentum, plot=True):
    n = len(x)
    #X = np.c_[np.ones((n,1)), x, x**2]
    X = designMatrix(x,2)

    sh = X.shape[1]
    XT_X = X.T @ X

    theta = np.random.randn(sh,1)
    eta = 0.1
    # Including AdaGrad parameter to avoid possible division by zero
    delta  = 1e-8
    # define the gradient
    training_gradient = grad(CostOLS)
    change = 0
    for iter in range(Niterations):
        Giter = np.zeros(shape=(sh,sh))
        eta = learning_schedule(iter)

        gradients = training_gradient(theta,y,X) + momentum*change
        # Calculate the outer product of the gradients
        Giter +=gradients @ gradients.T
        # Simpler algorithm with only diagonal elements
        Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Giter)))]

        # compute update
        change = np.multiply(Ginverse,gradients)
        theta -= change
    print("theta from own gd")
    print(theta)

    #xnew = np.array([[0],[2]])
    xnew = np.linspace(0,2,n)
    #Xnew = np.c_[np.ones((n,1)), xnew, xnew**2]
    Xnew = designMatrix(xnew,2)

    ypredict = Xnew.dot(theta)

    if plot:
        plt.plot(xnew, ypredict, "r-")
        plt.plot(x, y ,'ro')
        plt.axis([0,2.0,0, 15.0])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'Random numbers ')
        plt.show()
    else:
        return xnew,ypredict

#Added AdaGrad
def SGD_Tuned(x,y,Niterations, momentum, M, plot=True):
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

            gradients = (1.0/M)*training_gradient(theta,yi,xi) + momentum*change

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
    print(MSE(y,ypredict))
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



#Added AdaGrad
def SGD_Ridge(x,y,Niterations, momentum, M, lmbda, plot=True):
    n = len(x)
    #X = np.c_[np.ones((n,1)), x, x**2]
    X = designMatrix(x,2)

    sh = X.shape[1]

    
    theta = np.random.randn(sh,1)
    eta = 0.1
    # Including AdaGrad parameter to avoid possible division by zero
    delta  = 1e-8
    # define the gradient

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

            #ridge
            sum = np.sum((yi - xi@theta) * xi)
            gradients = -2 * sum + np.dot(2*lmbda, theta)

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
    xnew = np.linspace(0,1,n)
    #Xnew = np.c_[np.ones((n,1)), xnew, xnew**2]
    Xnew = designMatrix(xnew,2)

    ypredict = Xnew.dot(theta)
    #ypredict2 = Xnew.dot(theta_linreg)
    print(MSE(y,ypredict))

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





        
    #testSGD()
    #Best values: Momentum = 0.1, batch size = 5, iterations = 200-1000 seems suitable enough

    def test_tuning(x,y, x_exact, y_exact):
        fig, axs = plt.subplots(2,2)
        functions = [GD, GD_Tuned, SGD, SGD_Tuned]
        count = 0
        for i in range(2):
            for j in range(2):
                function = functions[count]
                count += 1
                if function == SGD or function == SGD_Tuned:
                    xnew, ypred = function(x,y, 200, 0.1, 5, False)
                else:
                    xnew, ypred = function(x,y, 1000, 0.1, False)
                axs[i,j].plot(x,y,'r.')
                axs[i,j].plot(x_exact,y_exact, 'k--', label="y_exact", zorder=100)
                axs[i,j].plot(xnew, ypred, label=f"{function.__name__}")
                axs[i,j].legend()
        plt.savefig("figures/A4_OneTuned.png")
        plt.show()
    #test_tuning(x,y)




#COMMENTS FOR OVERLEAF
"""
-when testing ridge MSE vs OLS MSE we got that OLS had an MSE of 3.5 while Ridge got an MSE of 5.7.
"""
