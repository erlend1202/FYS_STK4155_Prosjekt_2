import numpy as np
import matplotlib.pyplot as plt 

def MSE(y,y_tilde):
    sum = 0
    n = len(y)
    for i in range(n):
        sum += (y[i] - y_tilde[i])**2
    return sum/n

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return  z / (1 + z)

def designMatrix(x, polygrade):
    n = len(x) 
    X = np.ones((n,polygrade+1))      
    for i in range(1,polygrade+1):
        X[:,i] = (x**i).ravel()
    return X

def relu(z):
    a = np.maximum(0,z)
    return a

def delta_relu(z):
    return np.where(z > 0, 1, 0)

def learning_schedule(t):
    return t0/(t+t1)

t0, t1 = 5, 50

n = 1000
np.random.seed(4)
x = np.random.rand(n,1)
#x = np.linspace(0,1,n)
#x = x.reshape(n,1)
y = 4+3*x + 5*x**2 + np.random.randn(n,1)

x_exact = np.linspace(0,1,n)
y_exact = 4+3*x_exact + 5*x_exact**2

X = designMatrix(x,2)
Y = y
eta = 3

n_inputs, n_features = X.shape
n_hidden = 5 
n_output = 1


w1 = np.random.rand(n_features, 3)
w2 = np.random.rand(3, 8)
w3 = np.random.rand(8, 1)

b1 = np.zeros(3) + 0.001
b2 = np.zeros(8) + 0.001
b3 = np.zeros(1) + 0.001


for i in range(1000):
    A = relu(X@w1 + b1)
    B = relu(A@w2 + b2)
    Y_pred = B@w3 + b3

    #Backwards phase
    #dY_pred = 1/n * MSE(Y,Y_pred)
    dY_pred = 2/n * np.sum(Y_pred - Y)
    dw3 = 1/n * dY_pred * w3
    #print(w3.shape, dw3.shape)
    db3 = 1/n * np.sum(dY_pred)
    dB = dY_pred * w3.T
    dw2 = 1/n * dB * w2
    #print(w2.shape, dw2.shape)
    db2 = 1/n * np.sum(dB)

    #Surr herfra
    dA = dB * w2 @ np.sum(delta_relu(A@w2 + b2),axis = 0)/n 
    dw1 = 1/n * dA * w1
    db1 = 1/n * np.sum(dA)

    w1 -= eta*dw1
    w2 -= eta*dw2
    w3 -= eta*dw3

    b1 -= eta*db1 
    b2 -= eta*db2 
    b3 -= eta*db3 
    
    eta = learning_schedule(i)


print(MSE(Y,Y_pred))

X_exact = designMatrix(x_exact, 2)

A = relu(X_exact@w1 + b1)
B = relu(A@w2 + b2)
Y_pred = B@w3 + b3

plt.plot(x_exact,Y_pred, label="prediction")
#plt.plot(x_exact,y, label ="noise")
plt.plot(x_exact,y_exact, label ="exact")

plt.legend()
plt.show()