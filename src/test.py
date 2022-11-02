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

n = 100 
np.random.seed(4)
x = np.random.rand(n,1)
y = 4+3*x + x**2 +np.random.randn(n,1)

x_exact = np.linspace(0,2,n)
y_exact = 4+3*x_exact + x_exact**2

X = designMatrix(x,2)
Y = y
eta = 0.01


n_inputs, n_features = X.shape
n_hidden = 5 
n_output = 1

w_h = np.random.randn(n_features, n_hidden)
b_h = np.zeros(n_hidden) + 0.01 

w_o = np.random.randn(n_hidden, n_output)
b_o = np.zeros(n_output)

#print(X.shape, w_h.shape, w_o.shape)


"""
X = np.reshape(X, (3,100))
Y = np.reshape(Y, (1,100))

w1 = np.random.rand(5, n_features)
w2 = np.random.rand(8, 5)
w3 = np.random.rand(1, 8)

b1 = np.zeros((5,1)) + 0.001
b2 = np.zeros((8,1)) + 0.001
b3 = np.zeros((1,1)) + 0.001

A = relu(w1@X + b1)
B = relu(w2@A + b2)
Y_pred = w3@B + b3
#Backwards phase

dY_pred = 1/n * (Y_pred - Y)
dw3 = 1/n * w3.T @ dY_pred 
db3 = 1/n * np.sum(dY_pred)
dB = w3.T @ dY_pred
dw2 = 1/n * w2.T @ dB
db2 = 1/n * np.sum(dB)
dA = delta_relu(w2 @ A + b2) @ (w2.T @ dB).T 
dw1 = 1/n * dA @ w1
db1 = 1/n * np.sum(dA)

print(w1.shape, dw1.shape)
w1 -= eta*dw1

"""
w1 = np.random.rand(n_features, 5)
w2 = np.random.rand(5, 8)
w3 = np.random.rand(8, 1)

b1 = np.zeros(5) + 0.001
b2 = np.zeros(8) + 0.001
b3 = np.zeros(1) + 0.001


for i in range(100):
    A = relu(X@w1 + b1)
    B = relu(A@w2 + b2)
    Y_pred = B@w3 + b3

    #Backwards phase
    dY_pred = 1/n * MSE(Y,Y_pred)
    dw3 = 1/n * dY_pred * w3
    #print(w3.shape, dw3.shape)
    db3 = 1/n * np.sum(dY_pred)
    dB = dY_pred * w3.T
    dw2 = 1/n * dB * w2
    #print(w2.shape, dw2.shape)
    db2 = 1/n * np.sum(dB)

    #print(w2.shape, dB.shape, A.shape, b2.shape)
    #dA  = (dB @ w2.T).T  @ delta_relu(A@w2 + b2) #usikker om riktig
    #dA  = delta_relu(A@w2 + b2) #usikker om riktig
    #print(A.shape, dA.shape)
    #dw1 = 1/n * w1 @ dA

    #Surr herfra
    dA = dB * w2
    #print(dA.shape, dB.shape, w2.shape)
    #print(dA.shape)
    test1 = np.sum(delta_relu(A@w2 + b2),axis = 0)/n #Skal egt ha dette i dA
    #print("test1")
    #print(test1.shape)
    dA = dA @ test1 
    #print("dA", dA.shape)
    dw1 = 1/n * dA * w1
    #print(dw1.shape, w1.shape)
    db1 = 1/n * np.sum(dA)

    w1 -= eta*dw1
    w2 -= eta*dw2
    w3 -= eta*dw3

    b1 -= eta*db1 
    b2 -= eta*db2 
    b3 -= eta*db3 

    print(MSE(Y,Y_pred))