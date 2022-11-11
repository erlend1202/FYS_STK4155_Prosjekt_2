import numpy as np

def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return  z / (1 + z)

sigmoid = np.vectorize(sigmoid)

def relu(z):
    a = np.maximum(0,z)
    return a

def delta_relu(z):
    return np.where(z > 0, 1, 0)

def leaky_relu(z):
    a = np.maximum(0.01*z, z)
    return a 

def delta_leaky_relu(z):
    return np.where(z > 0, 1, 0.01)