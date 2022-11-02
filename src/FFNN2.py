# From: https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter10.html

import numpy as np
from sklearn import neural_network

def MSE(y,y_tilde):
    sum = 0
    n = len(y)
    for i in range(n):
        sum += (y[i] - y_tilde[i])**2
    return sum/n


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

class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, n_hidden_neurons = 50, n_categories = 10, batch_size = 100, eta = 0.1, lmbda = 0.0, epochs = 10):
        self.X = X
        self.Y = Y
        self.n_hidden_neurons = n_hidden_neurons
        self.n_inputs = X.shape[0] # Samples
        self.n_features = X.shape[1]
        self.n_categories = n_categories
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbda = lmbda


        # Creating biases and weights with initial values
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_weights = None
        self.output_bias = None
        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, 1)
        self.output_bias = np.zeros(self.n_categories) + 0.01
    
    def feed_forward(self): 
        self.z_h = np.matmul(self.current_X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)
        #self.a_h = self.z_h
        
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        #print(self.z_o)
        # Error when z_o becomes NaN
        #exp_term = np.exp(self.z_o)
        #self.probabilities = exp_term / np.sum(exp_term, axis = 1, keepdims = True)
        
        self.probabilities = self.z_o

    def feed_forward_out(self, X):
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis = 1, keepdims = True)

        return probabilities
    
    def backpropagation(self):
        error_output = self.probabilities - self.current_Y_data
        #print(error_output.shape, self.output_weights.shape, self.a_h.shape)
        error_hidden = np.matmul(error_output, self.output_weights) * self.a_h * (1 - self.a_h)
        
        print(self.a_h.T.shape, error_output.shape)
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.current_X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbda > 0.0:
            self.output_weights_gradient += self.lmbda * self.output_weights
            self.hidden_weights_gradient += self.lmbda * self.hidden_weights
        
        print(self.output_weights.shape, self.output_weights_gradient.shape)
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient


    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities
    
    def train(self):
        data_indices = np.arange(self.n_inputs)

        #for i in range(self.epochs):
        for i in range(1):
            for j in range(1):
            #for j in range(self.iterations):
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                self.current_X_data = self.X[chosen_datapoints]
                self.current_Y_data = self.Y[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
        y_pred = self.current_X_data.dot(self.probabilities.T)
        print(self.probabilities.shape)
        print(self.current_X_data.shape)

if __name__ == "__main__":
    n = 100
    np.random.seed(4)
    x = np.random.rand(n, 1)
    y = 4 + 3*x + x ** 2 + np.random.randn(n, 1)
    X = designMatrix(x,2)
    nn = FeedForwardNeuralNetwork(X, y, 3, 3, 10)
    nn.train()
    #print(nn.predict(y))