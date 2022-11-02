import numpy as np
from sklearn import neural_network
import matplotlib.pyplot as plt

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
    def __init__(self, X, Y, n_hidden_neurons = 50, n_categories = 1, batch_size = 100, eta = 0.1, lmbda = 0.0, epochs = 10):
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

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01
    
    def feed_forward(self): 
        self.z_h = np.matmul(self.current_X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)
        
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        
        self.probabilities = self.z_o

    def feed_forward_out(self, X):
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        return z_o
    
    def backpropagation(self):
        error_output = self.probabilities - self.current_Y_data
        #print(error_output.shape, self.output_weights.shape, self.a_h.shape)
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)
        
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.current_X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbda > 0.0:
            self.output_weights_gradient += self.lmbda * self.output_weights
            self.hidden_weights_gradient += self.lmbda * self.hidden_weights
        
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

        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                self.current_X_data = self.X[chosen_datapoints]
                self.current_Y_data = self.Y[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
        #y_pred = self.current_X_data.dot(self.probabilities.T)
        #y_pred = self.feed_forward_out(X)
        #print(y_pred)
"""
if __name__ == "__main__":
    n = 100
    dims = 1
    np.random.seed(4)
    x = np.random.rand(n, 1)
    y = 4 + 3*x + x ** 2 + np.random.randn(n, 1)
    X = designMatrix(x,dims)
    nn = FeedForwardNeuralNetwork(X, y, 3, 1, 10, epochs=1000)
    nn.train()

    x_exact = np.linspace(0,1,n)
    x_exact = x_exact.reshape(n,1)
    y_exact = 4 + 3*x_exact + x_exact**2
    X_exact = designMatrix(x_exact, dims)
    y_pred = nn.feed_forward_out(X_exact)
    print(y_pred)
    plt.plot(x_exact, y_pred)
    plt.plot(x_exact,y_exact)
    plt.show()
"""
if __name__ == "__main__":
    n = 1000
    np.random.seed(40)
    x = np.linspace(0, 1, n)
    x = x.reshape(n, 1)
    X = designMatrix(x,2)

    y_exact = 4 + 3*x + x ** 2 
    y = y_exact + np.random.randn(n,1)*0.1

    nn = FeedForwardNeuralNetwork(X, y, 3, 1, 10, epochs=100)
    nn.train()
    #plt.plot(x, y, label="noise")
    plt.plot(x, y_exact, label="exact")
    plt.plot(x, nn.predict_probabilities(X), label="predict")
    plt.legend()
    plt.show()