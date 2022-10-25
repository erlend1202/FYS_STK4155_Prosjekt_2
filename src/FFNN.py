import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1/(1 + np.exp(-x))

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

class FFNN:

    def __init__(self, X, Y, n_hidden_neurons = 50, n_categories = 10, eta = 0.1, lmbd=0.0, epochs = 10, batch_size = 10):
        self.X = X
        self.Y = Y
        self.eta = eta 
        self.lmbd = lmbd
        self.epochs = epochs 
        self.batch_size = batch_size
        # building our neural network

        self.n_inputs, self.n_features = X.shape
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.iterations = self.n_inputs // self.batch_size

        # we make the weights normally distributed using numpy.random.randn

        # weights and bias in the hidden layer
        self.hidden_weights = np.random.randn(self.n_features, n_hidden_neurons)
        self.hidden_bias = np.zeros(n_hidden_neurons) + 0.01

        # weights and bias in the output layer
        self.output_weights = np.random.randn(n_hidden_neurons, n_categories)
        self.output_bias = np.zeros(n_categories) + 0.01

    def feed_forward(self):
        # weighted sum of inputs to the hidden layer
        z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        # activation in the hidden layer
        self.a_h = sigmoid(z_h)
        
        # weighted sum of inputs to the output layer
        z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        # softmax output
        # axis 0 holds each input and axis 1 the probabilities of each category
        exp_term = np.exp(z_o)

        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def predict(self):
        return np.argmax(self.probabilities, axis=1)

    def backpropagation(self):
        self.feed_forward()
        a_h, probabilities = self.a_h, self.probabilities
        
        # error in the output layer
        error_output = probabilities - self.Y_data
        # error in the hidden layer
        error_hidden = np.matmul(error_output, self.output_weights.T) * a_h * (1 - a_h)
        
        # gradients for the output layer
        self.output_weights_gradient = np.matmul(a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)
        
        # gradient for the hidden layer
        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)


        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                #self.X_data = self.X[chosen_datapoints]
                #self.Y_data = self.Y[chosen_datapoints]

                #For å teste, fikk ikke kode over til å funke enda¨
                self.X_data = self.X 
                self.Y_data = self.Y 

                self.feed_forward()
                self.backpropagation()

if __name__ == "__main__":
    n = 100 
    np.random.seed(4)
    x = np.random.rand(n,1)
    y = 4+3*x + x**2 +np.random.randn(n,1)

    x_exact = np.linspace(0,2,11)
    y_exact = 4+3*x_exact + x_exact**2

    test = FFNN(x,y)
    test.train()