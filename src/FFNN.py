import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from FrankeFunction import FrankeFunctionNoised, FrankeFunction

def MSE(y,y_tilde):
    sum = 0
    n = len(y)
    for i in range(n):
        sum += (y[i] - y_tilde[i])**2
    return sum/n

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = len(np.unique(integer_vector)) 
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

def designMatrix(x, polygrade):
    n = len(x) 
    X = np.ones((n,polygrade+1))      
    for i in range(1,polygrade+1):
        X[:,i] = (x**i).ravel()
    return X

class FFNN:

    def __init__(self, X, Y, n_hidden_neurons = 50, n_categories = 1, eta = 0.1, lmbd=0.0, epochs = 10, batch_size = 10):
        self.X = X
        self.Y = Y
        self.eta = eta 
        self.lmbd = lmbd
        self.epochs = epochs 
        self.batch_size = batch_size
        # building our neural network

        try:
            self.n_inputs, self.n_features = X.shape
        except:
            self.n_inputs = len(X)
            self.n_features = 1

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
        #self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        self.probabilities = exp_term

    def predict(self):
        return np.argmax(self.probabilities, axis=1)

    def backpropagation(self):
        a_h, probabilities = self.a_h, self.probabilities
        
        # error in the output layer
        error_output = probabilities - self.Y_data

        #test
        #w = self.hidden_weights @ self.output_weights
        #y_pred = X.dot(w)
        #error_output = MSE(y,y_pred)
        #error_output = y - y_pred
        #test


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

        for i in range(1):
        #for i in range(self.epochs):
            #for j in range(self.iterations):
            for j in range(1):
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
    np.random.seed(4)

    
    n = 100 
    np.random.seed(4)
    x = np.random.rand(n,1)
    y = 4+3*x + x**2 +np.random.randn(n,1)
    
    x_exact = np.linspace(0,2,n)
    y_exact = 4+3*x_exact + x_exact**2

    X = designMatrix(x,2)
    test = FFNN(X,y)
    test.train()
    print(test.output_weights.shape)
    print(test.hidden_weights.shape)

    w = test.hidden_weights @ test.output_weights
    print(test.hidden_bias.shape)
    print(test.output_bias.shape)
    print(w + test.output_bias)


    """
    x = np.arange(0, 1, 0.1)
    y = np.arange(0, 1, 0.1)
    x, y = np.meshgrid(x,y)
    X = np.column_stack((x.ravel(), y.ravel()))
    X = X.ravel()
    Y = FrankeFunctionNoised(x,y,0.01).ravel()
    test = FFNN(x,Y)
    test.train()
    """