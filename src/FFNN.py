from cmath import isnan, nan
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt 
from mean_square_error import MSE
from activation_functions import * 
from design_matrix import *

def learning_schedule(t, t0, t1):
    return t0/(t+t1)

def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


class FeedForwardNeuralNetwork:
    def __init__(self, X, Y, layers, n_categories = 1, batch_size = 100, eta = 0.1, lmbda = 0.0, epochs = 10, func=sigmoid, problem="regression"):
        self.X = X
        self.Y = Y
        self.func = func
        self.problem = problem
        self.n_inputs = X.shape[0] # Samples
        self.n_features = X.shape[1]
        self.n_categories = n_categories
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbda = lmbda
        self.layers = layers 
        self.num_layers = len(layers)

        #for eta
        self.t0, self.t1 = 5, 50
        # Creating biases and weights with initial values
        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.weights = []
        self.bias = []
        for i in range(self.num_layers):
            if i==0:
                w = np.random.randn(self.n_features, self.layers[i])
                b = np.zeros(self.layers[i]) + 0.01
            else:
                w = np.random.randn(self.layers[i-1], self.layers[i])
                b = np.zeros(self.layers[i]) + 0.01
            self.weights.append(w)
            self.bias.append(b)
        w = np.random.randn(self.layers[i], self.n_categories)
        b = np.zeros(self.n_categories) + 0.01
        self.weights.append(w)
        self.bias.append(b)
             
    
    def feed_forward(self): 
        self.z = []
        self.a = []

        for i in range(self.num_layers):
            if i == 0:
                z = np.matmul(self.current_X_data, self.weights[i]) + self.bias[i]
                a = self.func(z)
            else:
                z = np.matmul(self.a[i-1], self.weights[i]) + self.bias[i]
                a = self.func(z)

            self.z.append(z)
            self.a.append(a)

        z = np.matmul(self.a[-1], self.weights[-1]) + self.bias[-1]
        self.z.append(z)
        self.a.append(z)

        if self.problem == "regression":
            self.probabilities = z

        #Assume its classification otherwise
        else:
            exp_term = np.exp(z)
            self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)


    def feed_forward_out(self, X):
        z_list = []
        a_list = []

        for i in range(self.num_layers):
            if i == 0:
                z = np.matmul(X, self.weights[i]) + self.bias[i]
                a = self.func(z)
            else:
                z = np.matmul(a_list[i-1], self.weights[i]) + self.bias[i]
                a = self.func(z)

            z_list.append(z)
            a_list.append(a)
            
        z = np.matmul(a_list[-1], self.weights[-1]) + self.bias[-1]
        
        if self.problem == "regression":
            return z

        #Asume its classification otherwise
        else:
            exp_term = np.exp(z)
            return exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    def backpropagation(self):
        error1 = self.probabilities - self.current_Y_data
        errors = [error1]
        self.w_grads = []
        self.bias_grads = []

        for i in range(self.num_layers):
            if i == 0:
                if self.func == sigmoid:
                    error = np.matmul(errors[i], self.weights[self.num_layers].T) * self.a[self.num_layers-1] * (1-self.a[self.num_layers-1])
                elif self.func == relu:
                    error = np.matmul(errors[i], self.weights[self.num_layers].T) * delta_relu(self.z[self.num_layers-1])
                elif self.func == leaky_relu:
                    error = np.matmul(errors[i], self.weights[self.num_layers].T) * delta_leaky_relu(self.z[self.num_layers-1])

            else:
                if self.func == sigmoid:
                    error = np.matmul(errors[i], self.weights[self.num_layers-i].T) * self.a[self.num_layers-i-1] * (1-self.a[self.num_layers-i-1])
                elif self.func == relu:
                    error = np.matmul(errors[i], self.weights[self.num_layers-i].T) * delta_relu(self.z[self.num_layers-1-i])
                elif self.func == leaky_relu:
                    error = np.matmul(errors[i], self.weights[self.num_layers-i].T) * delta_leaky_relu(self.z[self.num_layers-1-i])


            dw = np.matmul(self.a[self.num_layers-1-i].T, errors[i])
            db = np.sum(errors[i], axis=0)

            errors.append(error)
            self.w_grads.append(dw)
            self.bias_grads.append(db)

        dw = np.matmul(self.current_X_data.T, errors[self.num_layers])
        db = np.sum(errors[self.num_layers], axis=0)

        self.w_grads.append(dw)
        self.bias_grads.append(db)

        #self.w_grad = np.array(self.w_grads)
        #self.bias_grads = np.array(self.bias_grads)

        if self.lmbda > 0:
            self.w_grads += np.multiply(self.w_grads, self.lmbda, dtype=object)

        for i in range(self.num_layers+1):
            self.weights[i] -= self.eta * self.w_grads[self.num_layers-i]
            self.bias[i] -= self.eta * self.bias_grads[self.num_layers-i]


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
                self.eta = learning_schedule(i*self.iterations + j, self.t0, self.t1)

                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                self.current_X_data = self.X[chosen_datapoints]
                self.current_Y_data = self.Y[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()

def grid_search_hyperparameters(X, y, y_exact, layers, plot_title, func, n_categories = 1, batch_size = 10, epochs = 200, eta_vals = np.logspace(-5, 1, 7), lmd_vals = np.logspace(-5, 1, 7), verbose = False):

    mse_values = np.zeros((len(eta_vals), len(lmd_vals)))

    for i, eta in enumerate(eta_vals):
        for j, lmd in enumerate(lmd_vals):
            nn = FeedForwardNeuralNetwork(X, y, layers, n_categories, batch_size, epochs = epochs, eta = eta, lmbda = lmd, func = func)
            nn.train()
            y_tilde = nn.predict_probabilities(X)
            mse = MSE(y_exact, y_tilde)
            mse_values[i, j] = mse

            if verbose:
                print(f"eta:{eta}, lambda:{lmd} gives mse {mse}")

    def array_elements_to_string(arr):
        new_arr = []

        for element in arr:
            new_arr.append(str(element))
        
        return new_arr
    
    def show_values_in_heatmap(heatmap, axes, text_color = "white"):
        for i in range(len(heatmap)):
            for j in range(len(heatmap[0])):
                axes.text(j, i, np.round(heatmap[i, j], 2), ha="center", va="center", color=text_color)

    labels_x = array_elements_to_string(eta_vals)
    labels_y = array_elements_to_string(lmd_vals)

    plt.figure()
    show_values_in_heatmap(mse_values, plt.gca())
    plt.title(plot_title)
    plt.xticks(np.arange(0, len(eta_vals)), labels_x)
    plt.yticks(np.arange(0, len(lmd_vals)), labels_y)
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")
    plt.imshow(mse_values, norm=LogNorm())
    plt.colorbar()
    plt.savefig(f"figures/{plot_title}")

def epochs_plot(X, y, y_exact, layers, plot_title, max_epochs, lmda, eta, func, verbose = False):
    mse_values = np.zeros(max_epochs)

    for epoch in range(max_epochs):
        nn = FeedForwardNeuralNetwork(X, y, layers, 1, 10, epochs=epoch, eta=eta, lmbda=lmda)
        nn.train()
        mse_values[epoch] = MSE(y_exact, nn.predict_probabilities(X))
        if verbose:
            print(f"Testing epoch {epoch}")
    
    plt.figure()
    plt.title(plot_title)
    plt.plot(mse_values)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig(f"figures/{plot_title}")
"""

if __name__ == "__main__":
    n = 100
    np.random.seed(40)

    x = np.linspace(0, 1, n)
    x = x.reshape(n, 1)
    X = create_design_matrix(x,1)

    y_exact = 4 + 3*x + x ** 2 
    noise = np.random.normal(0, 0.1, n).reshape(n, 1)
    y = y_exact + noise

    layers = [3, 5, 3]
    
    grid_search_hyperparameters(X, y, y_exact, layers, "Training accuracy (Leaky RELU)", leaky_relu)
    epochs_plot(X, y, y_exact, layers, "Epochs (Leaky RELU)", 200, 0.01, 0.01, leaky_relu)

    grid_search_hyperparameters(X, y, y_exact, layers, "Training accuracy (sigmoid)", sigmoid)
    epochs_plot(X, y, y_exact, layers, "Epochs (sigmoid)", 200, 0.01, 0.01, sigmoid)

    grid_search_hyperparameters(X, y, y_exact, layers, "Training accuracy (RELU)", relu)
    epochs_plot(X, y, y_exact, layers, "Epochs (RELU)", 200, 0.01, 0.01, relu)
   
    test_classification()
    nn = FeedForwardNeuralNetwork(X, y, layers, 1, 10, epochs=1000, eta=0.01, lmbda=0.00001, func=sigmoid)
    nn.train()

    plt.figure()
    plt.plot(x, y_exact, label="Exact")
    plt.plot(x, nn.predict_probabilities(X), label="Prediction")
    plt.savefig("figures/FFNN prediction")
    plt.legend()
    plt.show()


"""