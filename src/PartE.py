from sklearn.model_selection import train_test_split
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from LogisticRegression import *
from matplotlib.colors import LogNorm

def grid_search_hyperparameters(plot_title, M = 10, epochs = 100, eta_vals = np.logspace(-5, 1, 7), lmd_vals = np.logspace(-5, 1, 7), verbose = False):
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size,
                                                    test_size=test_size)
    acc_values = np.zeros((len(eta_vals), len(lmd_vals)))

    for i, eta in enumerate(eta_vals):
        for j, lmd in enumerate(lmd_vals):
            lg = NumpyLogReg()
            lg.fit(X_train,Y_train, eta=eta, epochs=epochs, M = M, lmbda=lmd)
            Y_predict = lg.predict(X_test)
            acc = accuracy_score_numpy(Y_test, Y_predict, conf=False)

            acc_values[i, j] = acc

            if verbose:
                print(f"eta:{eta}, lambda:{lmd} gives acc {acc}")

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
    show_values_in_heatmap(acc_values, plt.gca())
    plt.title(plot_title)
    plt.xticks(np.arange(0, len(eta_vals)), labels_x)
    plt.yticks(np.arange(0, len(lmd_vals)), labels_y)
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")
    plt.imshow(acc_values, norm=LogNorm())
    plt.colorbar()
    plt.savefig(f"figures/{plot_title}")

def epochs_plot(plot_title, max_epochs, lmda=0.01, eta=0.1, increment = 5, verbose = False):
    num = int(max_epochs/increment)
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size,
                                                    test_size=test_size)
    acc_values = np.zeros(num)

    for epoch in range(0,max_epochs,increment):
        #nn = FeedForwardNeuralNetwork(X, y, layers, 1, 10, epochs=epoch, eta=eta, lmbda=lmda)
        #nn.train()
        #mse_values[epoch] = MSE(y_exact, nn.predict_probabilities(X))
        lg = NumpyLogReg()
        lg.fit(X_train,Y_train, eta=eta, epochs=epoch, M = 5, lmbda=lmda)
        Y_predict = lg.predict(X_test)
        acc = accuracy_score_numpy(Y_test, Y_predict, conf=False)
        acc_values[int(epoch/increment)] = acc
        
        if verbose:
            print(f"Testing epoch {epoch}, acc is {acc}")
    
    plt.figure()
    x = np.linspace(0,max_epochs,num)
    plt.title(plot_title)
    plt.plot(x, acc_values)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"figures/{plot_title}")


def test_classification():    
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size,
                                                    test_size=test_size)
    #Important to change n_categories to 2 and problem to anything else than regression
    lg = NumpyLogReg()
    lg.fit(X_train,Y_train, eta=0.1, epochs=100)
    Y_predict = lg.predict(X_test)
    #print(accuracy_score_numpy(Y_test, Y_predict))
    acc = accuracy_score_numpy(Y_test, Y_predict, conf=True)
    print(acc)


if __name__ == "__main__":
    test_classification()
    #grid_search_hyperparameters("Training_accuracy_Logistic", verbose=True)
    #epochs_plot("Epochs_Logistic", 200, verbose=True)