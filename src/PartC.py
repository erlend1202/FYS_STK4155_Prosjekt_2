from FFNN import *
import numpy as np
from design_matrix import *
from sklearn.model_selection import train_test_split
from sample_data import *


if __name__ == "__main__":
    x, y, z = create_data_samples(DataSamplesType.TEST)
    X = create_design_matrix(x, y, 5)

    X_train, X_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    layers = [5, 4, 3]

    grid_search_hyperparameters(X_train, y_train, y_train, layers, "Training accuracy (RELU)", relu)
    epochs_plot(X_train, y_train, y_train, layers, "Epochs (RELU)", 50, 0.01, 0.01, relu)
    
    grid_search_hyperparameters(X_train, y_train, y_train, layers, "Training accuracy (Leaky RELU)", leaky_relu)
    epochs_plot(X, y_train, y_train, layers, "Epochs (Leaky RELU)", 50, 0.01, 0.01, leaky_relu)