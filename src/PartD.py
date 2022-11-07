import numpy as np
from FFNN import *
from sklearn.datasets import load_breast_cancer

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

if __name__ == "__main__": 
    # Kjøre for flere hyperparametere eta, lambda
    layers = [5, 4, 3]
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    Y_onehot = to_categorical_numpy(Y)
    #Important to change n_categories to 2 and problem to anything else than regression
    nn = FeedForwardNeuralNetwork(X, Y_onehot, layers, 2, 10, epochs=200, eta=0.3, lmbda=0.01, func=sigmoid, problem="classification")
    nn.train()
    Y_predict = nn.predict(X)
    print(accuracy_score_numpy(Y, Y_predict))