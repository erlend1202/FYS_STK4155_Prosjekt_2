from FrankeFunction import FrankeFunctionNoised, FrankeFunction
from FFNN import FeedForwardNeuralNetwork
from sample_data import create_data_samples, DataSamplesType
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from mean_square_error import MSE

# Running the neural network on train and test dataset created from franke function
if __name__ == "__main__":
    x, y, z = create_data_samples(DataSamplesType.TEST)
    X = create_design_matrix(x, y, 5)

    X_train, X_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    layers = [5, 4, 3]
    nn = FeedForwardNeuralNetwork(X_train, y_train, layers, 1, 10, epochs=200, eta=0.01, lmbda=0.01)
    nn.train()

    prediction_train = nn.predict_probabilities(X_train)
    prediction_test = nn.predict_probabilities(X_test)

    train_mse = MSE(y_train, prediction_train)
    test_mse = MSE(y_test, prediction_test)

    print(train_mse, test_mse)