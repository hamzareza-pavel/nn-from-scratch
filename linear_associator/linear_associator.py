# Pavel, Hamza Reza
# 1001_741_797
# 2021_10_07
# Assignment_02_01

import math
import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.weights = np.zeros(shape=(number_of_nodes, input_dimensions))
        self.transfer_function = transfer_function
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)
        return None

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        num_rows, num_cols = W.shape
        if num_rows == self.number_of_nodes and num_cols == self.input_dimensions:
            self.weights = W
            return None
        else:
            return -1


    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        res = np.dot(self.weights, X)
        predicted = np.vectorize(self.activation_function)(res)
        return predicted

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        X_plus = np.linalg.pinv(X)#np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X))
        self.weights = np.dot(y, X_plus)

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        if learning.lower() == "Delta".lower():
            for time in range(num_epochs):
                input_dim, n_samples = X.shape
                num_batches = math.ceil(n_samples/batch_size)
                X_split = np.hsplit(X, num_batches)
                Y_split = np.hsplit(y, num_batches)
                for X_b, Y_b in zip(X_split, Y_split):
                    Y_predicted = self.predict(X_b)
                    e = Y_b - Y_predicted
                    X_mod = np.transpose(X_b)
                    self.weights = self.weights + alpha *  np.dot(e,  X_mod)
        elif learning.lower() == "Filtered".lower():
                input_dim, n_samples = X.shape
                num_batches = math.ceil(n_samples/batch_size)
                X_split = np.hsplit(X, num_batches)
                Y_split = np.hsplit(y, num_batches)
                for X_b, Y_b in zip(X_split, Y_split):
                    X_mod = np.transpose(X_b)
                    self.weights = (1-gamma)*self.weights + alpha *  np.dot(Y_b,  X_mod)
        elif learning.lower() == "Unsupervised_hebb".lower():
                input_dim, n_samples = X.shape
                num_batches = math.ceil(n_samples/batch_size)
                X_split = np.hsplit(X, num_batches)
                Y_split = np.hsplit(y, num_batches)
                for X_b, Y_b in zip(X_split, Y_split):
                    Y_predicted = self.predict(X_b)
                    X_mod = np.transpose(X_b)
                    self.weights = self.weights + alpha *  np.dot(Y_predicted,  X_mod)
        return None

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        y_predicted = self.predict(X)
        mse = (np.square(y - y_predicted)).mean()
        return np.sum(mse)

    def activation_function(self, x):
        if self.transfer_function.lower() == "Hard_limit".lower():
            if x >= 0:
                res=1
            else:
                res=0
        elif self.transfer_function.lower() == "Linear".lower():
            res=x
        return res

