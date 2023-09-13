# Pavel, Hamza Reza

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.bias_list = [] #the bias of the layers
        self.weights = [] #the weight of the layers
        self.transfer_function_names = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        if len(self.weights) == 0:
            weight = tf.Variable(np.random.randn(self.input_dimension,num_nodes), name="weights",  dtype="float64", trainable=True)
        else:
            weight = tf.Variable(np.random.randn(self.weights[-1].get_shape()[1], num_nodes), name="weights", dtype="float64",  trainable=True)
        bias = tf.Variable(np.random.randn(1, num_nodes), trainable=True)
        self.weights.append(weight)
        self.bias_list.append(bias)
        self.transfer_function_names.append(transfer_function)
        return None

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.bias_list[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.bias_list[layer_number] = biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(y, y_hat)

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        inputs = X.copy()
        for layer in range(0,len(self.weights)):
            outputs = (tf.matmul(inputs, self.weights[layer]) + self.bias_list[layer])
            outputs = self.activation_function(outputs, self.transfer_function_names[layer])
            inputs = outputs
        return outputs

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size)
        for epoch in range(num_epochs):
            for step, (X, Y) in enumerate(dataset):
                with tf.GradientTape(persistent=True) as tape:
                    Y_prediction = self.predict(X.numpy())
                    loss = self.calculate_loss(Y, Y_prediction)
                for layer in range(0, len(self.weights)):
                    dloss_dw, dloss_db = tape.gradient(loss, [self.weights[layer] , self.bias_list[layer]])
                    self.weights[layer].assign_sub(alpha * dloss_dw)
                    self.bias_list[layer].assign_sub(alpha * dloss_db)
                    #self.weights[layer].assign_sub(alpha * tape.gradient(loss, [self.weights[layer]])[0])
                    #self.bias_list[layer].assign_sub(alpha * tape.gradient(loss, [self.bias_list[layer]])[0])
                del tape

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred,axis =1)
        error_count = 0
        for i in range(0,len(y)):
            if y[i]!=y_pred[i]:
              error_count = error_count + 1
        percent_error = error_count/len(y)
        return percent_error

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        Y_predicted = self.predict(X)
        Y_predicted = tf.math.argmax(Y_predicted, axis = 1)
        # Compute confusion matrix
        confusion = tf.math.confusion_matrix(y, Y_predicted)
        return confusion

    def activation_function(self, x, transfer_function_name):
        if transfer_function_name.lower() == "linear":
            res = x
        elif transfer_function_name.lower() == "relu":
            res = tf.nn.relu(x)
        elif transfer_function_name.lower() == "sigmoid":
            res = tf.nn.sigmoid(x)
        return res
