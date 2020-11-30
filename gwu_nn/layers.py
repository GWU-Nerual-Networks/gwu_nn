import numpy as np
from abc import ABC, abstractmethod
from gwu_nn.activation_layers import Sigmoid, RELU, Softmax

activation_functions = {'relu': RELU, 'sigmoid': Sigmoid, 'softmax': Softmax}


def apply_activation_forward(forward_pass):
    """Decorator that ensures that a layer's activation function is applied after the layer during forward
    propagation.
    """
    def wrapper(*args):
        output = forward_pass(args[0], args[1])
        if args[0].activation:
            return args[0].activation.forward_propagation(output)
        else:
            return output
    return wrapper


def apply_activation_backward(backward_pass):
    """Decorator that ensures that a layer's activation function's derivative is applied before the layer during
    backwards propagation.
    """
    def wrapper(*args):
        output_error = args[1]
        learning_rate = args[2]
        if args[0].activation:
            output_error = args[0].activation.backward_propagation(output_error, learning_rate)
        return backward_pass(args[0], output_error, learning_rate)
    return wrapper


class Layer():

    def __init__(self, activation=None):
        self.type = "Layer"
        if activation:
            self.activation = activation_functions[activation]()
        else:
            self.activation = None

    @apply_activation_forward
    def forward_propagation(cls, input):
        pass

    @apply_activation_backward
    def backward_propogation(cls, output_error, learning_rate):
        pass


class Dense(Layer):

    def __init__(self, output_size, add_bias=False, activation=None, input_size=None):
        super().__init__(activation)
        self.type = None
        self.name = "Dense"
        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = add_bias
        self.v = {}
        self.s = {}


    def init_weights(self, input_size):
        """Initialize the weights for the layer based on input and output size

        Args:
            input_size (np.array): dimensions for the input array
        """
        if self.input_size is None:
            self.input_size = input_size

        self.weights = np.random.randn(input_size, self.output_size) / np.sqrt(input_size + self.output_size)
        if self.add_bias:
            self.bias = np.random.randn(1, self.output_size) / np.sqrt(input_size + self.output_size)


    @apply_activation_forward
    def forward_propagation(self, input):
        """Applies the forward propagation for a densely connected layer. This will compute the dot product between the
        input value (calculated during forward propagation) and the layer's weight tensor.

        Args:
            input (np.array): Input tensor calculated during forward propagation up to this layer.

        Returns:
            np.array(float): The dot product of the input and the layer's weight tensor."""
        self.input = input
        output = np.dot(input, self.weights)
        if self.add_bias:
            return output + self.bias
        else:
            return output

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate, optimizer=None):
        """Applies the backward propagation for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights (dot product of the layer's input values and the output_error).

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        if optimizer == 'adam':
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

            self.v["dW"] = np.zeros(self.weights.shape)
            self.s["dW"] = np.zeros(self.weights.shape)
            self.v["dW"] = (self.beta1 * self.v["dW"]) + ((1 - self.beta1) * self.weights)
            self.v["dW"] = self.v["dW"] / (1 - np.power(self.beta1, 2))
            self.s["dW"] = (self.beta2 * self.s["dW"]) + (
                    (1 - self.beta2) * np.power(weights_error, 2))
            self.s["dW"] = self.s["dW"] / (1 - np.power(self.beta2, 2))
            self.weights -= (learning_rate * (
                    self.v["dW"] / (np.sqrt(self.s["dW"]) + self.epsilon)))

            if self.add_bias:
                self.v["db"] = np.zeros(self.bias.shape)
                self.s["db"] = np.zeros(self.bias.shape)

                self.v["db"] = (self.beta1 * self.v["db"]) + ((1 - self.beta1) * self.bias)

                self.v["db"] = self.v["db"] / (1 - np.power(self.beta1, 2))

                self.s["db"] = (self.beta2 * self.s["db"]) + (
                        (1 - self.beta2) * np.power(self.bias, 2))

                self.s["db"] = self.s["db"] / (1 - np.power(self.beta2, 2))

                self.bias -= (learning_rate * (
                        self.v["db"] / (np.sqrt(self.s["db"]) + self.epsilon)))

        else:
            self.weights -= learning_rate * weights_error
            if self.add_bias:
                self.bias -= learning_rate * np.sum(output_error)
        return input_error
