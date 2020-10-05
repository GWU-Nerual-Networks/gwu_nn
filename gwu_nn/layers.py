import numpy as np
from abc import ABC, abstractmethod
from gwu_nn.activation_layers import Sigmoid, RELU

activation_functions = {'relu': RELU, 'sigmoid': Sigmoid}


def apply_activation_forward(forward_pass):
    def wrapper(*args):
        output = forward_pass(args[0], args[1])
        if args[0].activation:
            return args[0].activation.forward_propagation(output)
        else:
            return output
    return wrapper


def apply_activation_backward(backward_pass):
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

    def __init__(self, input_size, output_size, add_bias=False, activation=None):
        super().__init__(activation)
        self.type = None
        self.name = "Dense"
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.add_bias = add_bias
        if add_bias:
            self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    @apply_activation_forward
    def forward_propagation(self, input):
        self.input = input
        output = np.dot(input, self.weights)
        if self.add_bias:
            return output + self.bias
        else:
            return output

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        if self.add_bias:
            self.bias -= learning_rate * output_error
        return input_error
