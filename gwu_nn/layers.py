import numpy as np


class Dense:
    def __init__(self, input_size, output_size, add_bias=False):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.add_bias = add_bias
        if add_bias:
            self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def forward_propagation(self, input):
        self.input = input
        output = np.dot(input, self.weights)
        if self.add_bias:
            return output + self.bias
        else:
            return output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # bias_error = output_error

        self.weights -= learning_rate * weights_error
        if self.add_bias:
            self.bias -= learning_rate * output_error
        return input_error
