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

class Conv2D(Layer):

    def __init__(self, num_filters = 5, kernel_size = 3, stride = 1, add_bias=False, activation=None, input_shape=None):
        super().__init__(activation)
        self.type = None
        self.name = "Conv2D"
        self.num_filters = num_filters
        self.input_shape = input_shape
        if type(input_shape == 'tuple'):
            self.input_size = input_shape[0]
        else:
            self.input_size = input_shape
        self.add_bias = add_bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        convolved_dim = int((self.input_size - kernel_size)/stride) + 1
        self.output_size = convolved_dim

    def init_weights(self, input_size):
        """Initialize the weights for the layer and initializes the filters
        """
        if self.input_size is None:
            self.input_size = input_size

        #TO DO

        #initialize self.weights
        #initialize self.bias
        #initialize self.filters

    @apply_activation_forward
    def forward_propagation(self, input):
        """Applies the forward propagation for a Conv2D layer. calls convolve function """
        output = self.convolve(self.stride, self.add_bias, self.kernel_size,input,self.filters)
        return output

    @apply_activation_backward
    def backward_propogation(cls, output_error, learning_rate):
        """Applies the backward propagation for a Conv2D layer."""
        pass

    def convolve(stride, add_bias, kernel_size, sample, filters):
        """ Convolving filters over the image"""

        num_filters = len(filters)
        convolved_dim = int((input_dim - kernel_size)/stride) + 1
        all_filter_outputs = np.zeros((num_filters, convolved_dim, convolved_dim))

        for i in range(num_filters):
            x_pos = 0
            y_pos = 0
            for j in range(convolved_dim):
                for k in range(convolved_dim):
                    all_filter_outputs[i, j, k] = 0
                    for l in range(kernel_size):
                        for m in range(kernel_size):
                            all_filter_outputs[i, j, k] += sample[j+l, k+m]* filters[i,l,m]

                    x_pos += stride
                y_pos += stride

        return all_filter_outputs


class MaxPooling2D(Layer):

    def __init__(self, pool_size, activation=None,  stride = 1):
        super().__init__(activation)
        self.type = None
        self.name = "MaxPooling2D"
        self.pool_size = pool_size
        self.stride = pool_size



    def pool(stride, kernel_size, input_dim, sample):
        """ Downsampling the image by taking the max values"""

        pooled_dim = int((input_dim - self.pool_size)/stride) + 1
        pooled_sample = np.zeros((pooled_dim, pooled_dim))

        for i in range(pooled_dim):
            x_pos = 0
            y_pos = 0
            for j in range(pooled_dim):
                pooled_sample[i,j] = np.max(sample[y_pos:y_pos+kernel_size, x_pos:x_pos+kernel_size])
                x_pos += stride
            y_pos += stride

    def init_weights(self, input_size):

        self.input_size = input_size
        pooled_dim = int((input_size - self.pool_size)/self.stride) + 1
        self.output_size =  pooled_dim
    @apply_activation_forward
    def forward_propagation(cls, input):
        # TO DO: pool function is called here
        pass

    @apply_activation_backward
    def backward_propogation(cls, output_error, learning_rate):
        pass


class Flatten(Layer):

    def __init__(self, activation=None):
        super().__init__(activation)
        self.type = None
        self.name = "Flatten"

    def init_weights(self, input_size):

        self.input_size = input_size
        # TO DO: remeber to take the number of filters into account
        self.output_size =  input_size*input_size

    @apply_activation_forward
    def forward_propagation(cls, input):
        #flattens the input so that it has only 1 dimension, reshaping takes place here
        self.input = input.reshape(1, -1)
        output = self.input
        return output

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
    def backward_propagation(self, output_error, learning_rate):
        """Applies the backward propagation for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights (dot product of the layer's input values and the output_error).

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        if self.add_bias:
            self.bias -= learning_rate * output_error
        return input_error
