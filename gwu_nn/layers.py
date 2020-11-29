import numpy as np
from abc import ABC, abstractmethod
from gwu_nn.activation_layers import Sigmoid, RELU, Softmax
import matplotlib.pyplot as plt

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

    def __init__(self, num_filters = 1, kernel_size = 3, stride = 1, add_bias=False, activation=None, input_shape=None):
        super().__init__(activation)
        self.type = None
        self.name = "Conv2D"
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.input_size = None
        if type(input_shape == 'tuple') and input_shape != None:
            self.input_size = input_shape[0]
        elif input_shape != None:
            self.input_size = input_shape
        self.add_bias = add_bias
        self.kernel_size = kernel_size
        self.stride = stride
        if self.input_size != None:
            convolved_dim = int((self.input_size - kernel_size)/stride) + 1
            self.output_size = convolved_dim

    def init_weights(self, input_size):
        """Initialize the filters
        """
        if self.input_size is None:
            self.input_size = input_size
            convolved_dim = int((self.input_size - self.kernel_size)/self.stride) + 1
            self.output_size = convolved_dim

        filter_shape = (self.num_filters, self.kernel_size, self.kernel_size)
        stddev = 1/np.sqrt(np.prod(filter_shape ))
        self.filters = np.random.normal(loc = 0, scale = stddev, size = filter_shape )
        #TO DO
        #initialize self.bias

    @apply_activation_forward
    def forward_propagation(self, input):
        """Applies the forward propagation for a Conv2D layer. calls convolve function """
        output = self.convolve(self.stride, self.add_bias, self.kernel_size, input, self.filters)
        self.input = input #saves the input for later use in backprop
        # print(output.shape)
        # plt.imshow(output[0], cmap='gray')
        # plt.show()
        # plt.imshow(self.filters[0], cmap='gray')
        # plt.show()
        return output

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        """Applies the backward propagation for a Conv2D layer."""
        num_filters = self.filters.shape[0]
        filter_gradient = np.zeros(self.filters.shape)
        input_deriv = np.zeros(self.input.shape)

        for i in range(num_filters):
            y_pos = 0
            for j in range(self.convolved_dim):
                x_pos = 0
                for k in range(self.convolved_dim):
                    # TO DO: FIX THE ERROR WITH SELF.INPUT(I,J,K) HERE and input deriv [i,j,k]
                    filter_gradient[i] += output_error[i, j, k] * self.input[y_pos:y_pos+self.kernel_size, x_pos:x_pos+self.kernel_size]
                    input_deriv[y_pos:y_pos+self.kernel_size, x_pos:x_pos+self.kernel_size] += output_error[i, j, k] * self.filters[i]
                    x_pos += self.stride
                y_pos += self.stride

            self.filters[i] -= learning_rate * filter_gradient[i]


        return input_deriv


    def convolve(self, stride, add_bias, kernel_size, sample, filters):
        """ Convolving filters over the image"""

        num_filters = filters.shape[0]
        convolved_dim = int((self.input_size - kernel_size)/stride) + 1
        self.convolved_dim = convolved_dim
        all_filter_outputs = np.zeros((num_filters, convolved_dim, convolved_dim))

        # TO DO: Double check convolve logic, fix the issue with sample[i, j, k] (take num filters into acccount when conv2d isn't your first layer)
        for i in range(num_filters):
            y_pos = 0
            for j in range(convolved_dim):
                x_pos = 0
                for k in range(convolved_dim):
                    all_filter_outputs[i, j, k] = 0
                    for l in range(kernel_size):
                        for m in range(kernel_size):
                            all_filter_outputs[i, j, k] += sample[j+l, k+m]* filters[i,l,m]

                    x_pos += stride
                y_pos += stride
        return all_filter_outputs


class MaxPooling2D(Layer):

    def __init__(self, pool_size, activation=None,  stride = None):
        super().__init__(activation)
        self.type = None
        self.name = "MaxPooling2D"
        self.pool_size = pool_size

        #if stride has not been specified, set stride equal to pool size
        if stride == None:
            self.stride = pool_size
        else:
            self.stride = stride

    def pool(self,stride, pool_size, sample):
        """ Downsampling the image by taking the max values"""
        self.input_shape = sample.shape
        self.sample = sample #saves the input samples for later use in backward prop
        num_filters = sample.shape[0]
        input_dim = sample.shape[1]
        pooled_dim = int((input_dim - self.pool_size)/stride) + 1
        self.pooled_dim = pooled_dim
        pooled_sample = np.zeros((num_filters,pooled_dim, pooled_dim))

        # TO DO: Double check the maxpool logic
        for i in range(num_filters):
            y_pos = 0
            for j in range(pooled_dim):
                x_pos = 0
                for k in range(pooled_dim):
                    pooled_sample[i,j,k] = np.max(sample[i, y_pos:y_pos+self.pool_size, x_pos:x_pos+self.pool_size])
                    x_pos += stride
                y_pos += stride

        return pooled_sample

    def init_weights(self, input_size):
        self.input_size = input_size
        pooled_dim = int((input_size - self.pool_size)/self.stride) + 1
        self.output_size =  pooled_dim

    @apply_activation_forward
    def forward_propagation(self, input):
        output = self.pool(self.stride, self.pool_size, input)
        # print(output.shape)
        # plt.imshow(output[0], cmap='gray')
        # plt.show()
        return output

    def find_max_indices(self, sub_array):
        # code inspired from https://thispointer.com/find-max-value-its-index-in-numpy-array-numpy-amax/
        result = np.where(sub_array == np.amax(sub_array))
        coordinate_list = list(zip(result[0], result[1]))
        #always returns the first coordinates (in case values in a subarray are all equal)
        return coordinate_list[0]

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        #the output is an array with the original shape filled with zeros except max indices, doesn't mess with filters
        input_gradients = np.zeros(self.input_shape)
        num_filters = self.input_shape[0]

        # TO DO: Double check the maxpool backward logic
        for i in range(num_filters):
            y_pos = 0
            for j in range(self.pooled_dim):
                x_pos = 0
                for k in range(self.pooled_dim):
                    sub_array = self.sample[i, y_pos:y_pos+self.pool_size, x_pos:x_pos+self.pool_size]
                    indices= self.find_max_indices(sub_array)
                    input_gradients[i, j+indices[0], k + indices[1]] = output_error[i,j,k]
                    x_pos += self.stride
            y_pos += self.stride

        return input_gradients


class Flatten(Layer):

    def __init__(self, activation=None, num_filters = 1):
        super().__init__(activation)
        self.type = None
        self.name = "Flatten"
        self.num_filters = num_filters

    def init_weights(self, input_size):

        self.input_size = input_size
        # TO DO: remeber to take the number of filters into account in a fancier way
        self.output_size =  input_size*input_size * self.num_filters

    @apply_activation_forward
    def forward_propagation(self, input):
        #flattens the input so that it has only 1 dimension, reshaping takes place here
        self.input = input.reshape(1, -1)
        output = self.input
        # print(output.shape)
        return output

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):

        #back prop does not mess with weights for flatten layer, just reshapes it to the pool shape
        input_error = output_error.reshape(self.num_filters, self.input_size, self.input_size)
        # print("recreating the pooling shape ", input_error.shape)
        return input_error

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
        # print(output.shape)
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
