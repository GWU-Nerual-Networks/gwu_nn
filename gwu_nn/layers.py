import numpy as np
from scipy.signal import convolve2d
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
        #print(input)
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
        #print("stuff:", self.input.T, output_error)
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        #print("OKAY NEW WEIGHTS ARE", self.weights)
        if self.add_bias:
            self.bias -= learning_rate * output_error
        #print("WEIGHTS_ERROR, DECREMENT", weights_error, learning_rate * weights_error)
        return input_error

class Conv_2d(Layer):
    def __init__(self, input_size, kernel_size, activation=None):
        super().__init__(activation)
        self.type = None
        self.name = "Conv_2d"
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.output_size = ((input_size - kernel_size) + 1, (input_size - kernel_size) + 1)

    def init_weights(self, input_size):
        self.kernel = np.random.randn(self.kernel_size, self.kernel_size) / np.sqrt(2 * self.kernel_size)


    @apply_activation_forward
    def forward_propagation(self, input):
        """Applies the forward propogation for a Conv_2d layer. This convolves the input with the kernel."""
        self.input = input

        kern = self.kernel
        assert len(self.input) >= len(kern)
        assert len(self.input[0]) >= len(kern[0])

        # Dimensions of the result of the convolution.
        dim1 = len(self.input) - len(kern) + 1
        dim2 = len(self.input[0]) - len(kern[0]) + 1

        output = [[0.0] * dim2 for i in range(dim1)]

        for i in range(dim1):
            for j in range(dim2):
                w_sum = self.weighted_sum([row[j:j+len(kern[0])] for row in self.input[i:i+len(kern)]], kern)
                output[i][j] = w_sum

        return np.array(output)

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        """Applies the backward propogation for a Conv_2d layer. This takes the output_error, and also uses the 
        stored kernel weights and input."""
        weights_error = [[0] * self.kernel_size for i in range(self.kernel_size)]
        
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                weights_error[i][j] = self.weighted_sum([row[j:j+len(output_error[0])] for row in self.input[i:i+len(output_error)]], output_error)

        self.kernel -= learning_rate * np.array(weights_error)
        rotated_kernel = np.rot90(self.kernel, 2)
        inputs_error = convolve2d(rotated_kernel, output_error)

        return inputs_error


    def weighted_sum(self, mat1, mat2):
        """Helper method to compute the weighted sum of two (sub-)matrices."""

        assert len(mat1) == len(mat2)
        assert len(mat1[0]) == len(mat2[0])

        ws = 0
        for i in range(len(mat1)):
            for j in range(len(mat1[0])):
                ws += mat1[i][j] * mat2[i][j]

        return ws

class Flatten(Layer):
    def __init__(self, input_size):
        super().__init__(None)
        self.input_size = input_size
        self.output_size = input_size[0] ** 2

    def init_weights(self, input_size):
        pass
    
    def forward_propagation(self, input):
        self.orig_shape = input.shape
        flat = np.array([input.flatten()])

        return flat
    
    def backward_propagation(self, flat, learning_rate):
        return flat.reshape(self.orig_shape)


class Max_Pool(Layer):
    def __init__(self, kernel_size=2, stride=2, activation=None, input_size=None):
        super().__init__(None)
        self.type = None
        self.name = "Max_Pool"
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.stride = stride
        self.output_size = int((self.input_size - self.kernel_size) / self.stride) + 1

    def init_weights(self, input_size):
        pass

    @apply_activation_forward
    def forward_propagation(self, input):
        # Drops unused columns (known as "valid padding" in tensorflow terminology).
        self.input = input

        dim1 = int((len(self.input) - self.kernel_size) / self.stride) + 1
        dim2 = int((len(self.input[0]) - self.kernel_size) / self.stride) + 1

        output = np.array([[0.0] * dim2 for i in range(dim1)])
        self.backprop_mat = np.zeros((len(self.input), len(self.input[0])))
        self.coords_pairs = []

        for i in range(0, dim1):
            for j in range(0, dim2):
                max, coords = self.mat_max_and_coords(self.input, i*self.stride, i*self.stride+self.kernel_size, j*self.stride, j*self.stride+self.kernel_size)
                output[i][j] = max
                self.coords_pairs.append((coords[0], coords[1]))

        return np.array(output)

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        # The gradient is zero except at the maximum values themselves (and there, the gradient is unchanged).
        # Thankfully, we saved this data which saves us from having to backtrack from scratch.
        coords_used_up = 0
        for i in range(len(output_error)):
            for j in range(len(output_error[0])):
                self.backprop_mat[self.coords_pairs[coords_used_up][0]][self.coords_pairs[coords_used_up][1]] = output_error[i][j]
                coords_used_up += 1

        return self.backprop_mat

    def mat_max_and_coords(self, mat, low1, up1, low2, up2):
        """Given a matrix and boundaries for a submatrix, return the maximum value and its index."""
        max_so_far = -2**32
        max_coords_so_far = (-1, -1)
        for i in range(low1, up1):
            for j in range(low2, up2):
                if mat[i][j] > max_so_far:
                    max_so_far = mat[i][j]
                    max_coords_so_far = (i, j)

        return max_so_far, max_coords_so_far

    def flatten(self, mat):
        """Flattens the matrix into a (1d) vector to permit max function."""

        v = []
        for i in range(len(mat)):
            v.extend(mat[i])

        return v

    def max_mat(self, mat):
        """Helper method to compute the maximum value of a matrix."""

        return max(self.flatten(mat))




