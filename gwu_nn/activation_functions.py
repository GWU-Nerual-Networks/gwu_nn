import numpy as np
from abc import ABC, abstractmethod


# Todo: Change activations to remove the need for this decorator
def vectorize_activation(activation):
    """Decorator that ensures that activations are vectorized when used"""
    def wrapper(*args):
        vec_activation = np.vectorize(activation)
        input = args[1]
        return vec_activation(args[0], input)
    return wrapper


class ActivationFunction(ABC):
    """Abstract class that defines base functionality for activation functions"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def activation(cls, x):
        pass

    @abstractmethod
    def activation_partial_derivative(cls, x):
        pass


class SigmoidActivation(ActivationFunction):
    """Implements the sigmoid activation function typically used for logistic regression"""

    @classmethod
    @vectorize_activation
    def activation(cls, x):
        """Scales inputs to (0,1)

        Args:
            x (np.array): input into the layer/activation function

        Returns:
            np.array(floats): sigmoid(x)
        """
        out = 1 / (1 + np.exp(-x))
        return out

    @classmethod
    @vectorize_activation
    def activation_partial_derivative(cls, x):
        """Applies the partial derivative of the sigmoid function

        Args:
            x (np.array): partial derivative up to this layer/activation function

        Returns:
            np.array(floats): derivative of network up to this activation/layer
        """
        return np.exp(-x) / (1 + np.exp(-x))**2


class RELUActivation(ActivationFunction):

    @classmethod
    @vectorize_activation
    def activation(cls, x):
        """Zeroes out negative values

        Args:
            x (np.array): input into the layer/activation function

        Returns:
            np.array(floats): ReLU(x)
        """
        if x > 0:
            return x
        else:
            return 0

    @classmethod
    @vectorize_activation
    def activation_partial_derivative(cls, x):
        """Applies the partial derivative of the ReLU function to the input

        Args:
            x (np.array): partial derivative up to this layer/activation function

        Returns:
            np.array(floats): derivative of network up to this activation/layer
        """
        if x > 0:
            return 1
        else:
            return 0


class SoftmaxActivation(ActivationFunction):

    @classmethod
    def activation(cls, x):
        """Applies the softmax function to the input array

        Args:
            x (np.array): input into the layer/activation function

        Returns:
            np.array(floats): Softmax(x)
        """
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    # TODO: Fix partial div implementation of softmax
    @classmethod
    def activation_partial_derivative(cls, x):
        """Applies the partial derivative of the sigmoid function

        Args:
            x (np.array): partial derivative up to this layer/activation function

        Returns:
            np.array(floats): derivative of network up to this activation/layer
        """
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
