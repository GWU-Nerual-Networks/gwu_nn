import numpy as np
from gwu_nn.activation_functions import Softmax
from abc import ABC, abstractmethod


class LossFunction(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(cls, y_true, y_pred):
        pass

    @abstractmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        pass


class MSE(LossFunction):

    @classmethod
    def loss(cls, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size

class LogLoss(LossFunction):

    @classmethod
    def loss(cls, y_true, y_pred):
        return np.mean(-np.log(y_pred)*y_true + -np.log(1-y_pred)*(1-y_true))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        return -np.sum(y_true - y_pred)


class CrossEntropy(LossFunction):

    @classmethod
    def loss(cls, y_true, y_pred):
        return -np.mean(y_true*np.log(y_pred))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        m = y_true.shape[0]
        grad = Softmax.(y_pred)
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad
