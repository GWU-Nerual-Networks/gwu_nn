import numpy as np
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
        logprobs = 0
        # Compute cross-entropy loss
        scale_factor = 1 / float(y_pred.shape[1] - 1)
        for c in range(y_true.shape[0]):  # For each class
            if y_true[c][0] != 0:  # Positive classes
                logprobs = -np.log(y_pred[0][c]) * y_true[c][0] * scale_factor  # We sum the loss per class for each element of the batch

        loss = np.sum(logprobs) / 1  # number of samples in batch
        return loss

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        scale_factor = 1 / float(y_pred.shape[1]-1)
        grad = np.zeros(y_true.shape[0])
        for c in range(y_true.shape[0]):  # For each class
            if y_true[c][0] != 0:  # If positive class
                grad = scale_factor * (y_pred[0][c] - 1) + (1 - scale_factor) * y_pred[0][c]
        return grad / 1  # number of samples in batch
