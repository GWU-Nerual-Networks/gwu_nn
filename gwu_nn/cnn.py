import numpy as np
from gwu_nn.loss_functions import MSE, LogLoss, CrossEntropy
from gwu_nn.gwu_network import GWUNetwork

class CNN(GWUNetwork):
    def __init__(self):
        super().__init__()

    def conv_2d(self):
        """Slides the kernel over the 2d input and computes a weighted sum"""
        pass

    def average_pool(self):
        """Pools the data using an average."""
        pass

    def flatten(self, mat):
        """Flattens the matrix into a (1d) vector so as to maintain compatibility with the rest of the library."""

        v = []
        for i in range(len(mat)):
            v.extend(mat[i])

        return v


    def weighted_sum(self, mat1, mat2):
        """Helper method to compute the weighted sum of two (sub-)matrices."""

        # The matrices ought to be of the same size.
        assert len(mat1) == len(mat2)
        assert len(mat1[0]) == len(mat2[0])

        ws = 0
        for i in range(len(mat1)):
            for j in range(len(mat1[0])):
                ws += mat1[i][j] * mat2[i][j]

        return ws
