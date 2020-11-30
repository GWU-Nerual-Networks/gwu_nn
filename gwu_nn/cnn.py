import numpy as np
from gwu_nn.loss_functions import MSE, LogLoss, CrossEntropy
from gwu_nn.gwu_network import GWUNetwork

class CNN(GWUNetwork):
    def __init__(self):
        super().__init__()

    def conv_2d(self, in_mat, kernel):
        """Slides the kernel over the 2d input and computes a weighted sum."""

        #TODO: Consider zero padding?

        # The input matrix should be at least as large as the kernel.
        assert len(in_mat) >= len(kernel)
        assert len(in_mat[0]) >= len(kernel[0])

        dim1 = len(in_mat) - len(kernel) + 1
        dim2 = len(in_mat[0]) - len(kernel[0]) + 1

        out_mat = [[0] * dim2 for i in range(dim1)]

        for i in range(dim1):
            for j in range(dim2):
                w_sum = self.weighted_sum([row[j:j+len(kernel[0])] for row in in_mat[i:i+len(kernel)]], kernel)
                out_mat[i][j] = w_sum

        return out_mat

    def max_pool(self, in_mat, size, stride):
        """Pools the data using the maximum function."""

        # Drops unused columns (known as "valid padding" in tensorflow terminology).
        dim1 = int((len(in_mat) - size) / stride) + 1
        dim2 = int((len(in_mat[0]) - size) / stride) + 1

        out_mat = [[0] * dim2 for i in range(dim1)]

        for i in range(0, dim1):
            for j in range(0, dim2):
                pool = self.max_mat([row[j*stride:j*stride+size] for row in in_mat[i*stride:i*stride+size]])
                out_mat[i][j] = pool

        return out_mat
        

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

    def max_mat(self, mat):
        """Helper method to compute the maximum value of a matrix."""

        return max(self.flatten(mat))
