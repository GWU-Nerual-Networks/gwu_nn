import numpy as np
from gwu_nn.loss_functions import MSE, LogLoss, CrossEntropy
from gwu_nn.gwu_network import GWUNetwork

class CNN(GWUNetwork):
    def __init__(self):
        super().__init__()

    # Slides the kernel over the 2d input and computes a weighted sum
    def conv_2d(self):
        pass

    # Pools the data using an average.
    def average_pool(self):
        pass

    # Flattens the data into a (1d) vector so as to maintain compatibility with the rest of the library.
    def flatten(self):
        pass