import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Conv_2d, Flatten
from gwu_nn.activation_layers import Sigmoid

np.random.seed(1)

X_train = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
y_train = np.array([5])
network = GWUNetwork()
network.add(Conv_2d(3, 2, activation='sigmoid'))
network.add(Flatten((2, 2)))
network.compile('mse', 0.001)
network.fit(X_train, y_train, epochs=100)