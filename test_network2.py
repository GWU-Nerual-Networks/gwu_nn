import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Conv_2d, Flatten, Max_Pool
from gwu_nn.activation_layers import Sigmoid

np.random.seed(1)

print("EXHIBITING CONV_2D (AND FLATTEN) LAYERS")
X_train = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
y_train = np.array([5])
network = GWUNetwork()
network.add(Conv_2d(3, 2, activation='sigmoid'))
network.add(Flatten((2, 2)))
network.compile('mse', 0.001)
network.fit(X_train, y_train, epochs=100)

print("EXHIBITING MAXPOOL (AND FLATTEN) LAYERS")
X_train2 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
y_train2 = np.array([9])
network2 = GWUNetwork()
network2.add(Max_Pool(kernel_size=2, stride=2, input_size=4))
network2.add(Flatten((2, 2)))
network2.add(Dense(1, input_size=4, add_bias=False, activation='sigmoid'))
network2.compile('mse', 0.01)
network2.fit(X_train2, y_train2, epochs=11)
