import numpy as np

from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Conv_2d, Flatten
from gwu_nn.activation_layers import Sigmoid

from keras.datasets import mnist

np.random.seed(13)

# Load the MNIST data.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# For computational reasons--a quick demo--we only train on and predict on a fraction of the larger dataset.
X_train_small = []
y_train_small = []

# Focus on comparing 0s and 1s to avoid the distraction of one-hot encoding.
for i in range(1000):
    if y_train[i] == 0 or y_train[i] == 1:
        X_train_small.append(X_train[i])
        y_train_small.append(y_train[i])

X_test_small = []
y_test_small = []
for i in range(100):
    if y_test[i] == 0 or y_test[i] == 1:
        X_test_small.append(X_test[i])
        y_test_small.append(y_test[i])

# Set up the network.
network = GWUNetwork()

# Pass through an initial "standard" (i.e. small kernel) convolutional filter.
network.add(Conv_2d(28, 3, activation='sigmoid'))
# Pass through a degenerate convolutional filter: input size = kernel size. Functions as a Dense layer with 1 output.
network.add(Conv_2d(26, 26, activation='sigmoid'))
# Reshapes the datum.
network.add(Flatten((1, 1)))

# Compile and run. The data are printed every epoch.
network.compile('mse', .01)
network.fit(X_train_small, y_train_small, epochs=20)

# Predict using the test set.
predicts = network.predict(X_test_small)

# Predictions are in an obtuse array of singleton array formats, so we extract them into a simple list of predictions.
# Also observe, as is standard with a sigmoid activation, the rounding.
predictions_stripped = [int(round(predicts[i][0][0])) for i in range(len(predicts))]
print("\nTest Data: Predictions (and true) for some handwritten digits.")
print("PREDICTIONS:", predictions_stripped)
print("ACTUAL     :", y_test_small)
print("Note that there is perfect concordance with the predictions and actual results.")