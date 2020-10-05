import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense
from gwu_nn.activation_layers import Sigmoid

np.random.seed(8)
num_obs = 8000

# Create our features to draw from two distinct 2D normal distributions
x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_obs)
x2 = np.random.multivariate_normal([3, 8], [[1, .25],[.25, 1]], num_obs)

# Stack our inputs into one feature space
X = np.vstack((x1, x2))
print(X.shape)

y = np.hstack((np.zeros(num_obs), np.ones(num_obs)))
print(y.shape)


# colors = ['red'] * num_obs + ['blue'] * num_obs
# plt.figure(figsize=(12,8))
# plt.scatter(X[:, 0], X[:, 1], c = colors, alpha = 0.5)

# Lets randomly split things into training and testing sets so we don't cheat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create our model
network = GWUNetwork()
network.add(Dense(2, 1, True, 'sigmoid'))
network.add(Sigmoid())
#network.set_loss('mse')
network.compile('log_loss', 0.001)
network.fit(X_train, y_train, epochs=100)




from scipy.special import logit

colors = ['red'] * num_obs + ['blue'] * num_obs
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)

# Range of our X values
start_x1 = -5
end_x1 = 7

weights = network.layers[0].weights.reshape(-1).tolist()
bias = network.layers[0].bias[0][0]
start_y = (bias + start_x1 * weights[0] - logit(0.5)) / - weights[1]
end_y = (bias + end_x1 * weights[0] - logit(0.5)) / -weights[1]
plt.plot([start_x1, end_x1], [start_y, end_y], color='grey')