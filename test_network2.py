import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Conv_2d, Flatten, Max_Pool
from gwu_nn.activation_layers import Sigmoid

from keras.datasets import mnist
import matplotlib.pyplot as plt


np.random.seed(3)


(X_train, y_train), (X_test, y_test) = mnist.load_data()

#print(X_train[0])

X_train_small = []
y_train_small = []

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

print(len(X_train_small), len(y_train_small))
print(len(X_test_small), len(y_test_small))
#print(X_train_small[2])
#print(y_train_small[0:10])
plt.imsave('pre_filter.jpg', X_train_small[0], cmap='gray')
'''
plt.imsave('after_filter' + strinc + '.jpg', output, cmap='gray')

# SANITY CHECK
network = GWUNetwork()
network.add(Conv_2d(28, 3, activation='sigmoid'))
network.add(Conv_2d(26, 26, activation='sigmoid')) # works best when first has sigmoid and this has none
network.add(Flatten((1, 1)))
#network.add(Dense(1, input_size=24**2, add_bias=False, activation='relu'))
# dont i have to pop it into a single thing (i.e. use Dense layer?)... that's why the predictions were so messed up the first time?
network.compile('mse', .01)
network.fit(X_train_small, y_train_small, epochs=20)

predicts = network.predict(X_test_small)

#print(predicts[1], predicts[1][0][0])

predictions_stripped = [int(round(predicts[i][0][0])) for i in range(len(predicts))]
print("PREDICTIONS:", predictions_stripped)
print("ACTUAL     :", y_test_small)


'''




'''
network = GWUNetwork()
network.add(Conv_2d(28, 3, activation=None))
network.add(Max_Pool(kernel_size=2, stride=2, input_size=26))
network.add(Flatten((26, 26)))
network.compile('mse', .000001)
network.fit(X_train_small, y_train_small, epochs=1001)
'''


'''
X_train = np.array([[1, 2, 3, 4, 5]])
y_train = np.array([3])
network = GWUNetwork()
network.add(Dense(1, input_size=5, add_bias=False, activation='sigmoid'))
network.compile('mse', .001)
network.fit(X_train, y_train, epochs=1)
'''

'''
print("EXHIBITING CONV_2D (AND FLATTEN) LAYERS")
X_train = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
y_train = np.array([5])
network = GWUNetwork()
network.add(Conv_2d(3, 2, activation=None))
network.add(Conv_2d(2, 2, activation=None))
network.add(Flatten((1, 1)))
network.compile('mse', 0.001)
network.fit(X_train, y_train, epochs=11)
'''



print("EXHIBITING MAXPOOL (AND FLATTEN) LAYERS")
X_train2 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
y_train2 = np.array([9])
network2 = GWUNetwork()
network2.add(Max_Pool(kernel_size=2, stride=2, input_size=4))
network2.add(Flatten((2, 2)))
network2.add(Dense(1, input_size=4, add_bias=False, activation='sigmoid'))
network2.compile('mse', 0.01)
network2.fit(X_train2, y_train2, epochs=11)

