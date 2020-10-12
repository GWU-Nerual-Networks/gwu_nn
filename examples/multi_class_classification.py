from keras.utils import np_utils
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense

# load dataset
data = load_iris()
X = data.data
Y = data.target

# Encode data
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

network = GWUNetwork()
network.add(Dense(14, add_bias=True, activation='relu', input_size=X.shape[1]))
network.add(Dense(dummy_y.shape[1], add_bias=True, activation='softmax'))
network.compile(loss='cross_entropy', lr=.001)
network.fit(X, dummy_y, batch_size=10, epochs=100)