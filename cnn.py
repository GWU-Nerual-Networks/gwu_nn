from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
import gwu_nn.utils as utils

def cnn_model(input_shape, num_final_neurons):

    """
    Args:
        input_shape (int tuple): input shape (shape of image)
        num_final_neuron (int): number of neurons in the last layer
    Returns:
        GWU Network model with CNN functionality
    """
    model = GWUNetwork()

    model.add(Conv2D(kernel_size= 3,activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size= 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_final_neurons, activation='sigmoid'))

    model.compile(loss = 'log_loss', lr = 0.001)

    return model

(X_train, y_train), (X_test, y_test) = mnist.load_data()
digits = set(y_train)

#transofrming to a binary classification problem
x_train_binary, y_train_binary = utils.transofrm_to_binary(3, 7, X_train, y_train)
x_test_binary, y_test_binary = utils.transofrm_to_binary(3, 7, X_test, y_test)

model = cnn_model(x_train_binary[0].shape, 1)
model.fit(x_train_binary[:4000], y_train_binary[:4000], epochs= 1, batch_size=100)
print("done training!")
model.evaluate(x_test_binary[:400], y_test_binary[:400])
