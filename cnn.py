from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Conv2D, Flatten, MaxPooling2D
from gwu_nn.activation_layers import Softmax
import pandas as pd
import gzip
from keras.datasets import mnist
#For now I will be using keras only for loading mnist data because it is cleaner. I can later change this to load mnist from gzip data
#without using keras and using my own functions


def cnn_model(input_shape, num_classes):

    """
    Args:
        input_shape (int): input shape (shape of image)
        num_classes (int): number of labels in the output
    Returns:
        GWU Network model with CNN functionality
    """
    model = GWUNetwork()

    model.add(Conv2D(kernel_size= 3,activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size= 4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss = 'cross_entropy', lr = 0.001)

    return model

def read_data(file_name): #used for loading mnist data in csv format
    result = pd.read_csv(file_name)
    data_no_label = result.loc[:, 'pixel0':'pixel783'].copy()
    labels = result.loc[:, 'label'].copy()
    return data_no_label, labels

def load_gzip_data(file_name): #used for loading mnist data in gzip format
    # with gzip.open(file_name) as f:
    pass


(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = len(set(y_train))
model = cnn_model(X_train[0].shape, num_classes)
model.fit(X_train, y_train, epochs=3, batch_size=100)
# model.evaluate(X_test, y_test)
