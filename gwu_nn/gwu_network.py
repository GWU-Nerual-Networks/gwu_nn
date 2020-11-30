import numpy as np
from gwu_nn.loss_functions import MSE, LogLoss, CrossEntropy
import math

loss_functions = {'mse': MSE, 'log_loss': LogLoss, 'cross_entropy': CrossEntropy}

class GWUNetwork():

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        if len(self.layers) > 0:
            layer.init_weights(self.layers[-1].output_size)
        else:
            layer.init_weights(layer.input_size)
        self.layers.append(layer)

    def get_weights(self):
        pass

    def compile(self, loss, lr):
        layer_loss = loss_functions[loss]
        self.loss = layer_loss.loss
        self.loss_prime = layer_loss.loss_partial_derivative
        self.learning_rate = lr

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def evaluate(self, x, y):
        pass

    @staticmethod
    def __batch_random_division(x_train, y_train, batch_size, seed):
        np.random.seed(seed)
        c = 0
        m = x_train.shape[0]
        batches = []

        # Shuffle x_train & y_train
        permutation = list(np.random.permutation(m))
        randomized_X = x_train[permutation].transpose()
        randomized_Y = y_train[permutation].reshape((m, 1)).transpose()

        num_complete_batches = math.floor(m / batch_size)

        for c in range(0, num_complete_batches):
            batch_X = randomized_X[:, c * batch_size: (c + 1) * batch_size]
            batch_Y = randomized_Y[:, c * batch_size: (c + 1) * batch_size]

            batch = (batch_X, batch_Y)
            batches.append(batch)

        if m % batch_size != 0:
            mini_batch_X = randomized_X[:, (c + 1) * batch_size:]
            mini_batch_Y = randomized_Y[:, (c + 1) * batch_size:]

            batch = (mini_batch_X, mini_batch_Y)
            batches.append(batch)
        return batches

    # train the network
    def fit(self, x_train, y_train, epochs, batch_size=None):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j].reshape(1, -1)
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                y_true = np.array(y_train[j]).reshape(-1, 1)
                err += self.loss(y_true, output)

                # backward propagation
                error = self.loss_prime(y_true, output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learning_rate)

            # calculate average error on all samples
            if i % 10 == 0:
                err /= samples
                print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
                
    def __repr__(self):
        rep = "Model:"

        if len(self.layers) < 1:
            return "Model: Empty"
        else:
            rep += "\n"

        for layer in self.layers:
            if layer.type == "Activation":
                rep += f'{layer.name} Activation'
            else:
                rep += f'{layer.name} - ({layer.input_size}, {layer.output_size})\n'

        return rep
