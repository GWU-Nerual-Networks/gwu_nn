import numpy as np
from gwu_nn.loss_functions import MSE, LogLoss

loss_functions = {'mse': MSE, 'log_loss': LogLoss}

class GWUNetwork():

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
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

                if output < 0:
                    print("What?")
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
