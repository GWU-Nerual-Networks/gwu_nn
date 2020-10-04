from gwu_nn.activation_functions import SigmoidActivation, RELUActivation


class ActivationLayer:
    def __init__(self, activation):
        self.type = "Activation"
        layer_activation = activation
        self.activation = layer_activation.activation
        self.activation_prime = layer_activation.activation_partial_derivative

    def forward_propagation(self, input):
        self.input = input
        return self.activation(input)

    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__(SigmoidActivation)
        self.name = "Sigmoid"

class RELU(ActivationLayer):
    def __init__(self):
        super().__init__(RELUActivation)
        self.name = "RELU"

