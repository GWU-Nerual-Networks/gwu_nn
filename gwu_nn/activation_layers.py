from gwu_nn.activation_functions import SigmoidActivation, RELUActivation, SoftmaxActivation


class ActivationLayer:
    def __init__(self, activation):
        self.type = "Activation"
        layer_activation = activation
        self.activation = layer_activation.activation
        self.activation_prime = layer_activation.activation_partial_derivative

    def init_weights(self, input_size):
        pass

    def forward_propagation(self, input):
        """Applies the classes activation function to the provided input

        Args:
            input (np.array): output calculated forward up to this layer

        Returns:
            np.array(float): forward pass (output) up to this layer
        """
        self.input = input
        return self.activation(input)

    def backward_propagation(self, output_error, learning_rate):
        """Applies the classes activation function to the provided input

        Args:
            output_error (np.array): output_error calculated backwards to this layer

        Returns:
            np.array(float): backwards pass (output_error) up to this layer
        """
        return output_error * self.activation_prime(self.input)


class Sigmoid(ActivationLayer):
    """Layer that applies the Sigmoid activation function"""
    def __init__(self):
        super().__init__(SigmoidActivation)
        self.name = "Sigmoid"

class RELU(ActivationLayer):
    """Layer that applies the ReLU activation function"""
    def __init__(self):
        super().__init__(RELUActivation)
        self.name = "RELU"

class Softmax(ActivationLayer):
    """Layer that applies the Softmax activation function"""
    def __init__(self):
        super().__init__(SoftmaxActivation)
        self.name = "Softmax"