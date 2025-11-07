from .base import BaseLayer
import numpy as np

class LinearLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self, activation="relu"):
        mean = 0
        if activation == "relu":
            std_dev = np.sqrt(2 / self.input_size)
        else:
            std_dev = np.sqrt(2 / self.input_size + self.output_size)

        self.weights = np.random.normal(mean, std_dev, (self.output_size, self.input_size))
        self.biases = np.zeros(self.output_size)


    def forward(self, input):
        self.cached_input = input
        return input @ self.weights.T + self.biases

    def backward(self, grad_output):
        self.dw = grad_output.T @ self.cached_input
        self.db = grad_output @ np.ones(self.biases.shape)
        dx = grad_output @ self.weights
        return dx if grad_output.shape[0] > 1 else dx.squeeze()

