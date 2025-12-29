import numpy as np


class Sigmoid:
    def forward(self, input):
        self.cached_input = input
        return 1 / (1 + np.exp(-input))

    def backward(self, upstream_gradient):
        return upstream_gradient * (1 / (1 + np.exp(-self.cached_input))) * (1 - (1 / (1 + np.exp(-self.cached_input))))

    def parameters(self):
        return []

    def gradients(self):
        return []
