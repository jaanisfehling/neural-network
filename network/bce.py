import numpy as np


class BCELoss:
    def forward(self, predicted, truth):
        self.predicted = predicted
        self.truth = truth
        assert predicted.shape == truth.shape

        # Fully reduced loss, meaning average of batch aswell
        return -np.mean(truth * np.log(predicted + 1e-15) + (1 - truth) * np.log(1 - predicted + 1e-15))

    def backward(self):
        # Fully reduced loss, so divide by batch size + elements
        return (
            -(self.truth / (self.predicted + 1e-15) - (1 - self.truth) / (1 - self.predicted + 1e-15))
            / self.predicted.size
        )
