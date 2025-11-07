from abc import ABC, abstractmethod

class BaseLayer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass

