from abc import ABC, abstractmethod
from ._modules import Parameter
import numpy as np


class Optimizer(ABC):
    def __init__(self, params: list[Parameter], lr: float) -> None:
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self):
        raise NotImplementedError("Subclasses must implement the \"step\" method.")

    def zero_grad(self):
        for params in self.params:
            params.grad = np.zeros(params.shape)


class SGD(Optimizer):
    def __init__(self, params: list[Parameter], lr: float) -> None:
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data -= param.grad * self.lr
