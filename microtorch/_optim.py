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


class Adam(Optimizer):
    def __init__(self, params: list[Parameter], lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad
            self.m[i] = self.betas[0] * self.m[i] + (1.0 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1.0 - self.betas[1]) * grad ** 2
            m_hat = self.m[i] / (1.0 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1.0 - self.betas[1] ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
