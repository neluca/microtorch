from abc import ABC, abstractmethod
from collections.abc import Iterable
from ._autograd import Tensor, randn, uniform
import math


class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data, requires_grad=True)


class Module(ABC):
    def parameters(self) -> list[Parameter]:
        params = []
        for _, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params.append(value)
            if isinstance(value, Module):
                params.extend(value.parameters())
            if isinstance(value, ModuleList):
                for module in value:
                    params.extend(module.parameters())
        return list(set(params))

    def train(self):
        for p in self.parameters():
            p.requires_grad = True

    def eval(self):
        for p in self.parameters():
            p.requires_grad = False

    @abstractmethod
    def forward(self, *tensors: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the \"forward\" method.")

    def __call__(self, *tensors: Tensor) -> Tensor:
        return self.forward(*tensors)


class ModuleList(list):
    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__(modules)


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        k = 1 / math.sqrt(in_dim)
        self.w = Parameter(uniform(out_dim, in_dim, low=-k, high=k))
        self.b = (
            None if not bias else Parameter(uniform(out_dim, low=-k, high=k))
        )

    def forward(self, x: Tensor):
        x = x @ self.w.T
        if self.bias:
            x = x + self.b
        return x
