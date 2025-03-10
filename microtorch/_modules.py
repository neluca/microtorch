from abc import ABC, abstractmethod
from collections.abc import Iterable
from ._autograd import Tensor, uniform, dropout, softmax, zeros, stack
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
        return list(set(params))  # 去重

    def train(self):
        for param in self.parameters():
            param.requires_grad = True

    def eval(self):
        for param in self.parameters():
            param.requires_grad = False

    @abstractmethod
    def forward(self, *tensors: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the \"forward\" method.")

    def __call__(self, *tensors: Tensor) -> Tensor:
        return self.forward(*tensors)

    def _state_dict(self) -> dict[str, Parameter]:
        def _assign(root: dict, d: dict):
            for key, value in d.items():
                root[key] = value

        def _get_params(root: Module, prefix=""):
            _state: dict[str, Parameter] = {}
            for key, value in root.__dict__.items():
                if isinstance(value, Parameter):
                    key_str = f"{prefix}.{key}" if prefix != "" else f"{key}"
                    _state[key_str] = value
                elif isinstance(value, ModuleList):
                    module_list = [
                        _get_params(m, f"{prefix}.{key}.{idx}" if prefix != "" else f"{key}.{idx}")
                        for idx, m in enumerate(value)
                    ]
                    for _module in module_list:
                        _assign(_state, _module)
                elif isinstance(value, Module):
                    _assign(_state, _get_params(value, f"{prefix}.{key}" if prefix != "" else f"{key}"))
            return _state

        return _get_params(self)

    def state_dict(self):
        _state = self._state_dict()
        for key, value in _state.items():
            _state[key] = value.clone()
        return _state

    def load_state_dict(self, state_dict: dict) -> None:
        own_state = self._state_dict()
        missing_keys = set(own_state.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
        msg = ""
        if missing_keys:
            msg += f"Missing keys in state_dict: {missing_keys}\n"
        if unexpected_keys:
            msg += f"Unexpected keys in state_dict: {unexpected_keys}\n"
        if msg:
            raise KeyError("Error(s) in loading state_dict:\n" + msg)

        for key, value in state_dict.items():
            own_state[key].data = value.data.copy()

    def register_buffer(self, name: str, value: Tensor):
        value = value.clone() if isinstance(value, Tensor) else value
        setattr(self, name, value)

    def num_parameters(self) -> int:
        return sum(param.size for param in self.parameters())

    def __repr__(self) -> str:
        s = ""
        _state = self._state_dict()
        for key, value in _state.items():
            s += key
            s += f" {value.shape}\n"
        return s


class ModuleList(list):
    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__(modules)


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, req_bias: bool = True) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim

        k = 1 / math.sqrt(in_dim)
        self.weight = Parameter(uniform(out_dim, in_dim, low=-k, high=k))
        self.bias = (
            None if not req_bias else Parameter(uniform(out_dim, low=-k, high=k))
        )

    def forward(self, x: Tensor):
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return x


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Dropout(Module):
    def __init__(self, p: float):
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return dropout(x, p=self.p)


class Softmax(Module):
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, dim=self.dim)
