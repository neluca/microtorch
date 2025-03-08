from abc import ABC, abstractmethod
from itertools import accumulate
from typing import Any, Optional, TypeAlias

import numpy as np

__all__ = [
    "ArrayLike", "Function",
    "Add", "Sub", "Mul", "Div", "MatMul",
    "Exp", "Log", "Pow", "Sigmoid", "Tanh", "ReLU",
    "Sum", "Max", "Mean",
    "Transpose", "Reshape", "Select", "Concat", "Stack",
    "Softmax", "MSELoss", "CrossEntropyLoss", "Dropout",
]

ArrayLike: TypeAlias = np.ndarray


class Function(ABC):
    def __init__(self, op_args: Any) -> None:
        self.op_args = op_args  # graph
        self._cache: Any = None

    @property
    def name(self) -> str:  # graph
        return self.__class__.__name__

    def save_to_cache(self, *args: Any):
        self._cache = args

    def retrieve_from_cache(self) -> tuple[Any, ...]:
        assert self._cache is not None
        values, self._cache = self._cache, None
        return values

    @abstractmethod
    def forward(self, *arrays: Optional[ArrayLike], **kwargs: Any) -> ArrayLike:
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def backward(self, dy: ArrayLike) -> tuple[Any, ...]:
        raise NotImplementedError("Subclasses must implement the backward method.")


# binary ops ---------------------------------------------------------------------------
class Add(Function):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 + x2
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dx1 = dy
        dx2 = dy
        return tuple((dx1, dx2))


class Sub(Function):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 - x2
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dx1 = dy
        dx2 = -dy
        return tuple((dx1, dx2))


class Mul(Function):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 * x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy * x2
        dx2 = dy * x1
        return tuple((dx1, dx2))


class Div(Function):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 / x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy / x2
        dx2 = -(dy * x1) / (x2 * x2)
        return tuple((dx1, dx2))


class MatMul(Function):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 @ x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy @ x2.swapaxes(-1, -2)  # dy @ x2.T
        dx2 = x1.swapaxes(-1, -2) @ dy  # x1.T @ dy
        return tuple((dx1, dx2))


# unary ops ---------------------------------------------------------------------------
class Exp(Function):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.exp(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * y
        return tuple((dx,))


class Log(Function):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.log(x)
        self.save_to_cache(x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (x,) = self.retrieve_from_cache()
        dx = dy / x
        return tuple((dx,))


class Pow(Function):
    def forward(self, x: ArrayLike, *, power: int | float) -> ArrayLike:
        y = x ** power
        self.save_to_cache(x, power)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, power = self.retrieve_from_cache()
        dx = dy * power * x ** (power - 1)
        return tuple((dx,))


class Sigmoid(Function):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = 1 / (1 + np.exp(-x))
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * y * (1 - y)
        return tuple((dx,))


class Tanh(Function):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.tanh(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * (1 - y * y)
        return tuple((dx,))


class ReLU(Function):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.maximum(x, 0.0)
        self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (mask,) = self.retrieve_from_cache()
        dx = dy * mask
        return tuple((dx,))


# reduce ops ---------------------------------------------------------------------------
class Sum(Function):
    def forward(self, x: ArrayLike, *, dim: Optional[int | tuple[int, ...]], keepdims: bool) -> ArrayLike:
        y = x.sum(dim, keepdims=keepdims)
        self.save_to_cache(x.shape, dim, keepdims)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape, dim, keepdims = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = np.expand_dims(dy, dim)
        dx = np.broadcast_to(dy, x_shape)
        return tuple((dx,))


class Max(Function):
    def forward(self, x: ArrayLike, *, dim: Optional[int], keepdims: bool) -> ArrayLike:
        y = x.max(dim, keepdims=True)
        self.save_to_cache(dim, keepdims, x == y)
        return y if keepdims else y.squeeze()

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, keepdims, mask = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = np.expand_dims(dy, dim)
        dx = mask * dy / mask.sum(dim, dtype=dy.dtype, keepdims=True)
        return tuple((dx,))


class Mean(Function):
    def forward(self, x: ArrayLike, *, dim: Optional[int | tuple[int, ...]], keepdims: bool) -> ArrayLike:
        y = x.mean(dim, keepdims=keepdims)
        self.save_to_cache(x.shape, dim, keepdims, x.size / y.size)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape, dim, keepdims, size = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = np.expand_dims(dy, dim)
        dx = np.broadcast_to(dy / size, x_shape)
        return tuple((dx,))


# movement ops ---------------------------------------------------------------------------
class Transpose(Function):
    def forward(self, x: ArrayLike, *, dim1: int, dim2: int) -> ArrayLike:
        y = x.swapaxes(dim1, dim2)
        self.save_to_cache(dim1, dim2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim1, dim2 = self.retrieve_from_cache()
        dx = dy.swapaxes(dim1, dim2)
        return tuple((dx,))


class Reshape(Function):
    def forward(self, x: ArrayLike, shape: tuple[int, ...]) -> ArrayLike:
        self.save_to_cache(x.shape)
        y = np.reshape(x, shape)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (shape,) = self.retrieve_from_cache()
        dx = np.reshape(dy, shape)
        return tuple((dx,))


class Select(Function):
    def forward(self, x: ArrayLike, *, key: Any) -> ArrayLike:
        y = x[key]
        self.save_to_cache(x.shape, key)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape, key = self.retrieve_from_cache()
        dx = np.zeros(x_shape, dtype=dy.dtype)
        np.add.at(dx, key, dy)
        return tuple((dx,))


class Concat(Function):
    def forward(self, *arrays: ArrayLike, dim: int) -> ArrayLike:
        y = np.concatenate(arrays, dim)
        self.save_to_cache(dim, [a.shape[dim] for a in arrays])
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, split_sizes = self.retrieve_from_cache()
        split_indices = list(accumulate(s for s in split_sizes))
        dxs = np.split(dy, split_indices, dim)
        return tuple(dxs)


class Stack(Function):
    def forward(self, *arrays: ArrayLike | bool, dim: int) -> ArrayLike:
        y = np.stack(arrays, dim)
        self.save_to_cache(dim)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (dim,) = self.retrieve_from_cache()
        dxs = tuple(np.moveaxis(dy, dim, 0))
        return tuple(dxs)


# nn ops ---------------------------------------------------------------------------
def _softmax_fwd(x: ArrayLike, dim: int) -> ArrayLike:
    x = np.exp(x - x.max(dim, keepdims=True))
    return x / x.sum(dim, keepdims=True)


def _softmax_bwd(y: ArrayLike, dy: ArrayLike, dim: int) -> ArrayLike:
    return y * (dy - (dy * y).sum(dim, keepdims=True))


class Softmax(Function):
    def forward(self, x: ArrayLike, *, dim: int) -> ArrayLike:
        y = _softmax_fwd(x, dim)
        self.save_to_cache(dim, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, y = self.retrieve_from_cache()
        dx = _softmax_bwd(y, dy, dim)
        return tuple((dx,))


class MSELoss(Function):
    def forward(self, x: ArrayLike, y: ArrayLike, *, reduction: str) -> ArrayLike:
        diff = x - y
        loss = diff * diff
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.save_to_cache(x.size, diff, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, None]:
        x_size, diff, reduction = self.retrieve_from_cache()
        dx = dy * 2.0 * diff
        if reduction == "mean":
            dx /= float(x_size)
        return tuple((dx, None))


def _onehot(x: ArrayLike, n: int, dtype: type):
    return np.eye(n, dtype=dtype)[x]


class CrossEntropyLoss(Function):
    def forward(
            self, x: ArrayLike, y: ArrayLike, *, eta: float, reduction: str
    ) -> ArrayLike:
        probs = _softmax_fwd(x, dim=-1)
        y_onehot = _onehot(y, x.shape[-1], probs.dtype)
        loss = -(np.log(probs + eta) * y_onehot).sum(-1)
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.save_to_cache(y, probs, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, None]:
        y, probs, reduction = self.retrieve_from_cache()
        y = _onehot(y, probs.shape[-1], probs.dtype)
        dx = dy * (probs - y)
        if reduction == "mean":
            dx /= np.prod(y.shape[:-1], dtype=dx.dtype)
        return tuple((dx, None))


class Dropout(Function):
    def forward(self, x: ArrayLike, *, p: float) -> ArrayLike:
        dropout_mask = np.random.rand(x.shape) > p
        y = x * dropout_mask / (1 - p)
        self.save_to_cache(p, dropout_mask)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        p, dropout_mask = self.retrieve_from_cache()
        dx = dy * dropout_mask / (1 - p)
        return tuple((dx,))
