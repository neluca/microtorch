from abc import ABC, abstractmethod
from typing import Any, Optional, TypeAlias
from itertools import accumulate
import numpy as np

ArrayLike: TypeAlias = np.ndarray


class Op(ABC):
    def __init__(self, op_args: Any) -> None:
        self.op_args = op_args  # graph
        self._cache: Any = None

    @property
    def name(self) -> str:
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
class Add(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 + x2
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dx1 = dy
        dx2 = dy
        return tuple((dx1, dx2))


class Sub(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 - x2
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dx1 = dy
        dx2 = -dy
        return tuple((dx1, dx2))


class Mul(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 * x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy * x2
        dx2 = dy * x1
        return tuple((dx1, dx2))


class Div(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 / x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy / x2
        dx2 = -(dy * x1) / (x2 * x2)
        return tuple((dx1, dx2))


class MatMul(Op):
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
class Exp(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.exp(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * y
        return tuple((dx,))


class Log(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.log(x)
        self.save_to_cache(x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (x,) = self.retrieve_from_cache()
        dx = dy / x
        return tuple((dx,))


class Pow(Op):
    def forward(self, x: ArrayLike, *, power: int | float) -> ArrayLike:
        y = x ** power
        self.save_to_cache(x, power)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, power = self.retrieve_from_cache()
        dx = dy * power * x ** (power - 1)
        return tuple((dx,))


class Tanh(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.tanh(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * (1 - y * y)
        return tuple((dx,))


class Sqrt(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.sqrt(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * 0.5 / y
        return tuple((dx,))


class Sin(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.sin(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = np.cos(y)
        return tuple((dx,))


class Cos(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.cos(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = -np.sin(y)
        return tuple((dx,))


# reduce ops ---------------------------------------------------------------------------
class Sum(Op):
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


class Max(Op):
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


# other ops ---------------------------------------------------------------------------
class Reshape(Op):
    def forward(self, x: ArrayLike, shape: tuple[int, ...]) -> ArrayLike:
        self.save_to_cache(x.shape)
        y = np.reshape(x, shape)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (shape,) = self.retrieve_from_cache()
        dx = np.reshape(dy, shape)
        return tuple((dx,))


class Transpose(Op):
    def forward(self, x: ArrayLike, *, dim1: int, dim2: int) -> ArrayLike:
        y = x.swapaxes(dim1, dim2)
        self.save_to_cache(dim1, dim2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim1, dim2 = self.retrieve_from_cache()
        dx = dy.swapaxes(dim1, dim2)
        return tuple((dx,))


class Select(Op):
    def forward(self, x: ArrayLike, *, key: Any) -> ArrayLike:
        y = x[key]
        self.save_to_cache(x.shape, key)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape, key = self.retrieve_from_cache()
        dx = np.zeros(x_shape, dtype=dy.dtype)
        np.add.at(dx, key, dy)
        return tuple((dx,))


class Concat(Op):
    def forward(self, *arrays: ArrayLike, dim: int) -> ArrayLike:
        y = np.concatenate(arrays, dim)
        self.save_to_cache(dim, [a.shape[dim] for a in arrays])
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, split_sizes = self.retrieve_from_cache()
        split_indices = list(accumulate(s for s in split_sizes))
        dxs = np.split(dy, split_indices, dim)
        return tuple(dxs)


class View(Op):
    def forward(self, x: ArrayLike, *, shape: tuple[int, ...]) -> ArrayLike:
        y = np.reshape(x, shape)
        self.save_to_cache(x.shape)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (x_shape,) = self.retrieve_from_cache()
        dx = dy.reshape(x_shape)
        return tuple((dx,))


class Stack(Op):
    def forward(self, *arrays: ArrayLike | bool, dim: int) -> ArrayLike:
        y = np.stack(arrays, dim)
        self.save_to_cache(dim)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (dim,) = self.retrieve_from_cache()
        dxs = tuple(np.moveaxis(dy, dim, 0))
        return tuple(dxs)


class Where(Op):
    def forward(self, condition: ArrayLike, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = np.where(condition, x1, x2)
        self.save_to_cache(y == x1)
        return y

    def backward(self, dy: ArrayLike) -> tuple[None, ArrayLike, ...]:
        (mask,) = self.retrieve_from_cache()
        dx1 = dy * mask
        dx2 = dy * np.invert(mask)
        return tuple((None, dx1, dx2))


# nn ops ---------------------------------------------------------------------------
def _sigmoid_fwd(x: ArrayLike) -> ArrayLike:
    return 1 / (1 + np.exp(-x))


class Sigmoid(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = _sigmoid_fwd(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * y * (1 - y)
        return tuple((dx,))


class ReLU(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = np.maximum(x, 0.0)
        self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (mask,) = self.retrieve_from_cache()
        dx = dy * mask
        return tuple((dx,))


class GELU(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = 0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
        self.save_to_cache(x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (x,) = self.retrieve_from_cache()
        t = np.tanh(x * 0.79788 * (1 + 0.04472 * x * x))
        dx = dy * 0.5 * ((1 + t) + x * (1 - t * t) * (0.79788 + 0.10703 * x * x))
        return tuple((dx,))


def _softmax_fwd(x: ArrayLike, dim: int) -> ArrayLike:
    x = np.exp(x - x.max(dim, keepdims=True))
    return x / x.sum(dim, keepdims=True)


def _softmax_bwd(y: ArrayLike, dy: ArrayLike, dim: int) -> ArrayLike:
    return y * (dy - (dy * y).sum(dim, keepdims=True))


class Softmax(Op):
    def forward(self, x: ArrayLike, *, dim: int) -> ArrayLike:
        y = _softmax_fwd(x, dim)
        self.save_to_cache(dim, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, y = self.retrieve_from_cache()
        dx = _softmax_bwd(y, dy, dim)
        return tuple((dx,))


class Linear(Op):
    def forward(self, x: ArrayLike, w: ArrayLike, b: Optional[ArrayLike]) -> ArrayLike:
        y = x @ w.swapaxes(-1, -2)
        y = y if b is None else y + b
        self.save_to_cache(x, w, b is not None)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, b_requires_grad = self.retrieve_from_cache()
        dx = dy @ w
        dw = dy.swapaxes(-1, -2) @ x  # dy.T @ x
        db = None if not b_requires_grad else dy
        return tuple((dx, dw, db))


def _dropout_mask(shape: tuple[int, ...], p: float) -> ArrayLike:
    return np.random.rand(*shape) > p


def _dropout_fwd(x: ArrayLike, dropout_mask: ArrayLike, p: float) -> ArrayLike:
    return x * dropout_mask / (1 - p)


def _dropout_bwd(dy: ArrayLike, dropout_mask: ArrayLike, p: float) -> ArrayLike:
    return dy * dropout_mask / (1 - p)


class Dropout(Op):
    def forward(self, x: ArrayLike, *, p: float) -> ArrayLike:
        dropout_mask = _dropout_mask(x.shape, p)
        y = _dropout_fwd(x, dropout_mask, p)
        self.save_to_cache(p, dropout_mask)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        p, dropout_mask = self.retrieve_from_cache()
        dx = _dropout_bwd(dy, dropout_mask, p)
        return tuple((dx,))


class LayerNorm(Op):
    def forward(
            self, x: ArrayLike, w: ArrayLike, b: ArrayLike, *, eps: float
    ) -> ArrayLike:
        f_dims = tuple(range(x.ndim - w.ndim, x.ndim))

        mean = x.mean(f_dims, keepdims=True)
        x_shift = x - mean
        var = (x_shift * x_shift).mean(f_dims, keepdims=True)
        r_std = 1 / np.sqrt(var + eps)
        x_norm = x_shift * r_std
        y = x_norm * w + b

        self.save_to_cache(w, f_dims, r_std, x_norm)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        w, f_dims, r_std, xnorm = self.retrieve_from_cache()
        b_dims = tuple(range(dy.ndim - w.ndim))
        db = dy.sum(b_dims)
        dw = (dy * xnorm).sum(b_dims)
        dx_norm = dy * w
        dx = r_std * (
                dx_norm
                - dx_norm.mean(f_dims, keepdims=True)
                - xnorm * (dx_norm * xnorm).mean(f_dims, keepdims=True)
        )

        return tuple((dx, dw, db))


class Embedding(Select):
    pass


# loss ops ---------------------------------------------------------------------------
class MSELoss(Op):
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


class CrossEntropyLoss(Op):
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


class BCELoss(Op):
    def forward(self, x: ArrayLike, y: ArrayLike, *, reduction: str) -> ArrayLike:
        max_logits = np.maximum(x, 0.0)
        loss = max_logits - x * y + np.log(1 + np.exp(-np.abs(x)))
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.save_to_cache(x, y, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, None]:
        x, y, reduction = self.retrieve_from_cache()
        dx = dy * (_sigmoid_fwd(x) - y)
        if reduction == "mean":
            dx /= float(x.size)
        return tuple((dx, None))
