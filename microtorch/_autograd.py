from __future__ import annotations

from typing import Optional, Any, Literal
import numpy as np
from ._ops import *

_autograd_tracking_active: bool = True


def no_grad():
    class _Context:
        def __call__(self, func):
            def wrapper(*args, **kwargs):
                with _Context():
                    return func(*args, **kwargs)

            return wrapper

        def __enter__(self):
            global _autograd_tracking_active
            _autograd_tracking_active = False

        def __exit__(self, exc_type, exc_val, exc_tb):
            global _autograd_tracking_active
            _autograd_tracking_active = True

    return _Context()


class Tensor:
    def __init__(
            self,
            data: ArrayLike,
            op: Optional[Op] = None,
            src: Optional[tuple[Optional[Tensor], ...]] = None,
            requires_grad: bool = False
    ) -> None:
        self.data = data if isinstance(data, ArrayLike) else np.asarray(data)
        self.op = op
        self.src = src
        self.requires_grad = requires_grad
        self.grad: Optional[ArrayLike] = None

    @property
    def name(self) -> str:
        if self.op is not None:
            return self.op.name
        return self.__class__.__name__

    @property
    def T(self) -> Tensor:
        return _apply(Transpose, self, dim1=-2, dim2=-1)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        return f"tensor({self.data})"

    def __add__(self, x: Any) -> Tensor:
        return _apply(Add, self, _align(x))

    __radd__ = __add__

    def __sub__(self, x: Any) -> Tensor:
        return _apply(Sub, self, _align(x))

    def __rsub__(self, x: Any) -> Tensor:
        return _apply(Sub, _align(x), self)

    def __neg__(self) -> Tensor:
        return -1 * self

    def __mul__(self, x: Any) -> Tensor:
        return _apply(Mul, self, _align(x))

    __rmul__ = __mul__

    def __truediv__(self, x: Any) -> Tensor:
        return _apply(Div, self, _align(x))

    def __rtruediv__(self, x: Any) -> Tensor:
        return _apply(Div, _align(x), self)

    def __matmul__(self, x: Tensor) -> Tensor:
        return _apply(MatMul, self, x)

    def __pow__(self, power: int | float) -> Tensor:
        return _apply(Pow, self, power=power)

    def __getitem__(self, item: Any) -> Tensor:
        return _apply(Select, self, key=_parse_key(item))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, Tensor):
            value = value.data
        self.data[_parse_key(key)] = value

    def split(self, split_size: int, *, dim: int = -1) -> list[Tensor]:
        dim = dim % self.ndim
        pre_dim_slice = (slice(None),) * dim
        post_dim_slice = (slice(None),) * (self.ndim - dim - 1)
        return [
            self[pre_dim_slice + (slice(i, i + split_size),) + post_dim_slice]
            for i in range(0, self.shape[dim], split_size)
        ]

    def exp(self) -> Tensor:
        return _apply(Exp, self)

    def log(self) -> Tensor:
        return _apply(Log, self)

    def sigmoid(self) -> Tensor:
        return _apply(Sigmoid, self)

    def tanh(self) -> Tensor:
        return _apply(Tanh, self)

    def relu(self) -> Tensor:
        return _apply(ReLU, self)

    def sum(self, dim: Optional[int | tuple[int, ...]] = None, *, keepdims: bool = False) -> Tensor:
        return _apply(Sum, self, dim=dim, keepdims=keepdims)

    def max(self, dim: Optional[int | tuple[int, ...]] = None, *, keepdims: bool = False) -> Tensor:
        return _apply(Max, self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[int | tuple[int, ...]] = None, *, keepdims: bool = False) -> Tensor:
        return _apply(Mean, self, dim=dim, keepdims=keepdims)

    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        return _apply(Reshape, self, shape=shape)

    def accumulate_grad(self, dy: ArrayLike) -> None:
        self.grad = dy if self.grad is None else self.grad + dy

    def backward(self, dy: Optional[ArrayLike] = None):
        assert self.requires_grad, "Node is not part of a autograd graph."
        assert self.grad is None, "Cannot run backward multiple times."

        if dy is None:
            self.grad = np.ones(self.data.shape, dtype=np.float32)
        else:
            assert isinstance(dy, ArrayLike), "Gradient must be an array."
            self.grad = dy

        # run backward through traced graph
        node_queue = _computed_node_dfs(self, [], set())
        for node in reversed(node_queue):
            assert node.op is not None, "Node has no function context."
            assert node.src is not None, "Node has no source nodes."
            assert node.grad is not None, "Node has no grad that is constant."
            grads = node.op.backward(node.grad)
            for src_tensor, grad in zip(node.src, grads):
                if src_tensor is None or not src_tensor.requires_grad:
                    continue
                grad = _undo_broadcast(grad, src_tensor.data.shape)
                src_tensor.accumulate_grad(grad)

            # clear context of intermediate nodes
            node.grad, node.op, node.src = None, None, None

    def __hash__(self) -> int:
        return id(self)

    def __len__(self) -> int:
        return len(self.data)

    def detach(self) -> ArrayLike:
        return self.data

    def t(self) -> Tensor:
        return self.T

    def item(self) -> Any:
        return self.data.item()

    def float(self) -> Tensor:
        self.data = self.data.astype(np.float32)
        return self

    def long(self):
        self.data = self.data.astype(np.int16)
        return self


def _align(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=np.float32))


def _apply(op_: type(Op), *tensors: Optional[Tensor], **kwargs: Any) -> Tensor:
    tensor_args = [t for t in tensors if t is not None]
    op = op_(kwargs)

    fwd_args = [t.data if t is not None else None for t in tensors]
    data = op.forward(*fwd_args, **kwargs)

    result_req_grad = any(t.requires_grad for t in tensor_args)
    if _autograd_tracking_active and result_req_grad:
        return Tensor(data, op=op, src=tensors, requires_grad=True)

    return Tensor(data)


def _computed_node_dfs(node: Tensor, queue: list[Tensor], visited: set) -> list[Tensor]:
    if node not in visited:
        visited.add(node)
        if not node.src:
            return []
        for p in node.src:
            if p is not None and p.requires_grad:
                _ = _computed_node_dfs(p, queue, visited)
        queue.append(node)
    return queue


def _get_shape_diff(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(i for i in range(len(shape1)) if shape1[i] != shape2[i])


def _undo_broadcast(grad: ArrayLike, target_shape: tuple[int, ...]) -> ArrayLike:
    if grad.shape == target_shape:
        return grad
    target_ndim = len(target_shape)

    if grad.ndim == target_ndim:
        shape = _get_shape_diff(grad.shape, target_shape)
        grad = grad.sum(shape, keepdims=True)
    else:
        data_shape = tuple((1,) * (grad.ndim - target_ndim) + target_shape)
        shape = _get_shape_diff(grad.shape, data_shape)
        grad = grad.sum(shape)

    return grad.reshape(target_shape)


def _parse_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return tuple(k.data if isinstance(k, Tensor) else k for k in key)
    if isinstance(key, Tensor):
        return key.data
    return key


def tensor(data: Any, requires_grad=False) -> Tensor:
    if requires_grad:
        if isinstance(data, np.ndarray):
            return Tensor(data.copy(), requires_grad=True)
    if isinstance(data, np.ndarray):
        return Tensor(data, requires_grad=requires_grad)
    elif isinstance(data, Tensor):
        data.requires_grad = requires_grad
        return data
    return Tensor(np.asarray(data), requires_grad=requires_grad)


def uniform(*shape: int, low: float = -1, high: float = 1, requires_grad: bool = False) -> Tensor:
    data = np.random.uniform(low, high, shape)
    return Tensor(data, requires_grad=requires_grad)


def randn(*shape: int, mean: float = 0, std: float = 1, requires_grad: bool = False) -> Tensor:
    data = np.random.normal(mean, std, shape)
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
    data = np.zeros(shape, dtype=np.float32)
    return Tensor(data, requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False) -> Tensor:
    data = np.ones(shape, dtype=np.float32)
    return Tensor(data, requires_grad=requires_grad)


def argmax(_tensor: Tensor, axis=None) -> Tensor:
    return Tensor(np.array(np.argmax(_tensor.data, axis=axis)))


def arange(*args: int | float, requires_grad=False) -> Tensor:
    return Tensor(np.arange(*args), requires_grad=requires_grad)


def concat(tensors: list[Tensor], dim: int = 0):
    return _apply(Concat, *tensors, dim=dim)


def stack(tensors: list[Tensor], dim: int = 0):
    return _apply(Stack, *tensors, dim=dim)


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def relu(x: Tensor) -> Tensor:
    return x.relu()


def softmax(x: Tensor, *, dim: int = -1) -> Tensor:
    return _apply(Softmax, x, dim=dim)


def dropout(x: Tensor, *, p: float = 0.9) -> Tensor:
    return _apply(Dropout, x, p=p)


def mse_loss(
        logits: Tensor, targets: Tensor, reduction: Literal["sum", "mean"] = "mean"
) -> Tensor:
    assert not targets.requires_grad, "Targets cannot require gradients."
    return _apply(MSELoss, logits, targets, reduction=reduction)


def cross_entropy(
        logits: Tensor,
        targets: Tensor,
        eta: float = 1e-8,
        reduction: Literal["sum", "mean"] = "mean",
) -> Tensor:
    assert not targets.requires_grad, "Targets cannot require gradients."
    return _apply(CrossEntropyLoss, logits, targets, eta=eta, reduction=reduction)
