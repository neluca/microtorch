from __future__ import annotations

from .ops import *

_autograd_tracking_active: bool = True


class no_grad:
    def __enter__(self):
        global _autograd_tracking_active
        _autograd_tracking_active = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _autograd_tracking_active
        _autograd_tracking_active = True


class Tensor:
    def __init__(
            self,
            data: ArrayLike,
            op: Optional[Op] = None,
            src: Optional[tuple[Optional[Tensor], ...]] = None,
            requires_grad: bool = False,
            label: Optional[str] = None,
    ) -> None:
        self.data = data
        self.ctx = op
        self.src = src
        self.req_grad = requires_grad
        self._label = label
        self.grad: Optional[ArrayLike] = None

    @property
    def label(self) -> str:
        if self._label:
            return self._label
        if self.ctx is not None:
            return self.ctx.name
        return self.__class__.__name__

    @staticmethod
    def ensure_tensor(data: Any):
        if isinstance(data, Tensor):
            return data
        return Tensor(np.asarray(data))

    def __add__(self, x: Any) -> Tensor:
        return _apply(Add, self, self.ensure_tensor(x))

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, x: Any) -> Tensor:
        return _apply(Sub, self, self.ensure_tensor(x))

    __isub__ = __sub__

    def __rsub__(self, x: Any) -> Tensor:
        return _apply(Sub, self.ensure_tensor(x), self)

    def __mul__(self, x: Any) -> Tensor:
        return _apply(Mul, self, self.ensure_tensor(x))

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, x: Any) -> Tensor:
        return _apply(Div, self, self.ensure_tensor(x))

    def __rtruediv__(self, x: Any) -> Tensor:
        return _apply(Div, self.ensure_tensor(x), self)

    __itruediv__ = __rtruediv__

    def __matmul__(self, x: Tensor) -> Tensor:
        return _apply(MatMul, self, x)

    def __neg__(self) -> Tensor:
        return -1 * self

    def __len__(self) -> int:
        return len(self.data)


def _apply(op_: type(Op), *tensors: Optional[Tensor], **kwargs: Any) -> Tensor:
    tensor_args = [t for t in tensors if t is not None]
    op = op_(kwargs)

    fwd_args = [t.data if t is not None else None for t in tensors]
    data = op.forward(*fwd_args, **kwargs)

    result_req_grad = any(t.req_grad for t in tensor_args)
    if _autograd_tracking_active and result_req_grad:
        return Tensor(data, op=op, src=tensors, requires_grad=True)

    return Tensor(data)


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


def _computed_node_dfs(node: Tensor, queue: list[Tensor], visited: set) -> list[Tensor]:
    if node not in visited:
        visited.add(node)
        if not node.src:
            return []
        for p in node.src:
            if p is not None and p.req_grad:
                _ = _computed_node_dfs(p, queue, visited)
        queue.append(node)
    return queue
