from functools import reduce
from typing import Literal
from ._autograd import Tensor, uniform, randn

FanMode = Literal["fan_in", "fan_out"]


def _calculate_fan_in_and_fan_out(*shape: int) -> tuple[int, int]:
    assert len(shape) != 1, f"Tensor with dims {shape} is not supported. Must be at least 2D."

    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        kernel_prod = reduce(lambda x, y: x * y, shape[2:], 1)
        fan_in = shape[1] * kernel_prod
        fan_out = shape[0] * kernel_prod

    return fan_in, fan_out


def xavier_uniform(*shape: int, gain: float = 1.0, requires_grad: bool = False) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(*shape)
    bound = (6 / (fan_in + fan_out)) ** 0.5 * gain
    return uniform(*shape, low=-bound, high=bound, requires_grad=requires_grad)


def xavier_normal(*shape: int, gain: float = 1.0, requires_grad: bool = False) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(*shape)
    std = (2 / (fan_in + fan_out)) ** 0.5 * gain
    return randn(*shape, mean=0, std=std, requires_grad=requires_grad)


def kaiming_uniform(*shape: int, mode: FanMode = "fan_in", requires_grad: bool = False) -> Tensor:
    assert mode in {"fan_in", "fan_out"}, "mode must be either 'fan_in' or 'fan_out'."

    fan_in, fan_out = _calculate_fan_in_and_fan_out(*shape)
    fan = fan_in if mode == "fan_in" else fan_out
    bound = (6 / fan) ** 0.5
    return uniform(*shape, low=-bound, high=bound, requires_grad=requires_grad)


def kaiming_normal(*shape: int, mode: FanMode = "fan_in", requires_grad: bool = False) -> Tensor:
    assert mode in {"fan_in", "fan_out"}, "mode must be either 'fan_in' or 'fan_out'."

    fan_in, fan_out = _calculate_fan_in_and_fan_out(*shape)
    fan = fan_in if mode == "fan_in" else fan_out
    std = (2 / fan) ** 5
    return randn(*shape, mean=0, std=std, requires_grad=requires_grad)
