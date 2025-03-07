import torch
import microtorch
import numpy as np

np.random.seed(1024)


def test_add():
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_mt = microtorch.tensor(x, requires_grad=True)
    y_mt = microtorch.tensor(y, requires_grad=True)

    z_t = x_t + y_t
    z_t.sum().backward()

    z_mt = x_mt + y_mt
    z_mt.sum().backward()

    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), "Addition results do not match between pytorch and microtorch."
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), "Gradients for x do not match between pytorch and microtorch."
    assert np.allclose(
        y_t.grad.numpy(), y_mt.grad
    ), "Gradients for y do not match between pytorch and microtorch."


def test_sub():
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_mt = microtorch.tensor(x, requires_grad=True)
    y_mt = microtorch.tensor(y, requires_grad=True)

    z_t = x_t - y_t
    z_t.sum().backward()

    z_mt = x_mt - y_mt
    z_mt.sum().backward()

    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), "Subtraction results do not match between pytorch and microtorch."
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), "Gradients for x do not match between pytorch and microtorch."
    assert np.allclose(
        y_t.grad.numpy(), y_mt.grad
    ), "Gradients for y do not match between pytorch and microtorch."


def test_mul():
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_mt = microtorch.tensor(x, requires_grad=True)
    y_mt = microtorch.tensor(y, requires_grad=True)

    z_t = x_t * y_t
    z_t.sum().backward()

    z_mt = x_mt * y_mt
    z_mt.sum().backward()

    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), "Multiplication results do not match between pytorch and microtorch."
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), "Gradients for x do not match between pytorch and microtorch."
    assert np.allclose(
        y_t.grad.numpy(), y_mt.grad
    ), "Gradients for y do not match between pytorch and microtorch."


def test_div():
    x = np.random.rand(3, 3) + 0.1  # Adding 0.1 to avoid division by zero
    y = np.random.rand(3, 3) + 0.1

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_mt = microtorch.tensor(x, requires_grad=True)
    y_mt = microtorch.tensor(y, requires_grad=True)

    z_t = x_t / y_t
    z_t.sum().backward()

    z_mt = x_mt / y_mt
    z_mt.sum().backward()

    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), "Division results do not match between pytorch and microtorch."
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), "Gradients for x do not match between pytorch and microtorch."
    assert np.allclose(
        y_t.grad.numpy(), y_mt.grad
    ), "Gradients for y do not match between pytorch and microtorch."


def test_sum():
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_mt = microtorch.tensor(x, requires_grad=True)

    z_t = x_t.sum()
    z_t.backward()

    z_mt = x_mt.sum()
    z_mt.backward()

    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), "Sum results do not match between pytorch and microtorch."
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), "Gradients do not match between pytorch and microtorch."


def test_sum_axis(axis=1):
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_mt = microtorch.tensor(x, requires_grad=True)

    z_t = x_t.sum(dim=axis)
    z_t.sum().backward()  # Summing again to get a scalar for backward()

    z_mt = x_mt.sum(axis=axis)
    z_mt.sum().backward()  # Summing again to get a scalar for backward()

    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), f"Sum along axis {axis} results do not match between pytorch and microtorch."
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), f"Gradients along axis {axis} do not match between pytorch and microtorch."
