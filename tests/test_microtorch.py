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


def test_reshape():
    x = np.random.rand(2, 3, 4)

    x_t = torch.tensor(x, requires_grad=True)
    x_mt = microtorch.tensor(x, requires_grad=True)

    new_shape = (3, 2, 4)
    z_t = x_t.reshape(new_shape)
    z_t.sum().backward()

    z_mt = x_mt.reshape(new_shape)
    z_mt.sum().backward()

    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), "Reshape results do not match between pytorch and microtorch."
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), "Gradients do not match between pytorch and microtorch."


def test_custom_eq():
    # Define custom function f(x, y) within the scope of test_custom_eq
    def f(x, y):
        return x * x + (x * y) / (x + y) + x * (x + y)

    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_mt = microtorch.tensor(x, requires_grad=True)
    y_mt = microtorch.tensor(y, requires_grad=True)

    # Compute f(x, y) using pytorch
    z_t = f(x_t, y_t)
    z_t.sum().backward()

    # Compute f(x, y) using microtorch
    z_mt = f(x_mt, y_mt)
    z_mt.sum().backward()

    # Assertions to check if both results match
    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), "Custom eq results do not match between pytorch and microtorch."
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), "Gradients for x do not match between pytorch and microtorch."
    assert np.allclose(
        y_t.grad.numpy(), y_mt.grad
    ), "Gradients for y do not match between pytorch and microtorch."


def test_matmul():
    x = np.random.rand(3, 4)
    y = np.random.rand(4, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_mt = microtorch.tensor(x, requires_grad=True)
    y_mt = microtorch.tensor(y, requires_grad=True)

    z_t = x_t @ y_t
    z_t.sum().backward()

    z_mt = x_mt @ y_mt
    z_mt.sum().backward()

    # Assertions
    assert np.allclose(z_t.detach().numpy(), z_mt.data), "Matmul results do not match."
    assert np.allclose(x_t.grad.numpy(), x_mt.grad), "Gradients for x do not match."
    assert np.allclose(y_t.grad.numpy(), y_mt.grad), "Gradients for y do not match."


def test_pow():
    x = np.random.rand(3, 3)
    power = 2

    x_t = torch.tensor(x, requires_grad=True)
    x_mt = microtorch.tensor(x, requires_grad=True)

    z_t = x_t ** power
    z_t.sum().backward()

    z_mt = x_mt ** power
    z_mt.sum().backward()

    # Assertions
    assert np.allclose(z_t.detach().numpy(), z_mt.detach()), "Pow results do not match."
    assert np.allclose(x_t.grad.numpy(), x_mt.grad), "Gradients do not match."


def test_tanh():
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_mt = microtorch.tensor(x, requires_grad=True)

    z_t = torch.tanh(x_t)
    z_t.sum().backward()

    z_mt = microtorch.tanh(x_mt)
    z_mt.sum().backward()

    # Assertions
    assert np.allclose(z_t.detach().numpy(), z_mt.detach()), "Tanh results do not match."
    assert np.allclose(x_t.grad.numpy(), x_mt.grad), "Gradients do not match."


def test_relu():
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_mt = microtorch.tensor(x, requires_grad=True)

    z_t = torch.relu(x_t)
    z_t.sum().backward()

    z_mt = microtorch.relu(x_mt)
    z_mt.sum().backward()

    # Assertions
    assert np.allclose(z_t.detach().numpy(), z_mt.detach()), "ReLU results do not match."
    assert np.allclose(x_t.grad.numpy(), x_mt.grad), "Gradients do not match."


def test_mse_loss():
    y_pred = np.random.rand(2, 2)
    y_true = np.random.rand(2, 2)

    y_pred_t = torch.tensor(y_pred, requires_grad=True)
    y_true_t = torch.tensor(y_true, requires_grad=False)

    y_pred_mt = microtorch.tensor(y_pred, requires_grad=True)
    y_true_mt = microtorch.tensor(y_true, requires_grad=False)

    loss_t = torch.nn.functional.mse_loss(y_pred_t, y_true_t)
    loss_t.backward()

    loss_mt = microtorch.mse_loss(y_pred_mt, y_true_mt)
    loss_mt.backward()

    # Assertions
    assert np.allclose(loss_t.detach().numpy(), loss_mt.detach()), "MSE Loss results do not match."
    assert np.allclose(y_pred_t.grad.numpy(), y_pred_mt.grad), "Gradients do not match."


def test_transpose():
    x = np.random.rand(3, 4)  # Create a 3x4 random matrix

    x_t = torch.tensor(x, requires_grad=True)  # Create a pytorch tensor
    x_mt = microtorch.tensor(x, requires_grad=True)  # Create a microtorch tensor

    # Perform the transpose operation using pytorch
    z_t = x_t.t()
    z_t.sum().backward()  # Compute the gradients via backpropagation

    # Perform the transpose operation using microtorch
    z_mt = x_mt.t()
    z_mt.sum().backward()  # Compute the gradients via backpropagation

    # Assert that the transpose operation results are the same for both pytorch and microtorch
    assert np.allclose(
        z_t.detach().numpy(), z_mt.detach()
    ), "Transpose results do not match between pytorch and microtorch."

    # Assert that the gradients are the same for both pytorch and microtorch
    assert np.allclose(
        x_t.grad.numpy(), x_mt.grad
    ), "Gradients do not match between pytorch and microtorch."
