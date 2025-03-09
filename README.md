# microtorch[![unit_test](https://github.com/neluca/microtorch/actions/workflows/unit_test.yaml/badge.svg)](https://github.com/neluca/microtorch/actions/workflows/unit_test.yaml) 

[中文](./README_zh.md)

**microtorch** is a deep learning library created from scratch based on **the principle of first principles**, inspired by my other project [regrad](https://github.com/neluca/regrad). This library originates from a simple automatic differentiation engine and uses this engine to build and train complex neural networks. It aims to reveal the underlying principles and mechanisms of building deep learning models by demonstrating every detail and reducing the abstraction found in shiny machine learning libraries like`PyTorch`. An automatic differentiation engine is a tool for automatically computing the derivatives of functions, which is crucial in deep learning because the training process of neural networks requires calculating the gradients of the loss function with respect to model parameters, and the automatic differentiation engine can efficiently accomplish this task.

- Prioritizing learning and transparency over optimization;
- The `API` interface is very similar to `Pytorch`;
- Minimal third-party dependencies; the core relies only on `numpy`, and `pytorch` is used only for unit testing to verify the correctness of gradient calculations;
- All operator structures are clear and easy to understand.

### Usage Example

```python
from microtorch import tensor, mse_loss
import microtorch as nn
import microtorch as optim
from tools import draw_to_html

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

x = tensor([[-1.0, 0.0, 2.0]])

# Initialize the simple neural network.
# This layer has a weight matrix W of shape (3, 1) and a bias of shape (1,).
model = SimpleNN(input_dim=3, output_dim=1)

# Draw the computational graph of the model.
draw_to_html(model(x), "model")

# Use SGD with a learning rate of 0.03
optimizer = optim.SGD(model.parameters(), lr=0.03)

# We want the output to get close to 1.0 over time.
y_true = 1.0

for epoch in range(30):
    # Reset (zero out) all accumulated gradients before each update.
    optimizer.zero_grad()

    # --- Forward pass ---
    # prediction = xW^T + b
    y_pred = model(x)
    print(f"Epoch {epoch}: {y_pred.item()}")

    # Define a simple mean squared error function
    loss = mse_loss(y_pred, tensor(y_true))

    # --- Backward pass ---
    loss.backward()

    # --- Update weights ---
    optimizer.step()

weights = model.fc.parameters()[0]
bias = model.fc.parameters()[1]
gradient = model.fc.parameters()[0].grad

print("[After Training] Gradients for fc weights:", gradient)
print("[After Training] layer weights:", weights)
print("[After Training] layer bias:", bias)
print("y_pred: ", (x @ weights.T + bias).item())
```

```
......
Epoch 26: 0.999990167623951
Epoch 27: 0.9999937072793286
Epoch 28: 0.9999959726587704
Epoch 29: 0.999997422501613
[After Training] Gradients for fc weights: [[ 5.15499677e-06  0.00000000e+00 -1.03099935e-05]]
[After Training] layer weights: Tensor([[-0.53356579  0.39807214 -0.04513151]], requires_grad=True)
[After Training] layer bias: Tensor([0.55669557], requires_grad=True)
y_pred:  0.9999983504010324
```

The computational graph of the model is as follows:

![model](./model.png)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

