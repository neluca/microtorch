from microtorch import tensor
import microtorch as nn
import microtorch as optim
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        # A single linear layer (input_dim -> output_dim).
        # Mathematically: fc(x) = xW^T + b
        # where W is weight and b is bias.
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Simply compute xW^T + b without any additional activation.
        return self.fc(x)


# Create a sample input tensor x with shape (1, 3).
# 'requires_grad=True' means we want to track gradients for x.
x = tensor([[-1.0, 0.0, 2.0]], requires_grad=True)

# Initialize the simple neural network.
# This layer has a weight matrix W of shape (3, 1) and a bias of shape (1,).
model = SimpleNN(input_dim=3, output_dim=1)

# Use SGD with a learning rate of 0.03
optimizer = optim.SGD(model.parameters(), lr=0.03)

# We want the output to get close to 1.0 over time.
y_true = 1.0

for epoch in range(20):
    # Reset (zero out) all accumulated gradients before each update.
    optimizer.zero_grad()

    # --- Forward pass ---
    # prediction = xW^T + b
    y_pred = model(x)
    print(f"Epoch {epoch}: {y_pred}")

    # Define a simple mean squared error function
    loss = ((y_pred - y_true) ** 2).mean()

    # --- Backward pass ---
    # Ultimately we need to compute the gradient of the loss with respect to the weights
    # Specifically, if Loss = (pred - 1)^2, then:
    #   dL/d(pred) = 2 * (pred - 1)
    #   d(pred)/dW = d(xW^T + b) / dW = x^T
    # By chain rule, dL/dW = dL/d(pred) * d(pred)/dW = [2 * (pred - 1)] * x^T
    loss.backward()

    # --- Update weights ---
    optimizer.step()

weights = model.fc.parameters()[0]
bias = model.fc.parameters()[1]
gradient = model.fc.parameters()[0].grad

print("[After Training] Gradients for fc weights:", gradient)
print("[After Training] layer weights:", weights)
print("[After Training] layer bias:", bias)
print(x.data @ weights.data.T + bias.data)
