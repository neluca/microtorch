from microtorch import tensor, mse_loss
import microtorch as nn
import microtorch as optim
from tools import draw_to_html


class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        # A single linear layer (input_dim -> output_dim).
        # Mathematically: fc(x) = xW^T + b
        # where W is weight and b is bias.
        self.fc = nn.Linear(input_dim, output_dim)
        self.ac = nn.Tanh()

    def forward(self, x):
        # Simply compute xW^T + b without any additional activation.
        return self.ac(self.fc(x))


# Create a sample input tensor x with shape (1, 3).
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

print((x @ weights.T + bias).item())
