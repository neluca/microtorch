# microtorch[![unit_test](https://github.com/neluca/microtorch/actions/workflows/unit_test.yaml/badge.svg)](https://github.com/neluca/microtorch/actions/workflows/unit_test.yaml) 

**microtorch**是以**第一性原理**为准则，从零开始创建一个深度学习库，它参考自我的另一个项目[regrad](https://github.com/neluca/regrad)。该库来自于一个简单的自动微分引擎，并使用该库来构建和训练复杂的神经网络。它旨在通过展示每一个细节并减少像`PyTorch`这样闪亮的机器学习库所具有的抽象性，来揭示构建深度学习模型的底层原理和工作机制。自动微分引擎是一种用于自动计算函数导数的工具，它在深度学习中非常重要，因为神经网络的训练过程需要计算损失函数关于模型参数的梯度，而自动微分引擎可以高效地完成这一任务。

- 优先考虑学习和透明度，而不是过度关注优化；
- `API`接口与`Pytorch`非常相似；
- 最小第三方依赖，核心只依赖`numpy`，`pytorch`仅用于单元测试，验证梯度计算的正确性；
- 所有算子结构清晰，十分便于理解。

### 使用举例

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

模型的计算图如下：

![model](./model.png)

### 单元测试

运行单元测试需要安装[PyTorch](https://pytorch.org/)库，用于验证梯度计算的正确性。

```bash
python -m pytest
```

