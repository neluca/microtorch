import microtorch
import numpy as np

np.random.seed(1024)

x = np.random.rand(3, 3)
x_mt = microtorch.tensor(x, requires_grad=True)

y_mt = x_mt.max(dim=1)
loss_mt = y_mt.sum()
print(loss_mt)
