import torch
import numpy as np
import matplotlib.pyplot as plt


obj_func = lambda x: x ** 2
x = torch.tensor(10.0, requires_grad=True)
x_history = []
y_history = []
eta = torch.tensor(-0.99)
thresh = 0.1

cnt = 0
while True:
    y = obj_func(x)
    y_history.append(y)
    x_history.append(x.item())
    if abs(y.item()) < thresh:
        break

    y.backward()
    x = torch.tensor(x + eta * x.grad, requires_grad=True)
    cnt += 1
    if cnt > 1000:
        break

x_plot = np.arange(-10, 10, 0.1)
y_plot = obj_func(x_plot)
plt.plot(x_plot, y_plot, color="r")
plt.plot(x_history, y_history)
plt.savefig("hoge.jpg")
