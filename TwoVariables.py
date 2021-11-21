from mpl_toolkits.mplot3d import axes3d, Axes3D
import torch
import numpy as np
import matplotlib.pyplot as plt


obj_func = lambda x1, x2: x1 ** 2 + (x2-1)**2 + x1 + x2
x1 = torch.tensor(10.0, requires_grad=True)
x1_history = []
x2 = torch.tensor(5.0, requires_grad=True)
x2_history = []
y_history = []
eta1 = torch.tensor(-0.4)
eta2 = torch.tensor(-0.9)
thresh = 0.1

cnt = 0
while True:
    y = obj_func(x1, x2)
    y_history.append(y)
    x1_history.append(x1.item())
    x2_history.append(x2.item())
    if abs(y.item()) < thresh:
        break

    y.backward()
    x1 = torch.tensor(x1 + eta1 * x1.grad, requires_grad=True)
    x2 = torch.tensor(x2 + eta2 * x2.grad, requires_grad=True)
    cnt += 1
    if cnt > 1000:
        break

x1_plot = np.arange(-10, 10, 0.1)
x2_plot = np.arange(-10, 10, 0.1)
x1_mesh, x2_mesh = np.meshgrid(x1_plot, x2_plot)
print(x1_mesh.shape)
y_mesh = obj_func(x1_mesh, x2_mesh)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(x1_mesh, x2_mesh, y_mesh)
ax.plot(x1_history, x2_history, y_history, color='red')
#plt.savefig("twodim.jpg")
plt.show()
