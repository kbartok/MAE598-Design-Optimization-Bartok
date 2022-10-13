import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

T = 20
a = np.array(([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]]))
pSat_w = 10**(a[0, 0]-(a[0, 1]/(T+a[0, 2])))
pSat_d = 10**(a[1, 0]-(a[1, 1]/(T+a[1, 2])))
p = np.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])
x1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
x2 = 1-x1

p = torch.tensor(p, requires_grad=False, dtype=torch.float64)
x1 = torch.tensor(x1, requires_grad=False, dtype=torch.float64)
x2 = torch.tensor(x2, requires_grad=False, dtype=torch.float64)

# includes initial guess of A12 and A21
A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)

step = 0.0001
eps = 1e-3
P = x1 * torch.exp(A[0] * ((A[1]*x2)/(A[0]*x1+A[1]*x2)) ** 2) * pSat_w + x2 * torch.exp(A[1] * ((A[0]*x1)/(A[0]*x1+A[1]*x2)) ** 2) * pSat_d

loss = (P-p) ** 2
loss = loss.sum()

# computes gradient for current guess
loss.backward()

# norm of a grad
er = torch.norm(A.grad).item()
while er >= eps:
    P = x1 * torch.exp(A[0] * ((A[1] * x2) / (A[0] * x1 + A[1] * x2)) ** 2) * pSat_w + x2 * torch.exp(
        A[1] * ((A[0] * x1) / (A[0] * x1 + A[1] * x2)) ** 2) * pSat_d

    loss = (P - p) ** 2
    loss = loss.sum()

    # computes gradient for current guess
    loss.backward()

    # norm of a grad
    er = torch.norm(A.grad).item()
    # get numerical value of gradient
    with torch.no_grad():
        A -= step * A.grad

        # have to zero out the grad
        A.grad.zero_()

print('estimation A12 and A21 is: ', A)
print('final loss is: ', loss.data.numpy())

# Plotting
P = P.detach().numpy()
p = p.detach().numpy()
x1 = x1.detach().numpy()

plt.plot(x1, P, label='Predicted Pressure')
plt.plot(x1, p, label='Actual Pressure')
plt.xlabel("x1 (Water Mixture)")
plt.ylabel("Pressure")
plt.legend()
plt.title("Comparison of Predicted and Actual Pressure")
plt.show()

