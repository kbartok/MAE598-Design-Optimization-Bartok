import numpy as np
import matplotlib.pyplot as plt

#fun = lambda x: 10*x[1]**2 + 5*x[0]**2 - 8*x[0] - 14*x[1] + 12*x[0]*x[1] + 5


def f(x):
    return 10*x[1]**2 + 5*x[0]**2 - 8*x[0] - 14*x[1] + 12*x[0]*x[1] + 5


def g(x):
    return np.array([10*x[0]-8+12*x[1], 20*x[1]-14+12*x[0]])


# Termination criteria
eps = 1e-6
sol = np.array([1, 1])
alpha = 0.01
i = 0
error = [np.linalg.norm(g(sol))]
fn = [f([1,1])]
test = [0-np.linalg.norm(g(sol))]
while error[i] > eps:
    sol = sol - alpha*g(sol)
    #print(sol)
    i = i+1
    error.append(np.linalg.norm(g(sol)))
    fn.append(f(sol))

# ||gradient(x)|| < eplison (some small number) 1e-3 or 1e-6
print("Gradient Descent Convergence\n", sol)

# calc f
print("Newton")
# Newton's method
H = np.array([[10, 12], [12, 20]])
sol = np.array([0, 0])
e2 = np.linalg.norm(g(sol))
while e2 > eps:
    sol = sol - np.matmul(np.linalg.inv(H), g(sol))
    e2 = np.linalg.norm(g(sol))
    print(sol)
i = list(range(0, len(error)))
fn_error = fn - f(sol)
plt.plot(i, np.absolute(fn_error))
#plt.plot(i, np.log10(np.absolute(error)))
plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.suptitle("Convergence of Gradient Descent")
plt.title("Initial guess [x3, x4] = [1,1]")
plt.show()



