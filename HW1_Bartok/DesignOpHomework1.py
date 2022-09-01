# MAE598 Design Optimization HW1
# Kevin Bartok kbartok1@asu.edu
import numpy as np
from scipy.optimize import minimize


fun = lambda x: (x[0]-x[1])**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2 + (x[4]-1)**2

cons = ({'type': 'eq', 'fun': lambda x: x[0] + 3*x[1]},
        {'type': 'eq', 'fun': lambda x: x[2] + x[3] - 2*x[4]},
        {'type': 'eq', 'fun': lambda x: x[1] - x[4]})
bnds = ((-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10))
x0 = np.array([1, 2, 3, 4, 5])
res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)

print(res.fun)
