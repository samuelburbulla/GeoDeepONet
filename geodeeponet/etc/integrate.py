import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(3 * np.pi * x[:, 0]) * np.sin(3 * np.pi * x[:, 1])

exact_integral = 4 / (9 * np.pi**2)  # over [0, 1] x [0, 1]

def approx_integral(n):
    h = 1 / (n - 1)
    points = np.mgrid[0:1+1e-8:h, 0:1+1e-8:h].reshape(2, -1).T
    return np.sum(f(points)) * h**2

n_values = 2**np.arange(2, 12)

errors = []
for n in n_values:
    error = np.abs(approx_integral(n) - exact_integral)
    errors.append(error)

plt.loglog(n_values, errors)
plt.xlabel('n')
plt.ylabel('Error')
plt.show()