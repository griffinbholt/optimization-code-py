import sys; sys.path.append('../')

import numpy as np


def example_5_1():
    """Example 5.1: Computing the gradient descent direction."""
    def f(x): return x[0]*(x[1]**2)
    def grad_f(x): return np.array([x[1]**2, 2*x[0]*x[1]])
    x = np.array([1.0, 2.0])
    d = -grad_f(x)

    print("Unnormalized descent direction: d = ", d)
    print("Normalized descent direction: d = ", d/np.linalg.norm(d))
