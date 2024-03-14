import sys; sys.path.append('../')

import numpy as np

from ch06 import newtons_method
from TestFunctions import booth


def example_6_1():
    """Example 6.1: Newton's method used to minimize Booth's function"""
    x = np.array([9.0, 8.0])
    x_prime = newtons_method(booth.grad, booth.hess, x, eps=1e-5, k_max=1)

    print("After 1 iteration of Newton's Method, x = ", x_prime)
    print("Gradient at x: ", booth.grad(x))
