import sys; sys.path.append('../')

import numpy as np


def exercise_3_4():
    """Exercise 3.4: Applying Bisection to f(x) = x^2/2 - x, starting with [0, 1000]"""
    def f_prime(x): return x - 1
    a, b = 0.0, 1000.0
    y_a, y_b = f_prime(a), f_prime(b)

    for i in range(3):  # Execute 3 steps of the algorithm
        x = (a + b) / 2
        y = f_prime(x)
        if y == 0:
            a, b = x, x
        elif np.sign(y) == np.sign(y_a):
            a = x
        else:
            b = x
        print("Iteration " + str(i + 1) + ": ", (a, b))
