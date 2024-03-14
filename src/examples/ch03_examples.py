import sys; sys.path.append('../');

import numpy as np

from ch03 import PHI


def example_3_1():
    """
    Example 3.1: Using Fibonacci search with five function evaluations
    to optimize a univariate function
    """
    def f(x): return np.exp(x - 2) - x
    a, b = -2, 6
    n = 5
    eps = 1e-2

    # Fibonacci search (taken directly from ch03.py)
    print("Original Interval: ", (a, b), "\n")
    s = (1 - np.sqrt(5)) / (1 + np.sqrt(5))
    p = 1 / ((PHI*(1 - (s**(n + 1)))) / (1 - (s**n)))
    d = p*b + (1 - p)*a
    y_d = f(d)
    print("f(" + str(round(d, 2)) + ") = ", y_d)
    for i in range(1, n):
        if i == n - 1:
            c = eps*a + (1 - eps)*d
        else:
            c = p*a + (1 - p)*b
        y_c = f(c)
        print("f(" + str(round(c, 2)) + ") = ", y_c)
        if y_c < y_d:
            b, d, y_d = d, c, y_c
            print("Interval Update: ", (round(a, 2), round(b, 2)) if a < b else (round(b, 2), round(a, 2)), "\n")
        else:
            a, b = b, c
            print("Interval Update: ", (round(a, 2), round(b, 2)) if a < b else (round(b, 2), round(a, 2)), "\n")
        p = 1 / ((PHI*(1 - (s**(n - i + 1)))) / (1 - (s**(n - i))))
    print("Final Interval: ", (round(a, 2), round(b, 2)) if a < b else (round(b, 2), round(a, 2)))
