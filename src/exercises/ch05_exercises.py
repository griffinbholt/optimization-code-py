import sys; sys.path.append("../")

import numpy as np

from ch04 import line_search
from ch05 import GradientDescent


def exercise_5_2(x0: float):
    """Exercise 5.2: Gradient Descent for f(x) = x^4"""
    def f(x): return x**4
    def deriv(x): return 4*(x**3)

    M = GradientDescent(alpha=1.0)
    M.initialize(f, deriv, x0)
    print("Initial point: x0 = ", x0)
    print("Derivative at x0: ", deriv(x0))
    x = M.step(f, deriv, x0)
    print("After 1 iteration of Gradient Descent, x = ", x)
    print("Derivative at x: ", deriv(x))
    x = M.step(f, deriv, x)
    print("After 2 iterations of Gradient Descent, x = ", x)
    print("Derivative at x: ", deriv(x))


def exercise_5_3():
    """Exercise 5.3: Gradient Descent: Unit Step vs. Exact Line Search"""
    def f(x): return np.exp(x) + np.exp(-x)
    def deriv(x): return np.exp(x) - np.exp(-x)
    x0 = 10.0

    # Unit Step
    M = GradientDescent(alpha=1.0)
    M.initialize(f, deriv, x0)
    x = M.step(f, deriv, x0)
    print("With Unit Step:")
    print("After 1 iteration of Gradient Descent, x = ", x)
    with np.errstate(over='ignore'):
        print("Derivative at x: ", deriv(x), "\n")
    print("=> Gradient Descent diverges.")

    # Exact Line Search
    with np.errstate(over='ignore'):
        x = line_search(f, x0, deriv(x0))
    print("With Exact Line Search:")
    print("After 1 iteration of Gradient Descent, x = ", x)
    print("Derivative at x: ", deriv(x))
    print("=> Gradient Descent converges to the minimum.")


def exercise_5_7():
    """Exercise 5.7: Conjugate Gradient Descent"""
    def f(x): return x[0]**2 + x[0]*x[1] + x[1]**2 + 5
    def grad_f(x): return np.array([2*x[0] + x[1], 2*x[1] + x[0]])
    x0 = np.ones(2)

    # Conjugate Gradient Descent (taken directly from ch05.py)
    g = grad_f(x0)
    d = -g

    # First Step
    g_prime = grad_f(x0)
    beta = np.maximum(0, np.dot(g_prime, g_prime - g) / np.dot(g, g))
    d = -g_prime + beta*d
    x = line_search(f, x0, d)
    g = g_prime.copy()
    print("After 1 iteration of CG, the normalized descent direction is d = ", d/np.linalg.norm(d))

    # Second Step
    g_prime = grad_f(x)
    beta = np.maximum(0, np.dot(g_prime, g_prime - g) / np.dot(g, g))
    d = -g_prime + beta*d
    x = line_search(f, x, d)
    print("After 2 iterations of CG, x = ", x)
    print("Gradient at x: ", grad_f(x))
    print("=> Conjugate Gradient Descent converges after 2 iterations.")
