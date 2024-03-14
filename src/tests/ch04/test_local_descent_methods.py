import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np
import warnings

from ch04 import line_search, backtracking_line_search, strong_backtracking, trust_region_descent
from TestFunctions import rosenbrock


class TestLocalDescentMethods():
    def test_line_search(self):
        def f(x): return np.sin(x[0]*x[1]) + np.exp(x[1] + x[2]) - x[2]
        x = np.array([1.0, 2.0, 3.0])
        d = np.array([0.0, -1.0, -1.0])
        x_prime = line_search(f, x, d)
        exp_x_prime = np.array([1.0, -1.127, -0.127])
        assert np.all(np.abs(x_prime - exp_x_prime) < 1e-3)

    def test_backtracking_line_search(self):
        def f(x): return x[0]**2 + x[0]*x[1] + x[1]**2
        def grad_f(x): return np.array([2*x[0] + x[1], 2*x[1] + x[0]])
        x = np.array([1.0, 2.0])
        d = np.array([-1.0, -1.0])
        alpha = backtracking_line_search(f, grad_f, x, d, alpha=10)
        x_prime = x + alpha * d
        exp_x_prime = np.array([-1.5, -0.5])
        assert np.all(np.abs(x_prime - exp_x_prime) < 1e-10)

    def test_strong_backtracking(self):
        def f(x): return x[0]**2 + x[0]*x[1] + x[1]**2
        def grad_f(x): return np.array([2*x[0] + x[1], 2*x[1] + x[0]])
        x = np.array([1.0, 2.0])
        d = np.array([-1.0, -1.0])
        alpha = strong_backtracking(f, grad_f, x, d)
        x_prime = x + alpha * d
        assert f(x_prime) < f(x)
        assert f(x_prime) < 3.25

    def test_trust_region_descent(self, eps: float = 1e-8):
        warnings.simplefilter(action='ignore', category=FutureWarning)
       
        # Rosenbrock
        x = np.array([-5.0, -3.0])
        x_prime = trust_region_descent(f=rosenbrock,
                                       grad_f=rosenbrock.grad,
                                       H=rosenbrock.hess,
                                       x=x,
                                       k_max=15)
        f_min, x_min = rosenbrock.global_min()
        assert np.abs(rosenbrock(x_prime) - f_min) < eps
        assert np.all(np.abs(x_prime - x_min) < eps)
