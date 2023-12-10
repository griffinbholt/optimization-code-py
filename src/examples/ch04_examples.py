import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from ch04 import line_search, backtracking_line_search

def example_4_1():
    """Example 4.1: Line search used to minimize a function along a descent direction"""
    def f(x): return np.sin(x[0]*x[1]) + np.exp(x[1] + x[2]) - x[2]
    x = np.array([1.0, 2.0, 3.0])
    d = np.array([0.0, -1.0, -1.0])
    x_next = line_search(f, x, d)
    alpha_opt = (x_next[1] - x[1])/d[1]

    # Print results
    print("α* = ", alpha_opt)
    print("x' = ", x_next)

    # Plot line search objective
    alpha = np.arange(0.0, 5.0, 0.01)
    def ls_obj(alpha): return np.sin(2 - alpha) + np.exp(5 - 2*alpha) + alpha - 3
    plt.plot(alpha, ls_obj(alpha))
    plt.scatter([alpha_opt], [ls_obj(alpha_opt)], label='α*')
    plt.xlabel("")
    plt.ylabel("line search objective")
    plt.legend()
    plt.show()


def example_4_2():
    """Example 4.2: An example of backtracking line search, an approximate line search method"""
    def f(x): return x[0]**2 + x[0]*x[1] + x[1]**2
    def grad_f(x): return np.array([[2, 1], [1, 2]]) @ x
    x = np.array([1.0, 2.0])
    d = np.array([-1.0, -1.0])
    sigma = 0.9

    alpha_opt = backtracking_line_search(f, grad_f, x, d, alpha=10.0, p=0.5, beta=1e-4)
    candidate_x = x + alpha_opt*d
    cand_x_deriv_d = np.dot(grad_f(candidate_x), d)
    adj_x_deriv_d = sigma * np.dot(grad_f(x), d)

    print("α* = ", alpha_opt)
    print("x' = ", candidate_x)
    print("2nd Wolfe Condition: ", cand_x_deriv_d >= adj_x_deriv_d, " ({:.1f} >= {:.1f})".format(cand_x_deriv_d, adj_x_deriv_d))
