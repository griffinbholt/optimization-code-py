import sys; sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np

from ch04 import backtracking_line_search, solve_trust_region_subproblem
from TestFunctions import rosenbrock
from convenience import plot_contour


def figure_4_2():
    """
    Figure 4.2: Backtracking line search used on the Rosenbrock function.
    The black lines show the eight iterations taken by the descent method and
    the red lines show the points considered during each line search.
    """
    x = np.array([-1.75, -1.6])  # Starting point

    fig = plt.figure()
    plot_contour(fig, rosenbrock, xlim=(-2.1, 2.1), ylim=(-2.1, 2.1), xstride=0.01, ystride=0.01, levels=[0, 1, 2, 3, 5, 9, 25, 50, 100])
    plt.scatter([x[0]], [x[1]], c="black", s=10)
    plt.annotate("1", x, xytext=[-5, -13], textcoords='offset points')
    for i in range(7):
        d = -rosenbrock.grad(x)  # use negative gradient as the descent direction
        alpha_opt = backtracking_line_search(rosenbrock, rosenbrock.grad, x, d, alpha=100.0)
        x_next = x + alpha_opt*d
        plt.plot([x[0], x_next[0]], [x[1], x_next[1]], c="black")
        plt.scatter([x_next[0]], [x_next[1]], c="black", s=10)
        plt.annotate(str(i + 2), x_next, xytext=[-5, -13], textcoords='offset points')
        x = x_next
    plt.title("Figure 4.2")
    plt.show()


def figure_4_9(): # TODO - Needs some more work - something isn't working quite right
    """Figure 4.9: Trust region optimization used on the Rosenbrock function"""
    x = np.array([-1.75, -1.75])  # Starting point
    k_max, eta_1, eta_2, gamma_1, gamma_2, delta = 9, 0.25, 2.0, 0.5, 2.0, 1.0

    fig = plt.figure()
    plot_contour(fig, rosenbrock, xlim=(-2.1, 2.1), ylim=(-2.1, 3.1), xstride=0.01, ystride=0.01, levels=[0, 1, 2, 3, 5, 9, 25, 50, 100])
    plt.scatter([x[0]], [x[1]], c="black", s=10)
    plt.annotate("1", x, xytext=[-5, -13], textcoords='offset points')

    # Trust Region Descent (taking from ch04.py)
    y = rosenbrock(x)
    for i in range(k_max):
        circle = plt.Circle((x[0], x[1]), delta, color='black', fill=False, alpha=0.1*(i + 1))
        plt.gca().add_patch(circle)
        x_prime, y_prime = solve_trust_region_subproblem(rosenbrock.grad, rosenbrock.hess, x, delta)
        r = (y - rosenbrock(x_prime)) / (y - y_prime)
        if r < eta_1:
            delta *= gamma_1
        else:
            x, y = x_prime, y_prime
            if r > eta_2:
                delta *= gamma_2
        plt.scatter([x[0]], [x[1]], c="black", s=10)
        plt.annotate(str(i + 2), x, xytext=[-5, -13], textcoords='offset points')
    plt.xlim((-2.1, 2.1))
    plt.ylim((-2.1, 3.1))
    plt.gca().set_aspect('equal')
    plt.show()
