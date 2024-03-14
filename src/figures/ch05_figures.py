import sys; sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np

from ch05 import GradientDescent

from TestFunctions import rosenbrock
from convenience import plot_contour


def figure_5_1():  # TODO - To duplicate the effect, I need to see the parameters
    """
    Figure 5.1: Gradient descent can result in zig-zagging in narrow canyons.
    Here we see the effect on the Rosenbrock function.
    """
    x = np.array([-1.1, -1.1])  # Starting point
    M = GradientDescent(alpha = 0.025)

    fig = plt.figure()
    plot_contour(fig, rosenbrock, xlim=(-2.1, 2.1), ylim=(-2.1, 2.1), xstride=0.01, ystride=0.01, levels=[0, 1, 2, 3, 5, 9, 25, 50, 100])
    for i in range(10):
        x_next = M.step(rosenbrock, rosenbrock.grad, x)
        plt.plot([x[0], x_next[0]], [x[1], x_next[1]], c="black")
        x = x_next
    plt.title("Figure 5.1")
    plt.show()

# TODO - Figure 5.2
# TODO - Figure 5.3
# TODO - Figure 5.5
# TODO - Figure 5.6
# TODO - Figure 5.7