import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from TestFunctions import rosenbrock
from convenience import plot_surface, plot_contour

def example_1_1(display_contour_plot=False):
    """Example 1.1: Checking the first- and second-order necessary conditions
    of a point on the Rosenbrock function. The minimizer is indicated by the
    dot in the figure (when `display_contour_plot=True`)."""
    x = np.array([1.0, 1.0])

    print("Gradient at [1, 1]:")
    print(rosenbrock.grad(x))
    print()
    print("Hessian at [1, 1]:")
    print(rosenbrock.hess(x))

    if display_contour_plot:
        fig = plt.figure()
        plot_contour(fig,
                     rosenbrock, 
                     xlim=(-2.1, 2.1), 
                     ylim=(-2.1, 2.1), 
                     xstride=0.01, 
                     ystride=0.01, 
                     levels=[1, 2, 3, 5, 9, 25, 50, 100])
        plt.scatter([1], [1], c='black')
        plt.show()

def example_1_2():
    """Example 1.2: An example three-dimensional visualization and the associated contour plot"""
    def f(x): return x[0]**2 - x[1]**2

    fig = plt.figure(figsize=(10, 5))
    plot_surface(fig,
                 f,
                 xlim=(-2.1, 2.1),
                 ylim=(-2.1, 2.1),
                 zlim=(-5.1, 5.1),
                 xstride=0.05,
                 ystride=0.05,
                 subplot_coords=(1,2,1))
    plot_contour(fig,
                 f,
                 xlim=(-2.1, 2.1),
                 ylim=(-2.1, 2.1),
                 xstride=0.05,
                 ystride=0.05,
                 levels=[-4, -2, 0, 2, 4],
                 clabel=True,
                 subplot_coords=(1,2,2))
    plt.subplots_adjust(wspace=0.5)
    plt.show()
