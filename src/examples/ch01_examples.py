import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

from TestFunctions import rosenbrock

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
        X1 = np.arange(-2.1, 2.1, 0.01)
        X2 = np.arange(-2.1, 2.1, 0.01)
        X1, X2 = np.meshgrid(X1, X2)
        Z = rosenbrock(np.array([X1, X2]))
        plt.contour(X1, X2, Z, levels=[1, 2, 3, 5, 9, 25, 50, 100], cmap=cm.viridis.reversed())
        plt.scatter([1], [1], c='black')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.gca().set_aspect('equal')
        plt.show()

def example_1_2():
    """Example 1.2: An example three-dimensional visualization and the associated contour plot"""
    def f(x1, x2): return x1**2 - x2**2

    # Make data
    X1 = np.arange(-2.1, 2.1, 0.05)
    X2 = np.arange(-2.1, 2.1, 0.05)
    X1, X2 = np.meshgrid(X1, X2)
    Z = f(X1, X2)

    # Surface Plot
    fig = plt.figure(figsize=(10, 5))
    cmap = cm.viridis.reversed()
    surf_ax = fig.add_subplot(1,2,1, projection='3d')
    surf_ax.plot_surface(X1, X2, Z, cmap=cmap)
    surf_ax.set_zlim(-5.1, 5.1) # Customize the z-axis
    surf_ax.set_xlabel('x1')
    surf_ax.set_ylabel('x2')
    
    # Contour Plot
    cont_ax = fig.add_subplot(1,2,2)
    CS = cont_ax.contour(X1, X2, Z, levels=[-4, -2, 0, 2, 4], cmap=cmap)
    cont_ax.clabel(CS, inline=True, fontsize=10)
    cont_ax.set_aspect('equal')
    cont_ax.set_xlabel('x1')
    cont_ax.set_ylabel('x2')

    plt.subplots_adjust(wspace=0.5)
    plt.show()
