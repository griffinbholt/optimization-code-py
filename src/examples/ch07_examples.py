import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from ch07 import Interval, Intervals, reparametrize_to_unit_hypercube, rev_unit_hypercube_parametrization

# TODO - Example 7.1
# TODO - Example 7.2

def example_7_1(eps: float = 1e-5, k_max: int = 5): # TODO - Hitting some snags
    """Example 7.1: The DIRECT method applied to a univariate function."""
    def f(x): return np.sin(x) + np.sin(2*x) + np.sin(4*x) + np.sin(8*x)
    a = np.array([-2.0])
    b = np.array([2.0])

    g = reparametrize_to_unit_hypercube(f, a, b)
    intervals = Intervals()
    n = len(a)
    c = np.full(n, 0.0)
    interval = Interval(c, g(c), np.zeros(n))
    intervals.add_interval(interval)
    c_best, y_best = np.copy(interval.c), interval.y

    fig, ax = plt.subplots(k_max + 1, 2, sharey=True, figsize=(7, 9))
    t = np.linspace(-2.0, 2.0, 1000)
    f_t = f(t)
    ax[0, 0].plot(t, f_t, c="black")
    ax[0, 0].hlines([f(c)], xmin=-2.0, xmax=2.0, color="tab:blue")
    ax[0, 0].scatter([interval.c], [f(c)], color="tab:blue")
    ax[0, 0].set_xlim(-2.0, 2.0)
    ax[0, 1].scatter([2.0], [f(c)], color="tab:blue")
    ax[0, 1].set_xlim(0.0, 2.0)

    for i in range(1, k_max + 1):
        ax[i, 0].plot(t, f_t, color="black")
        ax[i, 0].set_xlim(-2.0, 2.0)
        ax[i, 1].set_xlim(0.0, 2.0)
        S = intervals.get_opt_intervals(eps, y_best)
        to_add = []
        for interval in S:
            new_intervals = interval.divide(g)
            to_add.extend(new_intervals)
            intervals[interval.vertex_dist()].get()
        for interval in to_add:
            c = rev_unit_hypercube_parametrization(interval.c, a, b)
            u = rev_unit_hypercube_parametrization(interval.c + (3.0**(-i)), a, b)
            l = rev_unit_hypercube_parametrization(interval.c - (3.0**(-i)), a, b)
            ax[i, 0].hlines([f(c)], xmin=l, xmax=u, color="gray")
            ax[i, 0].scatter([c], [f(c)], color="black")
            intervals.add_interval(interval)
            if interval.y < y_best:
                c_best, y_best = np.copy(interval.c), interval.y
    
    x = rev_unit_hypercube_parametrization(c_best, a, b)

    plt.ylim(-2.5, 2.5)
    plt.show()

example_7_1()