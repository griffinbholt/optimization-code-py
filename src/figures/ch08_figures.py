import sys; sys.path.append('./src/'); sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from ch05 import GradientDescent
from ch08 import NoisyDescent, rand_positive_spanning_set
from TestFunctions import wheeler
from convenience import plot_contour

def figure_8_1():
    """
    Figure 8.1: Adding stochasticity to a descent method helps with traversing
    saddle points such as f(x) = x1^2 - x2^2 shown here. Due to the
    initialization, the steepest descent method converges to the saddle point
    where the gradient is zero.
    """
    def f(x): return x[0]**2 - x[1]**2
    def grad_f(x): return np.array([2*x[0], -2*x[1]])

    alpha = 0.1
    x_gd = np.array([2.0, 0.0])
    x_sgd = x_gd.copy()
    GD = GradientDescent(alpha)
    SGD = NoisyDescent(GradientDescent(alpha), sigma=lambda k: 1/(k**3))

    fig = plt.figure()
    lim = (-2.5, 2.5)
    plot_contour(fig, f, xlim=lim, ylim=lim, xstride=0.01, ystride=0.01, levels=[-5, -2, 0, 2, 5])
    for _ in range(20):
        x_sgd_next = SGD.step(f, grad_f, x_sgd)
        plt.plot([x_sgd[0], x_sgd_next[0]], [x_sgd[1], x_sgd_next[1]], c="tab:red")
        x_sgd = x_sgd_next

        x_gd_next = GD.step(f, grad_f, x_gd)
        plt.plot([x_gd[0], x_gd_next[0]], [x_gd[1], x_gd_next[1]], c="tab:blue")
        x_gd = x_gd_next
    plt.xlim(lim)
    plt.ylim(lim)
    plt.legend(labels=["stochastic gradient descent", "steepest descent"])
    plt.title("Figure 8.1")
    plt.show()


def figure_8_2():
    """
    Figure 8.2: Mesh adaptive direct search proceeding left to right and top to bottom.
    """
    x = np.array([1.5, 1.5])
    spanning_sets = []
    alpha, y, n = 1.0, wheeler(x), len(x)

    fig = plt.figure(figsize=(20, 10))
    lim = (-0.1, 3.0)
    for j in range(1, 8 + 1):
        # Set up contour plot
        cont_ax = plot_contour(fig, wheeler, xlim=lim, ylim=lim, xstride=0.01, ystride=0.01, levels=np.arange(-1.0, -0.0, 0.1), subplot_coords=(2,4,j))
        cont_ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        cont_ax.set_xlabel(None)
        cont_ax.set_ylabel(None)
        cont_ax.set_xlim(lim)
        cont_ax.set_ylim(lim)

        improved = False
        D = rand_positive_spanning_set(alpha, n)
        
        # Plot spanning sets
        spanning_sets.append((alpha, x.copy(), D.copy()))
        for (k, (alpha_tmp, x_tmp, spanning_set)) in enumerate(spanning_sets):
            cont_ax.scatter([x_tmp[0]], [x_tmp[1]], c='black', s=30.0, zorder=2, alpha=0.5**(len(spanning_sets) - k - 1))
            for d in spanning_set:
                x_prime = x_tmp + alpha_tmp * d
                cont_ax.scatter([x_prime[0]], [x_prime[1]], c='black', s=10.0, zorder=2, alpha=0.5**(len(spanning_sets) - k - 1))
                cont_ax.plot([x_tmp[0], x_prime[0]], [x_tmp[1], x_prime[1]], c='black', zorder=2, alpha=0.5**(len(spanning_sets) - k - 1))
        
        # Mesh Adaptive Direct Search Algorithm
        for d in D:
            x_prime = x + alpha * d
            y_prime = wheeler(x_prime)
            if y_prime < y:
                x, y, improved = x_prime, y_prime, True
                x_prime = x + 3 * alpha * d
                y_prime = wheeler(x_prime)
                if y_prime < y:
                    x, y = x_prime, y_prime
                break
        alpha = np.minimum(4 * alpha, 1.0) if improved else alpha / 4
    plt.suptitle("Figure 8.2", fontsize=15, y=0.86)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def figure_8_3():
    """
    Figure 8.3: Several annealing schedules commonly used in simulated annealing.
    The schedules have an initial temperature of 10.
    """
    def logarithmic(k, t1): return t0 * np.log(2) / np.log(k + 1)
    def exponential(k, gamma, t1): return (gamma**(k - 1)) * t1
    def fast(k, t1): return t1 / k

    t0 = 10.0
    max_iters = 10000
    k = np.linspace(1, max_iters, 100000)

    plt.plot(k, [logarithmic(k[i], t0) for i in range(len(k))], color="tab:red", label="logarithmic")
    plt.plot(k, [exponential(k[i], 0.25, t0) for i in range(len(k))], color="tab:blue", alpha=1.0, label="exponential, $\gamma = 1/4$")
    plt.plot(k, [exponential(k[i], 0.5, t0) for i in range(len(k))], color="tab:blue", alpha=0.75, label="exponential, $\gamma = 1/2$")
    plt.plot(k, [exponential(k[i], 0.75, t0) for i in range(len(k))], color="tab:blue", alpha=0.5, label="exponential, $\gamma = 3/4$")
    plt.plot(k, [fast(k[i], t0) for i in range(len(k))], color="tab:green", label="fast")
    plt.xlim((1, max_iters))
    plt.xscale('log')
    plt.xlabel("iteration")
    plt.ylabel("temperature")
    plt.title("Figure 8.3")
    plt.legend()
    plt.show()


def figure_8_4():
    """
    Figure 8.4: The step multiplication factor as a function of acceptance for
    c = 2.
    """
    def factor(x, c):
        if x > 0.6:
            return 1 + c*((x - 0.6)/0.4)
        elif x < 0.4:
            return 1/(1 + c*((0.4 - x)/0.4))
        return 1.0

    c = 2
    x = np.linspace(0.0, 1.0, 1000)
    plt.plot(x, [factor(x_i, c=c) for x_i in x])
    plt.xlim((0, 1))
    plt.ylim((0, 1 + c + 0.1))
    plt.xticks([0.0, 0.4, 0.6, 1.0])
    plt.yticks([1/(1 + c), 1.0, 1 + c], labels=["$\\frac{1}{1 + c}$", "$1$", "$1 + c$"])
    plt.title("Figure 8.4")
    plt.show()

# TODO - Figure 8.5
# TODO - Figure 8.6
# TODO - Figure 8.7
# TODO - Figure 8.8
# TODO - Figure 8.9