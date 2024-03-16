import sys; sys.path.append('./src/'); sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from scipy.stats import norm, cauchy

from ch09 import rand_population_uniform, rand_population_normal, rand_population_cauchy,\
                 TruncationSelection, TournamentSelection, RouletteWheelSelection
from convenience import normalize


def figure_9_1():
    """
    Figure 9.1: A comparison of the normal distribution with standard deviation
    1 and the Cauchy distribution with scale 1. Although `sigma` is sometimes
    used for the scale parameter in the Cauchy distribution, this should not be
    confused with the standard deviation since the standard deviation of the
    Cauchy distribution is undefined. The Cauchy distribution is heavy-tailed,
    allowing it to cover the design space more broadly.
    """
    x = np.linspace(-6, 6, 1000)
    plt.plot(x, norm(loc=0, scale=1).pdf(x), c='tab:purple', label="Normal")
    plt.plot(x, cauchy(loc=0, scale=1).pdf(x), c='tab:blue', label="Cauchy")
    plt.xlabel("$x$")
    plt.ylabel("$p(x)$")
    plt.title("Figure 9.1")
    plt.xticks([-5, 0, 5])
    plt.yticks([0.0, 0.2, 0.4])
    plt.legend()
    plt.show()


def figure_9_2():
    """
    Figure 9.2: Initial populations of size 1,000 sampled using a uniform
    hyperrectangle with a = [-2, -2], b = [2, 2], a zero-mean normal distribution
    with diagonal covariance Sigma = I, and Cauchy distributions centered at the
    origin with scale sigma = 1.
    """
    m = 1000
    s = 10.0
    alpha = 0.55
    _, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    population = rand_population_uniform(m, a=np.array([-2.0, -2.0]), b=np.array([2.0, 2.0]))
    ax[0].scatter(population[:, 0], population[:, 1], s=s, alpha=alpha)
    ax[0].set_title("Uniform")
    
    population = rand_population_normal(m, mu=np.zeros(2), Sigma=np.eye(2))
    ax[1].scatter(population[:, 0], population[:, 1], s=s, alpha=alpha)
    ax[1].set_title("Normal")
    
    population = rand_population_cauchy(m, mu=np.zeros(2), sigma=np.ones(2))
    ax[2].scatter(population[:, 0], population[:, 1], s=s, alpha=alpha)
    ax[2].set_title("Cauchy")

    for i in range(3):
        ax[i].set_xlabel("$x_1$")
        ax[i].set_ylabel("$x_2$")
        ax[i].set_xlim(-4, 4)
        ax[i].set_ylim(-4, 4)
        ax[i].set_aspect('equal')
    plt.suptitle("Figure 9.2")
    plt.show()


def figure_9_4():
    """
    Figure 9.4: Truncation selection with a population size `m = 7` and sample
    size `k = 3`. The height of a bar indicates its objective function value
    whereas its color indicates what individual it corresponds to.
    """
    x, y, m, ax, colors = selection_setup()

    # Truncation Selection (taken directly from ch09.py) 
    k = 3
    p = np.argsort(y)
    new_colors = colors[p]
    new_colors[k:] = np.array([192, 192, 192, 255.0]) / 255.0
    ax[1].bar(x, y[p], width=1/(m - 1), color=new_colors, edgecolor='black')
    ax[1].tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set_ylim(0.0, 1.3)
    ax[1].set_xlabel("individual")
    ax[1].set_ylabel("$y$")
    plt.suptitle("Figure 9.4")
    plt.show()


def figure_9_5():
    """
    Figure 9.5: Tournament selection with a population size `m = 7` and a sample
    size `k = 3`, which is run separately for each parent. The height of a bar
    indicates its objective function value whereas its color indicates what
    individual it corresponds to.
    """
    x, y, m, ax, colors = selection_setup()

    # Tournament Selection (taken directly from ch09.py)
    k = 3
    def getparent():
        p = np.random.permutation(len(y))
        return p[np.argmin(y[p[:k]])]
    p = [getparent() for _ in range(k)]
    new_colors = np.array([[192, 192, 192, 255.0] for _ in range(m)]) / 255.0
    new_colors[p, :] = colors[p, :]
    ax[1].bar(x, y, width=1/(m - 1), color=new_colors, edgecolor='black')
    ax[1].tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set_ylim(0.0, 1.3)
    ax[1].set_xlabel("individual")
    ax[1].set_ylabel("$y$")
    plt.suptitle("Figure 9.5")
    plt.show()


def figure_9_6():
    """
    Figure 9.6: Roulette wheel selection with a population size `m = 7`, which
    is run separately for each parent. The approach used causes the individual
    with the worst objective function value to have a zero likelihood of being
    selected. The height of a bar indicates its objective function value (left),
    or its likelihood (right), whereas its color indicates what individual it
    corresponds to.
    """
    x, y, m, ax, colors = selection_setup()

    # Roulette Wheel Selection (taken directly from ch09.py)
    y = np.max(y) - y
    p = normalize(y, ord=1)
    ax[1].bar(x, p, width=1/(m - 1), color=colors, edgecolor='black')
    ax[1].tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set_ylim(0.0, 0.6)
    ax[1].set_xlabel("individual")
    ax[1].set_ylabel("likelihood")
    plt.suptitle("Figure 9.6")
    plt.show()


def selection_setup():
    m = 7
    y = np.array([1.0, 0.6, 0.2, 1.0, 0.9, 0.6, 1.1])
    x = np.linspace(0.0, 1.0, m)
    colors = cm.viridis(x)

    _, ax = plt.subplots(1, 2, figsize=(6, 2))
    ax[0].bar(x, y, width=1/(m - 1), color=colors, edgecolor='black')
    ax[0].tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set_ylim(0.0, 1.3)
    ax[0].set_xlabel("individual")
    ax[0].set_ylabel("$y$")

    return x, y, m, ax, colors


def figure_9_7():
    """Figure 9.7: Single-point crossover"""
    a, b, x, color = crossover_setup()

    # Single-Point Crossover (taken directly from ch09.py)
    i = np.random.randint(len(a))
    child = np.concatenate((a[:i], b[i:]))
    plt.scatter(x, 0.0*x - 0.3, color=color(child))
    plt.xticks([i - 0.5], labels=["crossover point"])
    plt.subplots_adjust(bottom=0.4)
    plt.title("Figure 9.7", y=0.9)
    plt.show()


def figure_9_8():
    """Figure 9.8: Two-point crossover"""
    a, b, x, color = crossover_setup()

    # Two-Point Crossover (taken directly from ch09.py)
    n = len(a)
    i, j = np.random.randint(n, size=2)
    if i > j:
        i, j = j, i
    child = np.concatenate((a[:i], b[i:j], a[j:]))
    plt.scatter(x, 0.0*x - 0.3, color=color(child))
    plt.xticks([i - 0.5, j - 0.5], labels=["crossover point 1", "crossover point 2"])
    plt.subplots_adjust(bottom=0.4)
    plt.title("Figure 9.8", y=0.9)
    plt.show()


def figure_9_9():
    """Figure 9.9: Uniform crossover"""
    a, b, x, color = crossover_setup()

    # Uniform Crossover (taken directly from ch09.py)
    child = np.copy(a)
    for i in range(len(a)):
        if np.random.rand() < 0.5:
            child[i] = b[i]
    plt.scatter(x, 0.0*x - 0.3, color=color(child))
    plt.tick_params(axis="x", which="both", bottom=False)
    plt.xticks([])
    plt.title("Figure 9.9", y=0.9)
    plt.show()


def crossover_setup():
    n = 45
    x = np.arange(n)
    a, b = np.zeros(n), np.ones(n)
    def color(x): return ['tab:red' if x_i == 1 else 'tab:blue' for x_i in x]

    plt.figure(figsize=(10, 2.0))
    plt.scatter(x, 0.0*x, color=color(a))
    plt.scatter(x, 0.0*x - 0.15, color=color(b))
    plt.ylim(-0.5, 0.2)
    plt.yticks([0.0, -0.15, -0.3], labels=["parent A", "parent B", "child"])
    plt.tick_params(axis="y", which="both", left=False)
    plt.gca().spines[['left', 'bottom', 'right', 'top']].set_visible(False)

    return a, b, x, color


def figure_9_10():
    """
    Figure 9.10: Mutation for binary string chromosomes gives each bit a
    small probability of flipping.
    """
    n = 45
    lam = 1/n
    x = np.arange(n)
    before = np.zeros(n).astype(bool)
    after = np.array([~v if np.random.rand() < lam else v for v in before])
    def color(x): return ['lightgreen' if x_i == 1 else 'tab:blue' for x_i in x]

    plt.figure(figsize=(10, 1.5))
    plt.scatter(x, 0.0*x, color=color(before))
    plt.scatter(x, 0.0*x - 0.15, color=color(after))
    plt.ylim(-0.35, 0.2)
    plt.xticks([])
    plt.yticks([0.0, -0.15], labels=["before mutation", "after mutation"])
    plt.tick_params(axis="both", which="both", left=False, bottom=False)
    plt.gca().spines[['left', 'bottom', 'right', 'top']].set_visible(False)
    plt.title("Figure 9.10")
    plt.tight_layout()
    plt.show()


# TODO - Figure 9.11
# TODO - Figure 9.13
# TODO - Figure 9.14
# TODO - Figure 9.15
# TODO - Figure 9.16
