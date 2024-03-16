import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm, multivariate_normal

from TestFunctions import ackley


def example_8_2():
    """
    Example 8.2: Exploring the effect of distribution variance and temperature
    on the performance of simulated annealing. The blue regions indicate the
    5% to 95% and 25% to 75% empirical Gaussian quantiles of the objective
    function value.
    """
    f = ackley
    x0 = np.array([15.0, 15.0])
    n_trials = 500
    k_max = 100
    iterations = np.arange(k_max + 1)

    _, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for p, sigma in enumerate([1.0, 5.0, 25.0]):
        T = multivariate_normal(np.zeros(2), sigma * np.eye(2))
        for q, t1 in enumerate([1.0, 10.0, 25.0]):
            def t(k, t1=t1): return t1 / k
            traj = np.zeros((n_trials, k_max + 1))

            # Run Trials
            for j in range(n_trials):
                # Simulated Annealing
                x = x0.copy()
                y = f(x)
                traj[j, 0] = y

                x_best, y_best = x, y
                for k in range(1, k_max + 1):
                    x_prime = x + T.rvs()
                    y_prime = f(x_prime)
                    delta_y = y_prime - y
                    if (delta_y <= 0) or (np.random.rand() < np.exp(-delta_y / t(k))):
                        x, y = x_prime, y_prime
                    if y_prime < y_best:
                        x_best, y_best = x_prime, y_prime
                    traj[j, k] = y
            
            # Plot the results
            traj_means = np.mean(traj, axis=0)
            traj_stds = np.std(traj, axis=0)
            quantiles = np.zeros((4, k_max + 1))
            for j in range(k_max + 1):
                quantiles[:, j] = norm(traj_means[j], traj_stds[j]).ppf(q=[0.05, 0.25, 0.75, 0.95])

            ax[p, q].fill_between(
                iterations,
                quantiles[3, :],
                quantiles[0, :],
                color="tab:blue",
                alpha=0.15
            )
            ax[p, q].fill_between(
                iterations,
                quantiles[2, :],
                quantiles[1, :],
                color="tab:blue",
                alpha=0.50
            )
            ax[p, q].plot(iterations, traj_means, color="tab:blue")
            ax[p, q].set_ylim((-5, 30))
            ax[p, q].set_xlim((0, k_max))
            ax[p, q].set_title("$\sigma = $" + str(int(sigma)) + ", $t^{(1)} = $" + str(int(t1)))
    for j in range(3):
        ax[2, j].set_xlabel("iteration")
        ax[j, 0].set_ylabel("$y$")
    plt.suptitle("Example 8.2")
    plt.show()


def example_8_3():
    """Example 8.3: An example of using the cross-entropy method."""
    # np.random
    pass

# TODO - Example 8.3
# TODO - Example 8.4