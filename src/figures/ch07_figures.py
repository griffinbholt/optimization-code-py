import sys; sys.path.append('./src/'); sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from ch04 import line_search
from ch07 import basis
from TestFunctions import booth, wheeler
from convenience import plot_contour


def figure_7_1(n_steps: int = 6):
    """
    Figure 7.1: Cyclic coordinate descent alternates between coordinate directions.
    """
    x = np.array([10.0, -5.0])  # Starting point
    n = len(x)

    fig = plt.figure()
    plot_contour(fig, booth, xlim=(-10.5, 10.5), ylim=(-10.1, 10.1), xstride=0.01, ystride=0.01, levels=[0, 1, 5, 10, 20, 50, 100, 200, 500, 1000])
    for _ in range(n_steps):
        for i in range(n):
            d = basis(i, n)
            x_next = line_search(booth, x, d)
            plt.plot([x[0], x_next[0]], [x[1], x_next[1]], c="black")
            x = x_next
    plt.title("Figure 7.1")
    plt.show()


def figure_7_3(n_steps: int = 6):
    """
    Figure 7.3: Adding the acceleration step to cyclic coordinate descent helps
    traverse valleys. Six steps are shown for both the original and accelerated
    versions.
    """
    x = np.array([10.0, -5.0])  # Starting point (Original)
    x_accel = x.copy()          # Starting point (Accelerated)
    n = len(x)

    fig = plt.figure()
    plot_contour(fig, booth, xlim=(-10.5, 10.5), ylim=(-10.1, 10.1), xstride=0.01, ystride=0.01, levels=[0, 1, 5, 10, 20, 50, 100, 200, 500, 1000])
    for _ in range(n_steps):
        x_accel_prev = x_accel.copy()
        for i in range(n):
            d = basis(i, n)
            # Original
            x_next = line_search(booth, x, d)
            plt.plot([x[0], x_next[0]], [x[1], x_next[1]], c="tab:blue")
            x = x_next

            # Accelerated
            x_accel_next = line_search(booth, x_accel, d)
            plt.plot([x_accel[0], x_accel_next[0]], [x_accel[1], x_accel_next[1]], c="tab:red")
            x_accel = x_accel_next
        # Acceleration Step
        x_accel_next = line_search(booth, x_accel, x_accel - x_accel_prev)
        plt.plot([x_accel[0], x_accel_next[0]], [x_accel[1], x_accel_next[1]], c="tab:red")    
        x_accel = x_accel_next
    plt.legend(labels=["original", "accelerated"], loc="lower left")
    plt.title("Figure 7.3")
    plt.show()


def figure_7_4():
    """
    Figure 7.4: Powell's method starts the same as cyclic coordinate descent but
    iteratively learns conjugate directions.
    """
    x = np.array([10.0, -5.0])  # Starting point
    n = len(x)
    U = np.eye(n)

    fig = plt.figure()
    plot_contour(fig, wheeler, xlim=(-10.5, 10.5), ylim=(-10.1, 10.1), xstride=0.01, ystride=0.01, levels=[0, 1, 5, 10, 20, 50, 100, 200, 500, 1000])
    for _ in range(2):
        x_prime = x.copy()
        for i in range(n):
            d = U[i]
            x_prime_next = line_search(booth, x_prime, d)
            plt.plot([x_prime[0], x_prime_next[0]], [x_prime[1], x_prime_next[1]], c="black")
            x_prime = x_prime_next
        for i in range(n - 1):
            U[i] = U[i + 1]
        U[n - 1] = d = x_prime - x
        x_prime_next = line_search(booth, x_prime, d)
        plt.plot([x_prime[0], x_prime_next[0]], [x_prime[1], x_prime_next[1]], c="black")
        x = x_prime_next
    plt.title("Figure 7.4")
    plt.show()


def figure_7_5(n_steps: int = 4):
    """
    Figure 7.5: The Hooke-Jeeves method, proceeding left to right. It begins
    with a large step size but then reduces it once it cannot improve by taking
    a step in any coordinate direction.
    """
    alpha, gamma = 0.5, 0.5
    x = np.array([0.7, 0.9])  # Starting point
    y, n = wheeler(x), len(x)
    
    fig = plt.figure(figsize=(5*n_steps, 5))
    for i in range(1, n_steps + 1):
        cont_ax = plot_contour(fig, wheeler, xlim=(-0.1, 3.0), ylim=(-0.1, 3.0), xstride=0.01, ystride=0.01, levels=np.arange(-1.0, -0.0, 0.1), subplot_coords=(1,n_steps,i))
        cont_ax.scatter([x[0]], [x[1]], c='black', s=30.0)
        improved = False
        x_best, y_best = x, y
        for j in range(n):
            for sgn in [-1, 1]:
                x_prime = x + sgn*alpha*basis(j, n)
                cont_ax.scatter([x_prime[0]], [x_prime[1]], c='black', s=10.0, zorder=2)
                y_prime = wheeler(x_prime)
                if y_prime < y_best:
                    x_best, y_best, improved = x_prime, y_prime, True
        x, y = x_best, y_best
        if not improved:
            alpha *= gamma
    plt.suptitle("Figure 7.5", y=0.78)
    plt.subplots_adjust(wspace=0.25)
    plt.show()

# TODO - Figure 7.11
# TODO - Figure 7.12
# TODO - Figure 7.13
# TODO - Figure 7.14
# TODO - Figure 7.20