import sys; sys.path.append('./src/'); sys.path.append('../')

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

def figure_7_5_gps(n_steps: int = 4):
    """
    Similar to Figure 7.5, but Generalized Pattern Search is used instead of
    the Hooke-Jeeves Method
    """
    alpha, gamma = 0.5, 0.5
    D = np.array([[1, 0], [0, 1], [-1, -1]])  # positive spanning set
    x = np.array([0.7, 0.9])  # Starting point
    y = wheeler(x)
    
    fig = plt.figure(figsize=(5*n_steps, 5))
    for i in range(1, n_steps + 1):
        cont_ax = plot_contour(fig, wheeler, xlim=(-0.1, 3.0), ylim=(-0.1, 3.0), xstride=0.01, ystride=0.01, levels=np.arange(-1.0, -0.0, 0.1), subplot_coords=(1,n_steps,i))
        cont_ax.scatter([x[0]], [x[1]], c='black', s=30.0)
        improved = False
        for j, d in enumerate(D):
            x_prime = x + alpha * d
            cont_ax.scatter([x_prime[0]], [x_prime[1]], c='black', s=10.0, zorder=2)
            y_prime = wheeler(x_prime)
            if y_prime < y:
                x, y, improved = x_prime, y_prime, True
                D = np.insert(np.delete(D, j, axis=0), 0, d, axis=0)
                break
        if not improved:
            alpha *= gamma
    plt.suptitle("Figure 7.5 (w/ Generalized Pattern Search)", y=0.78)
    plt.subplots_adjust(wspace=0.25)
    plt.show()

def figure_7_11():
    """
    Figure 7.11: The Nelder-Mead method, proceeding left to right and top to bottom.
    """
    S = np.array([[0.7, 1.4], [0.7, 0.9], [0.4, 0.7]])
    triangles = [S.copy()]
    f = wheeler
    alpha, beta, gamma = 1.0, 2.0, 0.5

    fig = plt.figure(figsize=(20, 15))
    y_arr = np.apply_along_axis(f, 1, S)
    for j in range(1, 12 + 1):
        cont_ax = plot_contour(fig, wheeler, xlim=(-0.1, 3.0), ylim=(-0.1, 3.0), xstride=0.01, ystride=0.01, levels=np.arange(-1.0, -0.0, 0.1), subplot_coords=(3,4,j))
        cont_ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        cont_ax.set_xlabel(None)
        cont_ax.set_ylabel(None)
        for triangle in triangles:
            cont_ax.add_patch(Polygon(triangle, fill=False, ec="black"))
        p = np.argsort(y_arr)         # sort lowest to highest
        S, y_arr = S[p], y_arr[p]
        xl, yl = S[0], y_arr[0]       # lowest
        xh, yh = S[-1], y_arr[-1]     # highest
        xs, ys = S[-2], y_arr[-2]     # second-highest
        xm = np.mean(S[:-1], axis=0)  # centroid
        xr = xm + alpha * (xm - xh)   # reflection point
        yr = f(xr)

        if yr < yl:
            xe = xm + beta * (xr - xm)   # expansion point
            ye = f(xe)
            S[-1], y_arr[-1] = (xe, ye) if ye < yr else (xr, yr)
        elif yr >= ys:
            if yr < yh:
                xh, yh, S[-1], y_arr[-1] = xr, yr, xr, yr
            xc = xm + gamma * (xh - xm)  # contraction point
            yc = f(xc)
            if yc > yh:
                for i in range(1, len(y_arr)):
                    S[i] = (S[i] + xl) / 2
                    y_arr[i] = f(S[i])
            else:
                S[-1], y_arr[-1] = xc, yc
        else:
            S[-1], y_arr[-1] = xr, yr
        triangles.append(S.copy())
    plt.suptitle("Figure 7.11", fontsize=20, y=0.91)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()

def figure_7_20():
    raise NotImplementedError  # TODO
