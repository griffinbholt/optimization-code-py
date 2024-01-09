import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy


def example_21_1(f1, f2):
    """
    Example 21.1: Basic code syntax for the assignment-based representation of
    multidisciplinary design optimization problems.
    """
    def F1(A):
        A["y1"] = f1(A["x"], A["y2"])
        return A
    
    def F2(A):
        A["y2"] = f2(A["x"], A["y1"])
        return A

    A = {"x": 1, "y1": 2, "y2": 3}


def example_21_2():
    """
    Example 21.2: An example that illustrates the importance of choosing an
    appropriate ordering when running a multidisciplinary analysis.
    """
    def F1(A):
        A["y1"] = A["y2"] - A["x"]
        return A
    
    def F2(A):
        A["y2"] = np.sin(A["y1"] + A["y3"])
        return A
    
    def F3(A):
        A["y3"] = np.cos(A["x"] + A["y2"] + A["y1"])

    def gauss_seidel(Fs, A, k_max, eps=1e-4):
        """Gauss-Seidel Algorithm (from Chapter 21), altered for plotting convergence"""
        k, converged = 0, False
        history = {var: [val] for (var, val) in A.items() if var != "x"}
        while (not converged) and (k < k_max):
            k += 1
            A_old = deepcopy(A)
            for F in Fs:
                F(A)
            converged = np.all([np.isclose(A[v], A_old[v], rtol=eps) for v in A])
            for (var, val) in A.items():
                if var != "x":
                    history[var].append(val)
        return (A, history, converged)
    
    # Run two orderings for 20 iterations each and plot
    k_max = 20
    k = np.arange(0, k_max + 1)
    orderings = [[F1, F2, F3], [F1, F3, F2]]
    _, axs = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)
    for i, Fs in enumerate(orderings):
        A = {"x": 1.0, "y1": 1.0, "y2": 1.0, "y3": 1.0}
        A, history, _ = gauss_seidel(Fs, A, k_max)
        print(A)
        axs[i].plot(k, history["y1"], label="y1", c="tab:purple")
        axs[i].plot(k, history["y2"], label="y2", c="tab:blue")
        axs[i].plot(k, history["y3"], label="y3", c="tab:green")
        axs[i].scatter(k, history["y1"], c="tab:purple")
        axs[i].scatter(k, history["y2"], c="tab:blue")
        axs[i].scatter(k, history["y3"], c="tab:green")
        axs[i].set_yticks([-2, -1, 0, 1])
    axs[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    axs[1].set_xticks([0, 5, 10, 15, 20])
    axs[1].set_xlabel("iteration")
    plt.tight_layout()
    plt.show()
