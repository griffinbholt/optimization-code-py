"""Chapter 15: Probabilistic Surrogate Models"""

import numpy as np

from typing import Callable

from Distributions import MvNormal


def mu(X: np.ndarray, m: Callable[[np.ndarray], float]) -> np.ndarray:
    """
    A method for constructing a mean vector given a list of design points `X` 
    and a mean function `m`.
    """
    return np.apply_along_axis(m, 1, X)


def Sigma(X: np.ndarray, k: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """
    A method for constructing a covariance matrix given one list of design
    points `X` and a covariance function `k`.
    """
    return np.ndarray([[k(x, x_prime) for x_prime in X] for x in X])


def K(X: np.ndarray, X_prime: np.ndarray, k: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """
    A method for constructing a covariance matrix given two lists of design
    points `X` and `X_prime`, and a covariance function `k`.
    """
    return  np.ndarray([[k(x, x_prime) for x_prime in X_prime] for x in X])


def mvnrand(mu: np.ndarray, Sigma: np.ndarray, inflation: float = 1e-6) -> np.ndarray:
    N = MvNormal(mu, Sigma + inflation*np.eye(len(mu)))
    return N.rand()

class GaussianProcess():
    def __init__(self, 
                 m: Callable[[np.ndarray], float],
                 k: Callable[[np.ndarray, np.ndarray], float],
                 X: np.ndarray,
                 y: np.ndarray,
                 v: float):
        self.m = m  # mean
        self.k = k  # covariance function
        self.X = X  # design points
        self.y = y  # objective values
        self.v = v  # noise variance

    def rand(self, X: np.ndarray) -> np.ndarray:
        return mvnrand(mu(X, self.m), Sigma(X, self.k))

    def predict(self, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, k, v = self.m, self.k, self.v
        tmp = np.linalg.solve(K(X_pred, self.X, k), K(self.X, self.X, k) + v * np.eye(len(self.X)))
        mu_p = mu(X_pred, m) + tmp @ (self.y - mu(self.X, m))
        S = K(X_pred, X_pred, k) - tmp @ K(self.X, X_pred, k)
        v_p = np.diag(S) + np.finfo(np.float64).eps  # eps prevents numerical issues
        return (mu_p, v_p)

    def append(self, x: np.ndarray, y: float):
        if len(self.X) == 0:
            self.X = np.array([x])
            self.y = np.array([y])
        else:
            self.X = np.append(self.X, x)
            self.y = np.append(self.y, y)

    def pop(self) -> tuple[np.ndarray, float]:
        popped_x = self.X[-1]
        popped_y = self.y[-1]
        self.X = np.delete(self.X, -1)
        self.y = np.delete(self.y, -1)
        return (popped_x, popped_y)