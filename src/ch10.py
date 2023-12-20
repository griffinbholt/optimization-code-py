"""Chapter 10: Constraints"""

import numpy as np

from typing import Callable

def penalty_method(f: Callable[[np.ndarray], float],
                   minimize: Callable[[Callable, np.ndarray], np.ndarray],
                   p: Callable[[np.ndarray], float],
                   x: np.ndarray,
                   k_max: int,
                   rho: float = 1.0,
                   gamma: float = 2.0) -> np.ndarray:
    """
    The penalty method for objective function `f`, penalty function `p`, initial
    point `x`, number of iterations `k_max`, initial penalty `rho` > 0, and
    penalty multiplier `gamma` > 1. The method `minimize` should be replaced
    with a suitable unconstrained minimization method.
    """
    for _ in range(k_max):
        x = minimize(lambda x: f(x) + rho * p(x), x)
        p *= gamma
        if p(x) == 0:
            return x
    return x


def augmented_lagrange_method(f: Callable[[np.ndarray], float],
                              h: Callable[[np.ndarray], np.ndarray],
                              minimize: Callable[[Callable, np.ndarray], np.ndarray],
                              x: np.ndarray,
                              k_max: int,
                              rho: float = 1.0,
                              gamma: float = 2.0) -> np.ndarray:
    """
    The augmented Lagrange method for objective function `f`, equality constraint
    function `h`, initial point `x`, number of iterations `k_max`, initial penalty
    `rho` > 0, and penalty multiplier `gamma` > 1. The function `minimize`
    should be replaced with the minimization method of your choice.
    """
    lam = np.zeros(len(h(x)))
    for _ in range(k_max):
        def p(x): return ((rho/2) * np.sum(h(x)**2)) - np.dot(lam, h(x))
        x = minimize(lambda x: f(x) + p(x), x)
        lam -= rho * h(x)
        rho *= gamma
    return x


def interior_point_method(f: Callable[[np.ndarray], float],
                          p: Callable[[np.ndarray], float],
                          minimize: Callable[[Callable, np.ndarray], np.ndarray],
                          x: np.ndarray,
                          rho: float = 1.0,
                          gamma: float = 2.0,
                          eps: float = 0.001) -> np.ndarray:
    """
    The interior point method for objective function `f`, barrier function `p`,
    initial point `x`, initial penalty `rho` > 0, penalty multiplier `gamma` > 1,
    and stopping tolerance `eps` > 0.
    """
    delta = np.inf
    while delta > eps:
        x_prime = minimize(lambda x: f(x) + (p(x) / rho), x)
        delta = np.linalg.norm(x_prime - x)
        x = x_prime
        rho *= gamma
    return x
