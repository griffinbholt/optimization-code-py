"""Chapter 6: Second-Order Methods"""

import numpy as np

from typing import Callable

from ch04 import line_search
from ch05 import DescentMethod


def newtons_method(grad_f: Callable[[np.ndarray], np.ndarray], 
                   H: Callable[[np.ndarray], np.ndarray], 
                   x: np.ndarray,
                   eps: float,
                   k_max: int) -> np.ndarray:
    """
    Newton's method, which takes the gradient of the function `grad_f`,
    the Hessian of the objective function `H`, an initial point `x`, a step size
    tolerance `eps`, and a maximum number of iterations `k_max`.
    """
    k, Delta = 0, np.full(len(x), np.inf)
    while (np.linalg.norm(Delta) > eps) and (k < k_max):
        Delta = np.linalg.solve(H(x), grad_f(x))
        x -= Delta
        k += 1
    return x


def secant_method(f_prime: Callable[[float], float], x0: float, x1: float, eps: float):
    """
    The secant method for univariate function minimization. The inputs are the
    first derivative `f_prime` of the target function, two initial points `x0`
    and `x1`, and the desired tolerance `eps`. The final x-coordinate is
    returned.
    """
    g0 = f_prime(x0)
    delta = np.inf
    while np.abs(delta) > eps:
        g1 = f_prime(x1)
        delta = ((x1 - x0) / (g1 - g0)) * g1
        x0, x1, g0 = x1, x1 - delta, g1
    return x1


class QuasiNewtonMethod(DescentMethod):
    """
    Just as the secant method approximates f'' in the univariate case,
    quasi-Newton methods approximate the inverse Hessian.
    """
    pass


class DFP(QuasiNewtonMethod):
    """The Davidon-Fletcher-Powell descent method"""
    def __init__(self, Q: np.ndarray):
        self.Q = Q  # approximate inverse Hessian

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        m = len(x)
        self.Q = np.eye(m)

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        x_prime = line_search(f, x, -self.Q @ g)
        g_prime = grad_f(x_prime)
        delta = x_prime - x
        gamma = g_prime - g
        self.Q -= self.Q_update(delta, gamma, self.Q @ gamma)
        return x_prime

    def Q_update(self, delta: np.ndarray, gamma: np.ndarray, Q_gamma: np.ndarray) -> np.ndarray:
        return (np.outer(Q_gamma, Q_gamma) / np.dot(Q_gamma, gamma)) - (np.outer(delta, delta) / np.dot(delta, gamma))


class BFGS(DFP):
    """
    The Broyden-Fletcher-Goldfarb-Shanno descent method
    
    NOTE: BFGS is the same as DFP, except for the `Q` update rule.
    """
    def __init__(self, Q: np.ndarray):
        super().__init__(Q)

    def Q_update(self, delta: np.ndarray, gamma: np.ndarray, Q_gamma: np.ndarray) -> np.ndarray:
        outer_dQg = np.outer(delta, Q_gamma)
        dot_dg = np.dot(delta, gamma)
        return ((outer_dQg + outer_dQg.T) / dot_dg)\
             - ((1 + (np.dot(Q_gamma, gamma) / dot_dg)) * (np.outer(delta, delta) / dot_dg))


class LimitedMemoryBFGS(QuasiNewtonMethod):
    """
    The Limited-memory BFGS descent method, which avoids storing the approximate
    inverse Hessian. The parameter `m` determines the history size. It also
    stores the step differences `deltas`, the gradient changes `gammas`, and
    storage vectors `qs`.
    """
    def __init__(self, m: int, deltas: list[np.ndarray], gammas: list[np.ndarray], qs: np.ndarray):
        self.m = m            # history size
        self.deltas = deltas  # step differences
        self.gammas = gammas  # gradient changes
        self.qs = qs          # storage vectors

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.deltas = []
        self.gammas = []
        self.qs = []

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        m = len(self.deltas)
        if m > 0:
            q = g
            for i in range(m - 1, -1, -1):
                self.qs[i] = q.copy()
                q -= (np.dot(self.deltas[i], q) / np.dot(self.gammas[i], self.deltas[i])) * self.gammas[i]
            z = (self.gammas[m - 1] * self.deltas[m - 1] * q) / np.dot(self.gammas[m - 1], self.gammas[m - 1])
            for i in range(m):
                z += self.deltas[i] * ((np.dot(self.deltas[i], self.qs[i]) - np.dot(self.gammas[i], z)) / np.dot(self.gammas[i], self.deltas[i]))
            x_prime = line_search(f, x, -z)
        else:
            x_prime = line_search(f, x, -g)
        g_prime = grad_f(x_prime)
        self.deltas.append(x_prime - x); self.gammas.append(g_prime - g); self.qs.append(np.zeros(len(x)))
        while len(self.deltas) > self.m:
            self.deltas.pop(0); self.gammas.pop(0); self.qs.pop(0)
        return x_prime
