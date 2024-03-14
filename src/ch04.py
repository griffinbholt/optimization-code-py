"""Chapter 4: Local Descent"""

import cvxpy as cp
import numpy as np
import warnings

from scipy.optimize import brent
from typing import Callable

from ch03 import bracket_minimum

warnings.simplefilter(action='ignore', category=FutureWarning)


def line_search(f: Callable[[np.ndarray], float],
                x: np.ndarray,
                d: np.ndarray,
                minimize: Callable[[Callable, float, float], float] = lambda f,a,b: brent(f, brack=(a, b))
                ) -> np.ndarray:
    """
    A method for conducting a line search, which finds the optimal step factor
    along a descent direction `d` from design point `x` to minimize function `f`.
    The `minimize` function can be implemented using a univariate optimization
    algorithm such as the Brent-Dekker method.
    """
    def objective(alpha): return f(x + alpha*d)
    a, b = bracket_minimum(objective)
    alpha = minimize(objective, a, b)
    return x + alpha*d


def backtracking_line_search(f: Callable[[np.ndarray], float],
                             grad_f: Callable[[np.ndarray], np.ndarray],
                             x: np.ndarray,
                             d: np.ndarray,
                             alpha: float,
                             p: float = 0.5,
                             beta: float = 1e-4) -> float:
    """
    The backtracking line search algorithm, which takes objective function `f`,
    its gradient `grad_f`, the current design point `x`, a descent direction `d`,
    and the maximum step size `alpha`. We can optionally specify the reduction
    factor `p` and the first Wolfe condition parameter `beta`.
    """
    y, g = f(x), grad_f(x)
    while f(x + alpha*d) > y + beta*alpha*np.dot(g, d):
        alpha *= p
    return alpha


def strong_backtracking(f: Callable[[np.ndarray], float],
                        grad_f: Callable[[np.ndarray], np.ndarray],
                        x: np.ndarray,
                        d: np.ndarray,
                        alpha: float = 1.0,
                        beta: float = 1e-4,
                        sigma: float = 0.1) -> float:
    """
    Strong backtracking approximate line search for satisfying the strong Wolfe
    conditions. It takes as input the objective function `f`, the gradient
    function `grad_f`, the design point `x` and direction `d` from which line
    search is conducted, an initial step size `alpha`, and the Wolfe condition
    parameters `beta` and `sigma`. The algorithm's bracket phase first brackets
    an interval containing a step size that satisfies the strong Wolfe conditions.
    It then reduces this bracketed interval in the zoom phase until a suitable
    step size is found. We interpolate with bisection, but other schemes can be
    used.
    """
    y_0, g_0, y_prev, alpha_prev = f(x), np.dot(grad_f(x), d), np.nan, 0
    alpha_lo, alpha_hi = np.nan, np.nan

    # Bracket Phase
    while True:
        y = f(x + alpha*d)
        if (y > y_0 + beta*alpha*g_0) or ((not np.isnan(y_prev)) and (y >= y_prev)):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break
        g = np.dot(grad_f(x + alpha*d), d)
        if abs(g) <= -sigma*g_0:
            return alpha
        elif g >= 0:
            alpha_lo, alpha_hi = alpha, alpha_prev
            break
        y_prev, alpha_prev, alpha = y, alpha, 2*alpha

    # Zoom Phase
    y_lo = f(x + alpha_lo*d)
    while True:
        alpha = (alpha_lo + alpha_hi) / 2
        y = f(x + alpha*d)
        if (y > y_0 + beta*alpha*g_0) or (y >= y_lo):
            alpha_hi = alpha
        else:
            g = np.dot(grad_f(x + alpha*d), d)
            if abs(g) <= -sigma*g_0:
                return alpha
            elif g*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha


def solve_trust_region_subproblem(grad_f: Callable[[np.ndarray], np.ndarray],
                                  H: Callable[[np.ndarray], np.ndarray],
                                  x0: np.ndarray,
                                  delta: float) -> tuple[np.ndarray, float]:
    """We have provided an example implementation of `solve_trust_region_subproblem`
    that uses a second-order Taylor approximation about `x0` with a circular trust region."""
    x = cp.Variable(len(x0))
    objective = cp.Minimize((grad_f(x0) @ (x - x0)) + (cp.quad_form(x - x0, H(x0)) / 2))
    constraints = [cp.norm(x - x0) <= delta]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return (x.value, problem.value)


def trust_region_descent(f: Callable[[np.ndarray], float],
                         grad_f: Callable[[np.ndarray], np.ndarray],
                         H: Callable[[np.ndarray], np.ndarray],
                         x: np.ndarray,
                         k_max: int,
                         eta_1: float = 0.25,
                         eta_2: float = 0.5,
                         gamma_1: float = 0.5,
                         gamma_2: float = 2.0,
                         delta: float = 1.0,
                         solve_trust_region_subproblem: Callable[[Callable, Callable, np.ndarray, float], tuple[np.ndarray, float]] = solve_trust_region_subproblem
                         ) -> np.ndarray:
    """
    The trust region descent method, where `f` is the objective function,
    `grad_f` produces the derivative, `H` produces the Hessian, `x` is an initial
    design point, and `k_max` is the number of iterations. The optional parameters
    `eta_1` and `eta_2` determine when the trust region radius `delta` is increased
    or decreased, and `gamma_` and `gamma_2` control the magnitude of the change.
    An implementation for `solve_trust_region_subproblem` must be provided that
    solves equation (4.10) in the texbook.
    """
    y = f(x)
    for _ in range(k_max):
        x_prime, y_prime = solve_trust_region_subproblem(grad_f, H, x, delta)
        r = (y - f(x_prime)) / (y - y_prime)
        if r < eta_1:
            delta *= gamma_1
        else:
            x, y = x_prime, y_prime
            if r > eta_2:
                delta *= gamma_2
    return x
