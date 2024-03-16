"""Chapter 16: Surrogate Optimization"""

import numpy as np

from scipy.stats import norm
from typing import Callable

from ch15 import GaussianProcess


def prob_of_improvement(y_min: float, mu: float, sigma: float) -> float:
    """
    Computing the probability of improvement for a given best y value `y_min`,
    mean `mu`, and standard deviation `sigma`.
    """
    return norm(mu, sigma).cdf(y_min)


def expected_improvement(y_min: float, mu: float, sigma: float) -> float:
    """
    Computing the expected improvment for a given best y value `y_min`,
    mean `mu`, and standard deviation `sigma`.
    """
    p_imp = prob_of_improvement(y_min, mu, sigma)
    p_ymin = norm(mu, sigma).pdf(y_min)
    return (y_min - mu)*p_imp + (sigma**2)*p_ymin


class SafeOpt():
    """
    The SafeOpt algorithm applied to an empty Gaussian process `GP`, a finite
    design space `X`, index of initial safe point `i`, objective function `f`.
    and safety threshold `y_max`. The optional parameters are the confidence
    scalar `beta` and the number of iterations `k_max`. A tuple containing the
    best safe upper bound and its index in `X` is returned.
    """
    def __call__(self,
                 GP: GaussianProcess,
                 X: np.ndarray,
                 i: int,
                 f: Callable[[np.ndarray], float],
                 y_max: float,
                 beta: float = 3.0,
                 k_max: int = 10) -> tuple[np.ndarray, int]:
        GP.append(X[i], f(X[i]))

        m = len(X)
        u, l = np.full(m, np.inf), np.full(m, -np.inf)
        S, M, E = np.full(m, False), np.full(m, False), np.full(m, False)

        for _ in range(k_max):
            u, l = self.update_confidence_intervals(GP, X, u, l, beta)
            S, M, E = self.compute_sets(GP, S, M, E, X, u, l, y_max, beta)
            i = self.get_new_query_point(M, E, u, l)
            if i == 0:
                break
            GP.push(X[i], f(X[i]))
        
        # return the best point
        u, l = self.update_confidence_intervals(GP, X, u, l, beta)
        S = (u <= y_max)
        if np.any(S):
            i_best = np.argmin(u[S])
            u_best = u[S][i_best]
            i_best = np.where(i_best == np.cumsum(S))[0][0]
            return (u_best, i_best)
        return (None, 0)

    def update_confidence_intervals(self,
                                    GP: GaussianProcess,
                                    X: np.ndarray,
                                    u: np.ndarray,
                                    l: np.ndarray,
                                    beta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        A method for updating the lower and upper bounds used in SafeOpt, which
        takes the Gaussian process `GP`, the finite search space `X`, the upper-
        and lower-bound vectors `u` and `l`, and the confidence scalar `beta`.
        """
        mu_p, v_p = GP.predict(X)
        u = mu_p + np.sqrt(beta * v_p)
        l = mu_p - np.sqrt(beta * v_p)
        return (u, l)

    def compute_sets(self,
                     GP: GaussianProcess,
                     S: np.ndarray,
                     M: np.ndarray,
                     E: np.ndarray,
                     X: np.ndarray,
                     u: np.ndarray,
                     l: np.ndarray,
                     y_max: float,
                     beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A method for updating the safe `S`, minimizer `M`, and expander `E` sets
        used in SafeOpt. The sets are all Boolean vectors indicating whether the
        corresponding design point in `X` is in the set. The method also takes
        the Gaussian process `GP`, the upper and lower bounds `u` and `l`,
        respectively, the safety threshold `y_max`, and the confidence scalar
        `beta`.
        """
        M.fill(False)
        E.fill(False)

        # safe set
        S = (u <= y_max)

        if np.any(S):
            # potential minimizers
            M[S] = (l[S] < np.min(u[S]))

            # maximum width (in M)
            w_max = np.max(u[M] - l[M])

            # expanders - skip values in M or those with w <= w_max
            E = S & ~M  # skip points in M
            if np.any(E):
                E[E] = (np.max(u[E] - l[E]) > w_max)
                for (i, e) in enumerate(E):
                    if e and (u[i] - l[i] > w_max):
                        GP.append(X[i], l[i])
                        mu_p, v_p = GP.predict(X[~S])
                        GP.pop()
                        E[i] = np.any(mu_p + np.sqrt(beta * v_p) >= y_max)
                        if E[i]:
                            w_max = u[i] - l[i]

        return (S, M, E)

    def get_new_query_point(self, M: np.ndarray, E: np.ndarray, u: np.ndarray, l: np.ndarray) -> int:
        """
        A method for obtaining the next query point in SafeOpt. The index of the
        point in `X` with the greatest width is returned.
        """
        ME = M | E
        if np.any(ME):
            v = np.argmax(u[ME] - l[ME])
            return np.where(v == np.cumsum(ME))[0][0]
        return 0
