"""Chapter 18: Uncertainty Propagation"""

import numpy as np

from itertools import product
from typing import Callable

from ch15 import K, GaussianProcess

# TODO - taylor_approx (need forward difference gradient and Hessian library)
# TODO - legendre, laguerre, hermite polynomials (need to understand what Polynomials library does)
# TODO - orthogonal_recurrence

def polynomial_chaos_bases(bases1d: list[Callable[[float], float]]) -> list[Callable[[float], float]]:
    """
    A method for constructing multivariate basis functions where `bases1d` contains
    lists of univariate orthogonal basis functions for each random variable.
    """
    bases = []
    for a in product(*bases1d):
        bases.append(lambda z: np.prod([b(z[i]) for (i, b) in enumerate(a)]))
    return bases


def bayesian_monte_carlo(GP: GaussianProcess,
                         w: np.ndarray,
                         mu_z: np.ndarray,
                         Sigma_z: np.ndarray) -> tuple[float, float]:
    """
    A method for obtaining the Bayesian Monte Carlo estimate for the expected
    value of a function under a Gaussian process `GP` with a Gaussian kernel
    with weights `w`, where the variables are drawn from a normal distribution
    with mean `mu_z` and covariance `Sigma_z`.
    """
    W = np.diag(w**2)
    invK = np.linalg.inv(K(GP.X, GP.X, GP.k))
    q = np.exp(-(np.dot(GP.X - mu_z, np.linalg.inv(W + Sigma_z @ (GP.X - mu_z)))) / 2)  # TODO - Need to check/test dimensions
    q *= np.linalg.det((1/W) @ Sigma_z + np.eye(len(w)))**(-0.5)
    mu = np.dot(q, invK @ GP.y)
    v = np.linalg.det(2 * (1/W) @ Sigma_z + np.eye(len(w)))**(-0.5) - np.dot(q, invK @ q)[0]
    return (mu, v)
