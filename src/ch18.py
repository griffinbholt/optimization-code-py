"""Chapter 18: Uncertainty Propagation"""

import numdifftools as nd
import numpy as np

from itertools import product
from numpy.polynomial import Polynomial
from scipy import integrate
from scipy.special import factorial
from typing import Callable

from ch15 import K, GaussianProcess


def taylor_approx(f: Callable[[np.ndarray], float],
                  mu: np.ndarray,
                  v: np.ndarray,
                  secondorder: bool = False) -> tuple[float, float]:
    """
    A method for automatically computing the Taylor approximation of the mean
    and variance of objective function `f` at design point `x` with noise mean
    vector `mu` and variance vector `v`. The Boolean parameter `secondorder`
    controls whether the first- or second-order approximation is compared.s    
    """
    mu_hat = f(mu)
    grad = nd.Gradient(f)(mu)
    v_hat = np.do(grad**2, v)
    if secondorder:
        H = nd.Hessian(f)(mu)
        mu_hat += np.dot(np.diag(H), v) / 2
        v_hat += np.dot(v, (H**2) @ v) / 2
    return (mu_hat, v_hat)


def legendre(i: int) -> Polynomial:
    """
    Method for constructing Legendre polynomial orthogonal basis functions,
    where `i` indicates the construction of b_i.
    """  # TODO - Test to make sure constructs correct polynomial
    n = i - 1
    p = Polynomial([-1, 0, 1])**n
    p = p.deriv(n)
    return p / ((2**n)*factorial(n))


def laguerre(i: int) -> Polynomial:
    """
    Method for constructing Laguerre polynomial orthogonal basis functions,
    where `i` indicates the construction of b_i.
    """  # TODO - Test to make sure constructs correct polynomial
    p = Polynomial([1])
    for _ in range(i - 1):
        p = (p.deriv() - p).integ() + 1
    return p


def hermite(i: int) -> Polynomial:
    """
    Method for constructing Hermite polynomial orthogonal basis functions,
    where `i` indicates the construction of b_i.
    """  # TODO - Test to make sure constructs correct polynomial
    p = Polynomial([1])
    x = Polynomial([0, 1])
    for _ in range(i - 1):
        p = x*p - p.deriv()
    return p


def orthogonal_recurrence(bs: list[Polynomial],
                          p: Callable[[float], float],
                          dom: tuple[float, float],
                          eps: float = 1e-6) -> Polynomial:
    """
    The Stieltjes algorithm for constructing the next polynomial basis function
    b_{i + 1} according to the orthogonal recurrence relation, where `bs` contains
    {b_1, ..., b_i}, `p` is the probability distribution, and `dom` is a tuple
    containing a lower and upper bound for z. The optional parameter `eps`
    controls the absolute tolerance of the numerical integration. We make use of
    the `numpy.polynomials.Polynomial` class.
    """
    i = len(bs)
    c1 = integrate.quad(lambda z: z*(bs[i](z)**2)*p(z), dom[0], dom[1], epsabs=eps)[0]
    c2 = integrate.quad(lambda z:   (bs[i](z)**2)*p(z), dom[0], dom[1], epsabs=eps)[0]
    alpha = c1 / c2
    if i > 1:
        c3 = integrate.quad(lambda z: (bs[i - 1](z)**2)*p(z), dom[0], dom[1], epsabs=eps)[0]
        beta = c2 / c3
        return Polynomial([-alpha, 1])*bs[i] - beta*bs[i - 1]
    return Polynomial([-alpha, 1])*bs[i]


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
