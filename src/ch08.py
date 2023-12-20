"""Chapter 8: Stochastic Methods"""

import numpy as np

from scipy.linalg import fractional_matrix_power
from scipy.stats import uniform
from typing import Callable

from ch05 import DescentMethod
from ch07 import basis

from Distributions import Distribution, MvNormal


class NoisyDescent(DescentMethod):
    """
    A noisy descent method, which augments another descent method with additive
    Gaussian noise. The method takes another `DescentMethod` `submethod`, a
    noise sequence `sigma` and stores the iteration count `k`.
    """
    def __init__(self, submethod: DescentMethod, sigma: Callable[[int], float], k: int):
        assert not isinstance(submethod, NoisyDescent)  # needs to be non-stochastic descent method
        self.submethod = submethod
        self.sigma = sigma
        self.k = k
    
    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.submethod.initialize(f, grad_f, x)
        self.k = 1
    
    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        x = self.submethod.step(f, grad_f, x)
        sigma = self.sigma(self.k)
        x += sigma * np.random.randn(len(x))
        self.k += 1
        return x


def rand_positive_spanning_set(alpha: float, n: int) -> np.ndarray:
    """
    Randomly sampling a positive spanning set of n + 1 directions according to
    mesh adaptive search with step size `alpha` and number of dimensions `n`.
    """
    delta = round(1 / np.sqrt(alpha))
    L = np.diag(delta * np.random.choice([1, -1], n))
    for i in range(n - 1):
        for j in range(i - 1):
            L[i, j] = np.random.randint(-delta + 1, delta)
    D = L[np.random.permutation(n), :]
    D = D[:, np.random.permutation(n)]
    D = np.hstack([D, -np.sum(D, axis=1, keepdims=True)])
    return D.T


def mesh_adaptive_direct_search(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float) -> np.ndarray:
    """
    Mesh adaptive direct search for an objective function `f`, an initial design
    `x`, and a tolerance `eps.
    """
    alpha, y, n = 1.0, f(x), len(x)
    while alpha > eps:
        improved = False
        for (i, d) in enumerate(rand_positive_spanning_set(alpha, n)):
            x_prime = x + alpha * d
            y_prime = f(x_prime)
            if y_prime < y:
                x, y, improved = x_prime, y_prime, True
                x_prime = x + 3 * alpha * d
                y_prime = f(x_prime)
                if y_prime < y:
                    x, y = x_prime, y_prime
                break
        alpha = np.min(4 * alpha, 1.0) if improved else alpha / 4
    return x


def simulated_annealing(f: Callable[[np.ndarray], float],
                        x: np.ndarray,
                        T: Distribution,
                        t: Callable[[int], float],
                        k_max: int) -> np.ndarray:
    """
    Simulated annealing, which takes as input an objective function `f`, an
    initial point `x`, a sampling function `T` for the transition distribution,
    an annealing schedule `t`, and the number of iterations `k_max`.
    """
    y = f(x)
    x_best, y_best = x, y
    for k in range(1, k_max + 1):
        x_prime = x + T.rand()
        y_prime = f(x_prime)
        delta_y = y_prime - y
        if (delta_y <= 0) or (np.random.rand() < np.exp(-delta_y / t(k))):
            x, y = x_prime, y_prime
        if y_prime < y_best:
            x_best, y_best = x_prime, y_prime
    return x_best


def corana_update(v: np.ndarray, a: np.ndarray, c: np.ndarray, ns: int) -> np.ndarray:
    """
    The update formula used by Corana et al. in adaptive simulated annealing,
    where `v` is a vector of coordinate step sizes, `a` is a vector of the
    number of accepted steps in each coordinate direction, `c` is a vector of
    step scaling factors for each coordinate direction, and `nz` is the number
    of cycles before running the step size adjustment.
    """
    for i in range(len(v)):
        a_i, c_i = a[i], c[i]
        if a_i > 0.6 * ns:
            v[i] *= 1 + (c_i * ((a_i/ns) - 0.6)) / 0.4
        elif a_i < 0.4 * ns:
            v[i] *= 1 + (c_i * (0.4 - (a_i/ns))) / 0.4
    return v


def adaptive_simulated_annealing(f: Callable[[np.ndarray], float],
                                 x: np.ndarray,
                                 v: np.ndarray,
                                 t: float,
                                 eps: float,
                                 ns: int = 20,
                                 neps: int = 3,
                                 nt: int = None,
                                 gamma = 0.85,
                                 c: np.ndarray = None) -> np.ndarray:
    """
    The adaptive simulated annealing algorithm, where `f` is the multivariate
    objective function, `x` is the starting point, `v` is the starting step
    vector, `t` is the starting temperature, and `eps` is the termintation
    criterion parameter. The optional parameters are the number of cycles before
    running the step adjustment size `ns`, the number of cycles before reducing
    the temperature `nt`, the number of successive temperature reductions to
    test for termination `neps`, the temperature reduction coefficient `gamma`,
    and the direction-wise varying criterion `c`.

    A flowchart for the adaptive simulated annealing algorithm, as presented in
    the original paper, is displayed on p. 134 of the textbook.
    """
    nt = max(100, 5 * len(x)) if nt is None else nt
    c = np.full(len(x), 2.0) if c is None else c

    y = f(x)
    x_best, y_best = x, y
    y_arr, n, U = [], len(x), uniform(-1, 2)
    a, counts_cycles, counts_resets = np.zeros(n), 0, 0

    while True:
        for i in range(n):
            x_prime = x + basis(i, n) * U.rvs() * v[i]
            y_prime = f(x_prime)
            delta_y = y_prime - y
            if (delta_y <= 0) or (np.random.rand() < np.exp(-delta_y / t)):
                x, y = x_prime, y_prime
                a[i] += 1
                if y_prime < y_best:
                    x_best, y_best = x_prime, y_prime
        
        counts_cycles += 1
        if counts_cycles < ns:
            continue

        counts_cycles = 0
        v = corana_update(v, a, c, ns)
        a.fill(0)
        counts_resets += 1
        if counts_resets < nt:
            continue

        t *= gamma
        counts_resets = 0
        y_arr.append(y)

        if not ((len(y_arr) > neps) and\
                (y_arr[-1] - y_best <= eps) and\
                np.all([np.abs(y_arr[-1] - y_arr[-1 - u]) <= eps for u in range(1, neps + 1)])):
            break
    return x_best


def cross_entropy(f: Callable[[np.ndarray], float],
                  P: Distribution,
                  k_max: int,
                  m: int = 100,
                  m_elite: int = 10) -> Distribution:
    """
    The cross-entropy method, which takes an objective function `f` to
    be minimized, a proposal distribution `P`, an iteration count `k_max`,
    a sample size `m`, and the number of samples to use when refitting the
    distribution `m_elite`. It returns the updated distribution over where the
    global minimum is likely to exist.
    """
    for _ in range(k_max):
        samples = P.rand(m)  # return shape (m, n), where n is dimension of random variable
        order = np.argsort(np.apply_along_axis(f, 1, samples))[0]
        P = type(P).fit(samples[order[:m_elite]])
    return P


def natural_evolution_strategies(f: Callable[[np.ndarray], float],
                                 rand: Callable[[np.ndarray, int], np.ndarray],
                                 grad_ll: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                 theta: np.ndarray,
                                 k_max: int,
                                 m: int = 100,
                                 alpha: float = 0.01) -> np.ndarray:
    """
    The natural evolution strategies method, which takes an objective function `f`
    to be minimized, a distribution `P`, an initial distribution parameter vector
    `theta` for the distribution `P`, an iteration count `k_max`, a sample size `m`,
    and a step factor `alpha`. An optimized parameter vector is returned.

    The method `rand(theta, m)` should sample `m` individuals from the distribution
    parameterized by `theta`, and `grad_ll(x, theta)` should return the
    log likelihood gradient.
    """
    for _ in range(k_max):
        samples = rand(theta, m)
        theta -= alpha * np.sum([f(x) * grad_ll(x, theta) for x in samples]) / m
    return theta


def covariance_matrix_adaptation(f: Callable[[np.ndarray], float],
                                 x: np.ndarray,
                                 k_max: int,
                                 sigma: float = 1.0,
                                 m: int = None,
                                 m_elite: int = None) -> np.ndarray:
    """
    Covariance matrix adaptation, which takes an objective function `f` to be
    minimized, an initial design point `x`, and an iteration count `k_max`.
    One can optionally specify the step-size scalar `sigma`, the sample size `m`,
    and the number of elite samples `m_elite`.

    The best candidate design point is returned, which is the mean of the final
    sample distribution.

    The covariance matrix undergoes an additional operation to ensure that it
    remains summetroc; otherwise small numerical inconsistencies can cause the
    matrix no longer to be positive definite.

    This implementation uses a simplified normalization strategy for the
    negative weights. The original can be found in Equations 50-53 of N. Hansen,
    "The CMA Evolution Strategy: A Tutorial," ArXiv, no. 1604.00772, 2016.
    """
    m = 4 + np.floor(3 * np.log(len(x))).astype(int) if m is None else m
    m_elite = m // 2

    mu, n = np.copy(x), len(x)
    ws = np.log((m + 1) / 2) - np.log(np.arange(1, m + 1))
    ws[:m_elite] /= np.sum(ws[:m_elite])
    mu_eff = 1 / np.sum(ws[:m_elite]**2)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_Sigma = (4 + (mu_eff/n)) / (n + 4 + 2 * (mu_eff/n))
    c_1 = 2 / (((n + 1.3)**2) + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + (1/mu_eff)) / (((n + 2)**2) + mu_eff))
    ws[m_elite:] *= -(1 + (c_1/c_mu)) / np.sum(ws[m_elite:])
    E = np.sqrt(n) * (1 - (1/(4*n)) + (1/(21*(n**2))))
    p_sigma, p_Sigma, Sigma = np.zeros(n), np.zeros(n), np.eye(n)
    for k in range(1, k_max + 1):
        P = MvNormal(mu, (sigma**2)*Sigma)
        xs = P.rand(m)
        ys = np.apply_along_axis(f, 1, xs)
        idcs = np.argsort(ys)  # best to worst

        # selection and mean update
        delta_s = (xs - mu) / sigma
        delta_w = np.sum([ws[i] * delta_s[idcs[i]] for i in range(m_elite)], axis=0)
        mu += sigma * delta_w

        # step-size control
        C = fractional_matrix_power(Sigma, -0.5)
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ delta_w)
        sigma *= np.exp((c_sigma/d_sigma) * ((np.linalg.norm(p_sigma) / E) - 1))

        # covariance adaptation
        h_sigma = int(np.linalg.norm(p_sigma) / np.sqrt(1 - ((1 - c_sigma)**(2*k))) < (1.4 + (2/(n + 1)))*E)
        p_Sigma = (1 - c_Sigma) * p_Sigma + h_sigma * np.sqrt(c_Sigma * (2 - c_Sigma) * mu_eff) * delta_w
        w0 = [ws[i] if ws[i] >= 0 else n*ws[i]/(np.linalg.norm(C @ delta_s[idcs[i]])**2) for i in range(m)]
        Sigma = (1 - c_1 - c_mu) * Sigma +\
                c_1 * (np.outer(p_Sigma, p_Sigma) + (1 - h_sigma) * c_Sigma * (2 - c_Sigma) * Sigma) +\
                c_mu * np.sum([w0[i] * np.outer(delta_s[idcs[i]], delta_s[idcs[i]]) for i in range(m)], axis=0)
        Sigma = np.triu(Sigma) + np.triu(Sigma, 1).T  # enforce symmetry
    return mu
