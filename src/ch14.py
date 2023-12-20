"""Chapter 14: Surrogate Models"""

import numpy as np

from itertools import product
from typing import Callable


def design_matrix(X: np.ndarray) -> np.ndarray:
    """A method for constructing a design matrix from a list of design points `X`"""
    m = len(X)
    return np.hstack([np.ones((m, 1)), X])


def linear_regression(X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], float | np.ndarray]:
    """
    A method for fitting a surrogate model using linear regression to a list of
    design points `X` and a vector of objective function values `y`.
    """
    theta = np.pinv(design_matrix(X)) @ y
    return lambda x: np.dot(x, theta[1:]) + theta[0]


def regression(X: np.ndarray,
               y: np.ndarray,
               bases: list[Callable[[np.ndarray], float]],
               lam: float = 0.0) -> Callable[[np.ndarray], float | np.ndarray]:
    """
    A method for fitting a surrogate model to a list of design points `X` and
    corresponding objective function values `y` using regression with basis
    functions contained in the `bases` list.

    `lam` is an optional smoothing term, for regression in the presence of noise.
    """
    B = np.array([[b(x) for b in bases] for x in X])
    theta = np.linalg.solve(B.T @ B + lam * np.eye(len(bases)), B.T @ y)
    return lambda x: np.sum([theta[i] * bases[i](x) for i in range(len(theta))], axis=-1)


def polynomial_bases_1d(i: int, k: int) -> list[Callable[[np.ndarray], float]]:
    """
    A method for constructing a list of polynomial basis functions up to a degree `k`
    for the `i`th component of a design point.
    """
    return [lambda x: x[i]**p for p in range(k + 1)]


def polynomial_bases(n: int, k: int) -> list[Callable[[np.ndarray], float]]:
    """
    A method for constructing a list of `n`-dimensional polynomial bases for
    terms up to degree `k`.
    """
    bases = [polynomial_bases_1d(i, k) for i in range(n)]
    terms = []
    for ks in product(*[range(k + 1) for i in range(n)]):
        if sum(ks) <= k:
            terms.append(lambda x, ks=ks: np.prod([b[j](x) for (j, b) in zip(ks, bases)]))
    return terms


def sinusoidal_bases_1d(j: int, k: int, a: np.ndarray, b: np.ndarray) -> list[Callable[[np.ndarray], float]]:
    """
    Produces a list of sinusoidal basis function up to degree `k` for the `i`th
    component of the design vector given lower bound `a` and upper bound `b`.
    """
    T = b[j] - a[j]
    bases = [lambda x: 0.5]
    for i in range(1, k + 1):
        bases.append(lambda x: np.sin(2*np.pi*i*x[j]/T))
        bases.append(lambda x: np.cos(2*np.pi*i*x[j]/T))


def sinusoidal_bases(k: int, a: np.ndarray, b: np.ndarray) -> list[Callable[[np.ndarray], float]]:
    """
    Produces all sinusoidal base function combinations up to degree `k` for
    lower-bound vector `a` and upper-bound vector `b`.
    """
    n = len(a)
    bases = [sinusoidal_bases_1d(i, k, a, b) for i in range(n)]
    terms = []
    for ks in product(*[range(2*k + 1) for i in range(n)]):
        powers = [(k + 1) // 2 for k in ks]
        if sum(powers) <= k:
            terms.append(lambda x, ks=ks: np.prod([b[j][x] for (j, b) in zip(ks, bases)]))
    return terms


def radial_bases(psi: Callable[[float], float], C: np.ndarray, p: float = 2) -> list[Callable[[np.ndarray], float]]:
    """
    A method for obtaining a list of basis functions given a radial basis
    function `psi`, a list of centers `C`, and an L_p norm parameter p`.
    """
    return [lambda x: psi(np.linalg.norm(x - c, p)) for c in C]


class TrainTest():
    """
    A utility type for training a model and then validating it on a metric.
    Here, `train` and `test` are arrays of indices into the training data.
    """
    def __init__(self, train: np.ndarray, test: np.ndarray):
        self.train = train
        self.test = test


def train_and_validate(X: np.ndarray,
                       y: np.ndarray,
                       tt: TrainTest,
                       fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], float]],
                       metric: Callable[[Callable[[np.ndarray], float], np.ndarray, np.ndarray], float]) -> float:
    """
    A utility method for training a model and then validating it on a metric.
    Here, `X` is a list of design points, `y` is the vector of corresponding
    function evaluations, `tt` is a train-test partition, `fit` is a model
    fitting function, and `metric` evaluates a model on a test set to produce an
    estimate of generalization error.
    """
    model = fit(X[tt.train], y[tt.train])
    return metric(model, X[tt.test], y[tt.test])


def holdout_partition(m: int, h: int = None) -> TrainTest:
    """
    A method for randomly partitioning `m` data samples into training and
    holdout sets, where `h` samples are assigned to the holdout set.
    """
    h = m // 2 if h is None else h
    p = np.random.permutation(m)
    train = p[h:]
    holdout = p[:h]
    return TrainTest(train, holdout)


def random_subsampling(X: np.ndarray,
                       y: np.ndarray,
                       fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], float]],
                       metric: Callable[[Callable[[np.ndarray], float], np.ndarray, np.ndarray], float],
                       h: int = None,
                       k_max: int = 10) -> float:
    """
    The random subsampling method used to obtain mean and standard deviation
    estimates for model generalization error using `k_max` runs of the holdout
    method.
    """
    m = len(X)
    return np.mean([train_and_validate(X, y, holdout_partition(m, h), fit, metric) for _ in range(k_max)])


def k_fold_cross_validation_sets(m: int, k: int) -> list[TrainTest]:
    """
    Constructs the sets needed for `k`-fold cross validation on `m` samples,
    with `k` <= `m`.
    """
    perm = np.random.permutation(m)
    sets = []
    for i in range(k):
        validate = perm[i:m:k]
        train = perm[np.setdiff1d(range(m), range(i, m, k))]
        sets.append(TrainTest(train, validate))
    return sets
    

def multiset_validation_estimate(X: np.ndarray,
                                 y: np.ndarray,
                                 sets: list[TrainTest],
                                 fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], float]],
                                 metric: Callable[[Callable[[np.ndarray], float], np.ndarray, np.ndarray], float]) -> float:
    """
    Computes the mean of the generalization error estimate by training and
    validating on the list of train-validate sets contained in `sets`. The other
    variables are the list of design points `X`, the corresponding objective
    function values `y`, a function `fit` that trains a surrogate model, and a
    function `metric` that evaluates a model on a data set.

    NOTE: Works for Cross-Validation sets and Bootstrap sets
    """
    return np.mean([train_and_validate(X, y, tt, fit, metric) for tt in sets])


def bootstrap_sets(m: int, b: int) -> list[TrainTest]:
    """A method for obtaining `b` bootstrap samples, each for a data set of size `m`"""
    return [TrainTest(np.random.randint(m, size=m), np.arange(m)) for i in range(b)]


def leave_one_out_bootstrap_estimate(X: np.ndarray,
                                     y: np.ndarray,
                                     sets: list[TrainTest],
                                     fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], float]],
                                     metric: Callable[[Callable[[np.ndarray], float], np.ndarray, np.ndarray], float]) -> float:
    """
    A method for computing the leave-one-out bootstrap generalization error
    estimate using the train-validate sets `sets`. The other variables are the
    list of design points `X`, the corresponding objective function values `y`,
    a function `fit` that trains a surrogate model, and a function `metric` that
    evaluates a model on a data set.
    """
    m, b = len(X), len(sets)
    error = 0.0
    models = [fit(X[tt.train], y[tt.train]) for tt in sets]
    for j in range(m):
        c = 0
        delta = 0.0
        for i in range(b):
            if j not in sets[i].train:
                c += 1
                delta += metric(models[i], np.array([X[j]]), np.array([y[j]]))
        error += delta / c
    return error / m


def bootstrap_632_estimate(X: np.ndarray,
                           y: np.ndarray,
                           sets: list[TrainTest],
                           fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], float]],
                           metric: Callable[[Callable[[np.ndarray], float], np.ndarray, np.ndarray], float]) -> float:
    """
    A method for obtaining the 0.632 bootstrap estimate for data points `X`,
    objective function values `y`, fitting function `fit`, and
    metric function `metric`.
    """
    eps_loob = leave_one_out_bootstrap_estimate(X, y, sets, fit, metric)
    eps_boot = multiset_validation_estimate(X, y, sets, fit, metric)
    return 0.632 * eps_loob + 0.368 * eps_boot
