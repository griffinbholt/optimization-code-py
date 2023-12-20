"""Chapter 13: Sampling Plans"""

import numpy as np

from abc import abstractmethod
from itertools import product
from numpy import ndarray
from primePy import primes
from typing import Callable

from ch03 import PHI

# TODO - Rethink the classing in this chapter


class SamplingPlan():
    def __init__(self, *args):
        assert len(args) == 1
        self.X = args[0]  # array of points in sampling plan

    def pairwise_distances(self, p: float = 2) -> np.ndarray:
        """
        A function for obtaining the list of pairwise distances between points in
        sampling plan `self` using the L_p norm specified by `p`.
        """
        m = len(self.X)
        return np.array([np.linalg.norm(self.X[i] - self.X[j], p) for i in range(m - 1) for j in range(i, m)])

    def compare(self, other: 'SamplingPlan', p: float = 2) -> int:
        """
        A function for comparing the degree to which two sampling plans `self`
        and `other` are space-filling using the L_p norm specified by `p`.

        The function returns: * -1, if `self` is more space-filling than `other`
                              *  1, if `self` is more space-filling than `other`
                              *  0, if they are equivalent
        """
        p_self = np.sort(self.pairwise_distances(p))
        p_other = np.sort(other.pairwise_distances(p))
        for (d_self, d_other) in zip(p_self, p_other):
            if d_self < d_other:
                return 1
            elif d_self > d_other:
                return -1
        return 0

    def phiq(self, q: float = 1, p: float = 2) -> float:
        """
        An implementation of the Morris-Mitchell criterion which takes a list of
        design points `X`, the criterion parameter `q` > 0, and a norm parameter
        `p` >= 1.
        """
        dists = self.pairwise_distances(p)
        return np.sum(dists**(-q))**(1/q)

    def copy(self) -> 'SamplingPlan':
        return SamplingPlan(np.copy(self.X))

    def append(self, x: np.ndarray):
        self.X = np.append(self.X, x)

    def __contains__(self, x: np.ndarray) -> bool:
        return x in self.X

    def __iter__(self):
        for x in self.X:
            yield x

    def __getitem__(self, key):
        return self.X[key]

    def __setitem__(self, key, value):
        self.X[key] = value


class FullFactorialPlan(SamplingPlan):
    """
    A function for obtaining all sample locations for the full factorial grid.
    Here, `a` is a vector of variable lower bounds, `b` is a vector of variable
    upper bounds, and `m` is a vector of sample counts for each dimension.
    """
    def __init__(self, a: np.ndarray, b: np.ndarray, m: np.ndarray):
        ranges = [np.linspace(a[i], stop=b[i], num=m[i]) for i in range(len(a))]
        X = np.array(list(product(*ranges)))
        super().__init__(X)


class UniformProjectionPlan(SamplingPlan):
    """
    A function for constructing a uniform projection plan for an `n`-dimensional
    hypercube with `m` samples per dimension. It returns a vector of index vectors.
    """
    def __init__(self, m: int, n: int):
        perms = [np.random.permutation(m) for _ in range(n)]
        X = np.array([[perms[i][j] for i in range(n)] for j in range(m)])
        super().__init__(X)

    def mutate(self):
        """
        A function for mutating uniform projection plan `X`, while maintaining
        its uniform projection property.
        """
        m, n = self.X.shape
        j = np.random.randint(n)
        i = np.random.permutation(m)[:2]
        self.X[i[0], j], self.X[i[1], j] = self.X[i[1], j], self.X[i[0], j]


def d_max(A: SamplingPlan, B: SamplingPlan, p: float = 2) -> float:
    """
    The set L_p distance metrics between two discrete sets, where `A` and `B`
    are lists of design points and `p` is the L_p norm parameter.
    """
    def min_dist(a, B, p) -> float:
        return np.min([np.linalg.norm(a - b, p) for b in B])
    return np.max([min_dist(a, B, p) for a in A])


def greedy_local_search(X: SamplingPlan, 
                        m: int, 
                        d: Callable[[SamplingPlan, SamplingPlan], float] = d_max) -> SamplingPlan:
    """
    Greedy local search, for finding `m`-element sampling plans that minimize
    a distance metric `d` for discrete set `X`.
    """
    S = SamplingPlan(np.array([X[np.random.randint(m)]]))
    for _ in range(m - 1):
        j = np.argmin([np.inf if x in S else d(X, S.copy().append(x)) for x in X])
        S.append(X[j])
    return S


def exchange_algorithm(X: SamplingPlan,
                       m: int,
                       d: Callable[[SamplingPlan], float] = d_max) -> SamplingPlan:
    """
    The exchange algorithm for finding `m`-element sampling plans that minimize
    a distance metric `d` for discrete set `X`.
    """
    S = SamplingPlan(X[np.random.permutation(m)])
    delta, done = d(X, S), False
    while not done:
        best_pair = (0,0)
        for i in range(m):
            s = S[i]
            for (j, x) in enumerate(X):
                if x not in S:
                    S[i] = x
                    delta_prime = d(X, S)
                    if delta_prime < delta:
                        delta = delta_prime
                        best_pair = (i,j)
            S[i] = s
        done = best_pair == (0,0)
        if not done:
            i,j = best_pair
            S[i] = X[j]
    return S


def multistart_local_search(X: SamplingPlan,
                            m: int,
                            alg: Callable,
                            k_max: int,
                            d: Callable[[SamplingPlan, SamplingPlan], float] = d_max) -> SamplingPlan:
    """
    Multistart local search runs a particular search algorithm multiple times
    and returns the best result. Here, `X` is the list of points, `m` is the size
    of the desired sampling plan, `alg` is either `exchange_algorithm` or
    `greedy_local_search`, `k_max` is the number of iterations to run, and `d`
    is the distance metric.
    """
    assert alg.__name__ in ['exchange_algorithm', 'greedy_local_search']
    sets = [alg(X, m, d) for _ in range(k_max)]
    return sets[np.argmin([d(X, S) for S in sets])]


class FillingSet(SamplingPlan):
    def __init__(self, m: int, n: int, max_prime: int):
        bs = primes.upto(max(np.ceil(n*(np.log(n) + np.log(np.log(n)))), max_prime))
        seqs = np.array([self._get_filling_set(m, b) for b in bs[:n]])
        super().__init__(seqs.T)

    @abstractmethod
    def _get_filling_set(self, m: int, b: int) -> np.ndarray:
        pass


class AdditiveRecurrenceFillingSet(FillingSet):
    """
    Additive recurrence for constructing `m`-element filling sequences over
    `n`-dimensional hypercubes. The `primePy` package is used to generate
    the first `n` prime numbers, where the kth prime number is bounded by

    k(log(k) + loglog(k))

    for k > 6, and `primes.upto(a)` returns all primes up to `a`. Note that 13
    is the sixth prime number.
    """
    def __init__(self, m: int, n: int):
        super().__init__(m, n, max_prime=13)

    def _get_filling_set(self, m: int, b: int = None) -> np.ndarray:
        c = np.sqrt(b) if b is not None else PHI - 1
        X = np.random.rand(1)
        for _ in range(m - 1):
            X = np.append(X, (X[-1] + c) % 1)
        return X


class HaltonFillingSet(FillingSet):
    """
    Halton quasi-random `m`-element filling sequences over `n`-dimensional unit
    hypercubes, where `b` is the base. The bases `bs` must be coprime.
    """
    def __init__(self, m: int, n: int):
        super().__init__(m, n, max_prime=6)
    
    def _get_filling_set(self, m: int, b: int = 2) -> ndarray:
        return np.array([self.halton(i, b) for i in range(1, m + 1)])

    def halton(self, i: int, b: int) -> float:
        result, f = 0.0, 1.0
        while i > 0:
            f = f / b
            result += f * (i % b)
            i = np.floor(i / b)
        return result
