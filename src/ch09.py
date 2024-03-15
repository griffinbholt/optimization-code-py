"""Chapter 9: Population Methods"""

import numpy as np

from abc import ABC, abstractmethod
from typing import Callable

from Distributions import Distribution, MvNormal, Cauchy
from convenience import normalize


def rand_population_uniform(m: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    A method for sampling an initial population of `m` design points over a
    uniform hyperrectangle with lower-bound vector `a` and upper-bound vector `b`.
    """
    d = len(a)
    return a + np.random.rand(d) * (b - a)


def rand_population_normal(m: int, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    A method for sampling an initial population of `m` design points using a
    multivariate normal distribution with mean `mu` and covariance `Sigma`.
    """
    D = MvNormal(mu, Sigma)
    return D.rand(m)


def rand_population_cauchy(m: int, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    A method for sampling an initial population of `m` design points using a
    Cauchy distribution with location `mu` and scale `sigma` for each dimension.
    The location and scale are analogous to the mean and standard deviation used
    in a normal distribution.
    """
    n = len(mu)
    return np.array([[Cauchy(mu[j], sigma[j]).rand() for j in range(n)] for _ in range(m)])


def genetic_algorithms(f: Callable[[np.ndarray], float],
                       population: np.ndarray,
                       k_max: int,
                       S: 'SelectionMethod',
                       C: 'CrossoverMethod',
                       M: 'MutationMethod') -> np.ndarray:
    """
    The genetic algorithm, which takes an objective function `f`, an initial
    population `population`, number of iterations `k_max`, a `SelectionMethod` `S`,
    a `CrossoverMethod` `C`, and a `MutationMethod` `M`.
    """
    for _ in range(k_max):
        parents = S.select(np.apply_along_axis(f, 1, population))
        children = [C.crossover(population[p[0]], population[p[1]]) for p in parents]
        population = M.mutate(children)
    return population[np.argmin(np.apply_along_axis(f, 1, population))]


def rand_population_binary(m: int, n: int) -> np.ndarray:
    """
    A method for sampling random starting populations of `m` bit-string
    chromosomes of length `n`.
    """
    return np.random.randint(2, size=(m, n), dtype=bool)


class SelectionMethod(ABC):
    """
    Several selection methods for genetic algorithms. Calling selection with a
    `SelectionMethod` and the list of objective function values `y` will produce
    a list of parental pairs.
    """
    @abstractmethod
    def select(self, y: np.ndarray) -> np.ndarray:
        pass


class TruncationSelection(SelectionMethod):
    def __init__(self, k: int):
        self.k = k  # top k to keep

    def select(self, y: np.ndarray) -> np.ndarray:
        p = np.argsort(y)
        return np.array([p[np.random.choice(self.k, 2)] for _ in y])


class TournamentSelection(SelectionMethod):
    def __init__(self, k: int):
        self.k = k  # top k to keep

    def select(self, y: np.ndarray) -> np.ndarray:
        def getparent():
            p = np.random.permutation(len(y))
            return p[np.argmin(y[p[:self.k]])]
        return np.array([[getparent(), getparent()] for _ in y])


class RouletteWheelSelection(SelectionMethod):
    def select(self, y: np.ndarray) -> np.ndarray:
        y = np.max(y) - y
        p = normalize(y, ord=1)
        return np.random.choice(len(y), size=(len(y), 2), p=p)


class CrossoverMethod(ABC):
    """
    Several crossover methods for genetic algorithms. Calling crossover with a
    `CrossoverMethod` and two parents `a` and `b` will produce a child
    chromosome that contains a mixture of the parents' genetic codes.
    """
    @abstractmethod
    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        pass


class SinglePointCrossover(CrossoverMethod):
    """Works for both binary string and real-valued chromosomes"""
    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        i = np.random.randint(len(a))
        return np.concatenate((a[:i], b[i:]))


class TwoPointCrossover(CrossoverMethod):
    """Works for both binary string and real-valued chromosomes"""
    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        i, j = np.random.randint(n, size=2)
        if i > j:
            i, j = j, i
        return np.concatenate((a[:i], b[i:j], a[j:]))


class UniformCrossover(CrossoverMethod):
    """Works for both binary string and real-valued chromosomes"""
    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        child = np.copy(a)
        for i in range(len(a)):
            if np.random.rand() < 0.5:
                child[i] = b[i]
        return child


class InterpolationCrossover(CrossoverMethod):
    """
    A crossover method for real-valued chromosomes which performs linear
    interpolation between the parents.
    """
    def __init__(self, lam: float):
        self.lam = lam  # interpolation parameter

    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (1 - self.lam) * a + self.lam * b


class MutationMethod(ABC):
    @abstractmethod
    def mutate(self, child: np.ndarray) -> np.ndarray:
        pass


class BitwiseMutation(MutationMethod):
    """
    The bitwise mutation method for binary string chromosomes.
    Here, `lam` is the mutation rate.
    """
    def __init__(self, lam: float):
        self.lam = lam  # mutation rate

    def mutate(self, child: np.ndarray) -> np.ndarray:
        return np.array([~v if np.random.rand() < self.lam else v for v in child])


class GaussianMutation(MutationMethod):
    """
    The Gaussian mutation method for real-valued chromosomes.
    Here, `sigma` is the standard deviation.
    """
    def __init__(self, sigma: float):
        self.sigma = sigma  # standard deviation

    def mutate(self, child: np.ndarray) -> np.ndarray:
        return child + np.random.randn(len(child)) * self.sigma


def differential_evolution(f: Callable[[np.ndarray], float],
                           population: np.ndarray,
                           k_max: int,
                           p: float = 0.5,
                           w: float = 1.0) -> np.ndarray:
    """
    Differential evolution, which takes an objective function `f`, a population
    `population`, a number of iterations `k_max`, a crossover probability `p`,
    and a differential weight `w`. The best individual is returned.
    """
    m, n = population.shape
    for _ in range(k_max):
        for (k, x) in enumerate(population):
            a, b, c = np.random.choice(population, 
                                       p=normalize(np.array([j != k for j in range(m)]), ord=1), 
                                       size=3, replace=False)
            z = a + w * (b - c)
            j = np.random.randint(len(n))
            x_prime = np.array([z[i] if ((i == j) or (np.random.rand() < p)) else x[i] for i in range(n)])
            if f(x_prime) < f(x):
                x = x_prime
    return population[np.argmin(np.apply_along_axis(f, 1, population))]


class Particle():
    """
    Each particle in particle swarm optimization has a position `x` and velocity
    `v` in design space and keeps track of the best design point found so far,
    `x_best`.
    """
    def __init__(self, x: np.ndarray, v: np.ndarray, x_best: np.ndarray):
        self.x = x
        self.v = v
        self.x_best = x_best


def particle_swarm_optimization(f: Callable[[np.ndarray], float],
                                population: list[Particle],
                                k_max: int,
                                w: float = 1.0,
                                c1: float = 1.0,
                                c2: float = 1.0) -> list[Particle]:
    """
    Particle swarm optimization, which takes an objective function `f`, a list
    of particles `population`, a number of iterations `k_max`, an inertia `w`,
    an momentum coefficients `c1` and `c2`.
    
    The default values are those used by R. Eberhart and J. Kennedy, "A New
    Optimizer Using Particle Swarm Theory," in International Symposium on Micro
    Machine and Human Science, 1995.
    """
    n = len(population[0].x)
    x_best, y_best = np.copy(population[0].x_best), np.inf
    for P in population:
        y = f(P.x)
        if y < y_best:
            x_best, y_best = P.x, y
    for _ in range(k_max):
        for P in population:
            r1, r2 = np.random.rand(n), np.random.rand(n)
            P.x += P.v
            P.v = w*P.v + c1*r1*(P.x_best - P.x) + c2*r2*(x_best - P.x)
            y = f(P.x)
            if y < y_best:
                x_best, y_best = P.x, y
            if y < f(P.x_best):
                P.x_best = P.x
    return population


def firefly(f: Callable[[np.ndarray], float],
            population: np.ndarray,
            k_max: int,
            beta: float = 1.0,
            alpha: float = 0.1,
            brightness: Callable[[float], float] = lambda r: np.exp(-(r**2))) -> np.ndarray:
    """
    The firefly algorithm, which takes an objective function `f`, a population
    `population` consisting of design points, a number of iterations `k_max`,
    a source intensity `beta`, a random walk step size `alpha`, and an intensity
    function `brightness`. The best design point is returned.
    """
    m = len(population[0])
    N = MvNormal(np.zeros(m), np.eye(m))
    for _ in range(k_max):
        for a in population:
            for b in population:
                if f(b) < f(a):
                    r = np.linalg.norm(b - a)
                    a += beta * brightness(r) * (b - a) + alpha * N.rand()
    return population[np.argmin(np.apply_along_axis(f, 1, population))]


class Nest():
    def __init__(self, x: np.ndarray, y: float):
        self.x = x  # position
        self.y = y  # value, f(x)


def cuckoo_search(f: Callable[[np.ndarray], float],
                  population: list[Nest],
                  k_max: int,
                  p_a: float = 0.1,
                  C: Distribution = Cauchy(0, 1)) -> list[Nest]:
    """
    Cuckoo search, which takes an objective function `f`, an initial set of
    nests `population`, a number of iterations `k_max`, percent of nests to
    abandon `p_a`, and flight distribution `C`. The flight distribution is
    typically a centered Cauchy distribution.
    """
    m, n = len(population), len(population[0].x)
    a = round(m*p_a)
    for _ in range(k_max):
        i, j = np.random.randint(m, size=2)
        x = population[j].x + C.rand(n)
        y = f(x)
        if y < population[i].y:
            population[i].x = x
            population[i].y = y
    
        p = np.argsort([-nest.y for nest in population])
        for i in range(len(a)):
            j = np.random.randint(m - a) + a
            population[p[i]] = Nest(population[p[j]].x + C.rand(n), f(population[p[i]].x))
    return population
