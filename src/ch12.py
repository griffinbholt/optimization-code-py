"""Chapter 12: Multiobjective Optimization"""

import numpy as np

from typing import Callable

from ch09 import SelectionMethod, CrossoverMethod, MutationMethod


def dominates(y: np.ndarray, y_prime: np.ndarray) -> bool:
    """
    A method for checking whether x dominates x_prime, where `y` is the vector
    of objective values for f(x) and `y_prime` is the vector of objective values
    for f(x_prime).
    """
    return np.all(y <= y_prime) and np.any(y < y_prime)


def naive_pareto(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    A method for generating a Pareto frontier using randomly sampled design
    ponts `xs` and their multiobjective values `ys`. Both the Pareto-optimal
    design points and their objective values are returned.
    """
    pareto_xs, pareto_ys = [], []
    for (x, y) in zip(xs, ys):
        if not np.any([dominates(y_prime, y) for y_prime in ys]):
            pareto_xs.append(x)
            pareto_ys.append(y)
    return (np.array(pareto_xs), np.array(pareto_ys))


def weight_pareto(f1: Callable[[np.ndarray], float],
                  f2: Callable[[np.ndarray], float],
                  optimize: Callable[[Callable[[np.ndarray], float]], np.ndarray],
                  npts: int) -> np.ndarray:
    """
    The weighted sum method for generating a Pareto frontier, which takes
    objective functions `f1` and `f2` and number of Pareto points `npts`.
    """
    return np.array([optimize(lambda x: w1 * f1(x) + (1 - w1) * f2(x)) for w1 in np.linspace(0, stop=1, num=npts)])


def vector_evaluated_genetic_algorithm(f: Callable[[np.ndarray], np.ndarray],
                                       population: np.ndarray,
                                       k_max: int,
                                       S: SelectionMethod,
                                       C: CrossoverMethod,
                                       M: MutationMethod) -> np.ndarray:
    """
    The vector-evaluated genetic algorithm which takes a vector-valued objective
    function `f`, an initial population, number of iterations `k_max`, a
    `SelectionMethod` `S`, a `CrossoverMethod` `C`, and a `MutationMethod` `M`.
    The resulting population is returned.
    """
    m = len(f(population[0]))
    m_pop = len(population)
    m_subpop = m_pop // m
    for _ in range(k_max):
        ys = np.apply_along_axis(f, 1, population)
        parents = np.apply_along_axis(lambda y: S.select(y)[:m_subpop], 0, ys)

        p = np.random.permutation(2*m_pop)
        def p_ind(i): return parents[(p[i] - 1) % m_pop][(p[i] - 1) // m_pop]
        parents = np.array([[p_ind(i), p_ind(i + 1)] for i in range(0, 2*m_pop, 2)])
        children = np.array([C.crossover(population[p[0]], population[p[1]]) for p in parents])
        population = np.array([M.mutate(c) for c in children])
    return population


def get_non_domination_levels(ys: np.ndarray) -> np.ndarray:
    """
    A function for getting the nondomination levels of an array of
    multiobjective function evaluations `ys`.
    """
    L, m = 0, len(ys)
    levels = np.zeros(m).astype(int)
    while np.min(levels) == 0:
        L += 1
        for (i, y) in enumerate(ys):
            if (levels[i] == 0) and\
               not np.any([(levels[i] == 0 or levels[i] == L) & dominates(ys[i], y) for i in range(m)]):
                levels[i] = L
    return levels


def discard_closest_pair(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This method is used to remove one individual from a filter that is above
    capacity. The method takes the filter's list of design points `xs` and
    associated objective function values `ys`.
    """
    index, min_dist = 0, np.inf
    for (i, y) in enumerate(ys):
        for (j, y_prime) in enumerate(ys[i:]):
            dist = np.linalg.norm(y - y_prime)
            if dist < min_dist:
                index, min_dist = np.random.choices([i, j]), dist
    xs = np.delete(xs, index, axis=0)
    ys = np.delete(ys, index, axis=0)
    return (xs, ys)


def update_pareto_filter(filter_xs: np.ndarray,
                         filter_ys: np.ndarray,
                         xs: np.ndarray,
                         ys: np.ndarray,
                         capacity: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    A method for updating a Pareto filter with design points `filter_xs`,
    corresponding objective function values `filter_ys`, a population with
    design points `xs` and objective values `ys`, and filter capacity `capactity`
    which defaults to the population size.
    """
    capacity = len(xs) if capacity is None else capacity
    for (x, y) in zip(xs, ys):
        if not np.any([dominates(y_prime, y) for y_prime in filter_ys]):
            filter_xs = np.append(filter_xs, x)
            filter_ys = np.append(filter_ys, y)
    filter_xs, filter_ys = naive_pareto(filter_xs, filter_ys)
    while len(filter_xs) > capacity:
        filter_xs, filter_ys = discard_closest_pair(filter_xs, filter_ys)
    return (filter_xs, filter_ys)
