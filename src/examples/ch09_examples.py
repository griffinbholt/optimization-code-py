import sys; sys.path.append('../')

import numpy as np

from ch09 import rand_population_uniform, genetic_algorithm,\
                 TruncationSelection, SinglePointCrossover, GaussianMutation


def example_9_1():
    """
    Example 9.1: Demonstration of using a genetic algorithm for optimizing a
    simple function.
    """
    np.random.seed(0)
    def f(x): return np.linalg.norm(x)
    m = 100     # population size
    k_max = 10  # number of iterations
    population = rand_population_uniform(m, a=np.array([-3.0, -3.0]), b=np.array([3.0, 3.0]))
    S = TruncationSelection(10)  # select top 10
    C = SinglePointCrossover()
    M = GaussianMutation(0.5)    # small mutation rate
    x = genetic_algorithm(f, population, k_max, S, C, M)
    print("x = ", x)

# TODO - Example 9.2 (Maybe eventually: need to construct the algorithm for Lamarckian and Baldwinian learning)