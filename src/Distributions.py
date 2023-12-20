from abc import ABC, abstractmethod

from scipy.stats import cauchy, multivariate_normal, norm

import numpy as np


class Distribution(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def rand(self, size: int = 1) -> float | np.ndarray:
        return self._distrib.rvs(size)

    @staticmethod
    @abstractmethod
    def fit(x: np.ndarray) -> 'Distribution':
        pass


class Cauchy(Distribution):
    def __init__(self, loc: float, scale):
        self.loc = loc
        self.scale = scale
        self._distrib = cauchy(loc, scale)
    
    def fit(x: np.ndarray) -> 'Cauchy':
        raise NotImplementedError  # TODO


class MvNormal(Distribution):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = mean
        self._distrib = multivariate_normal(mean, cov)

    def fit(x: np.ndarray) -> 'MvNormal':
        return MvNormal(np.mean(x, axis=0), np.cov(x, rowvar=0))


class Normal(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        self._distrib = norm(mean, std)

    def fit(x: np.ndarray) -> 'Normal':
        return Normal(np.mean(x), np.std(x))

    def cdf(self, x: np.ndarray) -> float:
        return self._distrib.cdf(x)

    def pdf(self, x: np.ndarray) -> float:
        return self._distrib.pdf(x)
