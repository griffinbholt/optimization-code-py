from abc import ABC, abstractmethod

from scipy.stats import multivariate_normal, cauchy

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


class MvNormal(Distribution):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = mean
        self._distrib = multivariate_normal(mean, cov)

    def fit(x: np.ndarray) -> 'MvNormal':
        return MvNormal(np.mean(x, axis=0), np.cov(x, rowvar=0))


class Cauchy(Distribution):
    def __init__(self, loc: float, scale):
        self.loc = loc
        self.scale = scale
        self._distrib = cauchy(loc, scale)
    
    def fit(x: np.ndarray) -> 'Cauchy':
        raise NotImplementedError  # TODO
