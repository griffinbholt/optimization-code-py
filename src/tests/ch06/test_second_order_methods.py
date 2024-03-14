import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch06 import newtons_method, secant_method, DFP, BFGS, LimitedMemoryBFGS
from TestFunctions import booth

class TestSecondOrderMethods():
    def test_newtons_method(self, eps=1e-8):
        f_min, x_min = booth.global_min()
        x = np.array([9.0, 8.0])
        x_prime = newtons_method(booth.grad, booth.hess, x, eps=1e-5, k_max=1)
        assert np.abs(booth(x_prime) - f_min) < eps
        assert np.all(np.abs(x_prime - x_min) < eps)

    def test_secant_method(self, eps=1e-8):
        pass  # TODO

    def test_DFP(self, eps=1e-8):
        pass  # TODO

    def test_BFGS(self, eps=1e-8):
        pass

    def test_limited_memory_BFGS(self, eps=1e-8):
        pass
