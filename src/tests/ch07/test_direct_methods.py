import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch07 import cyclic_coordinate_descent, powell, hooke_jeeves, generalized_pattern_search, nelder_mead
from TestFunctions import booth, wheeler

class TestDirectMethods():
    eps = 1e-5

    def test_cyclic_coord_descent(self):
        f_min, x_min = booth.global_min()
        x = np.array([10.0, -5.0])
        x = cyclic_coordinate_descent(booth, x, self.eps)
        assert np.abs(booth(x) - f_min) < self.eps
        assert np.all(np.abs(x - x_min) < self.eps)

    def test_cyclic_coord_descent_with_accel(self):
        f_min, x_min = booth.global_min()
        x = np.array([10.0, -5.0])
        x = cyclic_coordinate_descent(booth, x, self.eps, with_acceleration=True)
        assert np.abs(booth(x) - f_min) < self.eps
        assert np.all(np.abs(x - x_min) < self.eps)

    def test_powell(self):
        f_min, x_min = booth.global_min()
        x = np.array([10.0, -5.0])
        x = powell(booth, x, self.eps)
        assert np.abs(booth(x) - f_min) < self.eps
        assert np.all(np.abs(x - x_min) < self.eps)

    def test_hooke_jeeves(self):
        f_min, x_min = wheeler.global_min()
        x = np.array([0.7, 0.9])
        x = hooke_jeeves(wheeler, x, alpha=0.5, eps=self.eps, gamma=0.5)
        assert np.abs(wheeler(x) - f_min) < self.eps
        assert np.all(np.abs(x - x_min) < self.eps)

    def test_generalized_pattern_search(self):
        pass  # TODO

    def test_nelder_mead(self):
        pass  # TODO
