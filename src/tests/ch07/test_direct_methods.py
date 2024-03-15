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
        possible_Ds = [
            np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]]),  # Equivalent to Hooke-Jeeves
            np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]),  # Diagonal Directions
            np.array([[1, 0], [0, 1], [-1, -1]])             # 3 Directions: Up, Right, Down-Left
        ]
        for D in possible_Ds:
            f_min, x_min = wheeler.global_min()
            x = np.array([0.7, 0.9])
            x = generalized_pattern_search(wheeler, x, alpha=0.5, D=D, eps=self.eps, gamma=0.5)
            assert np.abs(wheeler(x) - f_min) < 10*self.eps
            assert np.all(np.abs(x - x_min) < 10*self.eps)
    
    def test_nelder_mead(self):
        f_min, x_min = wheeler.global_min()
        S = np.array([[0.7, 1.4], [0.7, 0.9], [0.4, 0.7]])
        x = nelder_mead(wheeler, S, eps=self.eps)
        assert np.abs(wheeler(x) - f_min) < 1e-4
        assert np.all(np.abs(x - x_min) < 1e-2)
