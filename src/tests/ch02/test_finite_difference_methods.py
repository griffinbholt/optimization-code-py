import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch02 import diff_forward, diff_central, diff_backward, diff_complex


class TestFiniteDifferenceMethods():
    tol = 1e-7

    def test_diff_forward(self):
        self.run_test_finite_difference_method(diff_forward)

    def test_diff_central(self):
        self.run_test_finite_difference_method(diff_central)

    def test_diff_backward(self):
        self.run_test_finite_difference_method(diff_backward)

    def test_diff_complex(self):
        self.run_test_finite_difference_method(diff_complex)

    def run_test_finite_difference_method(self, diff):
        x = np.linspace(-100, 100, 1000)
        assert np.all(np.abs(np.cos(x) - diff(np.sin, x)) < self.tol)
