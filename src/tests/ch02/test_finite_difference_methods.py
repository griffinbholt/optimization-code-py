import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch02 import diff_forward, diff_central, diff_backward, diff_complex


class TestFiniteDifferenceMethods():
    tol = 1e-7
    
    def test(self):
        for diff in [diff_forward, diff_central, diff_backward, diff_complex]:
            self.run_test_finite_difference_method(diff)

    def run_test_finite_difference_method(self, diff):
        x = np.linspace(-100, 100, 1000)
        assert np.all(np.abs(np.cos(x) - diff(np.sin, x)) < self.tol)
