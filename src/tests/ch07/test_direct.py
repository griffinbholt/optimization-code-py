import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch07 import direct
from TestFunctions import branin

class TestDIRECT():
    def test(self, eps=1e-5):
        f_min, x_min = branin.global_min()
        x = direct(branin, a=np.array([-5.0, -5.0]), b=np.array([20.0, 20.0]), eps=eps, k_max=50)
        assert np.abs(branin(x) - f_min[0]) < eps
        assert np.any([np.all(np.abs(x - x_min_i) < eps) for x_min_i in x_min.T])
