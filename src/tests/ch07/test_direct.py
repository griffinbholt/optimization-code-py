import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch07 import direct
from TestFunctions import ackley, booth, branin, flower, michalewicz, rosenbrock, wheeler

class TestDIRECT():
    def test(self, eps=1e-5):
        # Ackley's Function
        f_min, x_min = ackley.global_min()
        x = direct(ackley, a=np.array([-30.0, -30.0]), b=np.array([30.0, 30.0]), eps=eps, k_max=50)
        assert np.abs(ackley(x) - f_min) < eps
        assert np.all(np.abs(x - x_min) < eps)

        # Booth's Function
        f_min, x_min = booth.global_min()
        x = direct(booth, a=np.array([-10.0, -10.0]), b=np.array([10.0, 10.0]), eps=eps, k_max=40)
        assert np.abs(booth(x) - f_min) < eps
        assert np.all(np.abs(x - x_min) < eps)

        # Branin's Function
        f_min, x_min = branin.global_min()
        x = direct(branin, a=np.array([-5.0, -5.0]), b=np.array([20.0, 20.0]), eps=eps, k_max=50)
        assert np.abs(branin(x) - f_min[0]) < eps
        assert np.any([np.all(np.abs(x - x_min_i) < eps) for x_min_i in x_min.T])

        # Michalewicz Function
        f_min, x_min = michalewicz.global_min()
        x = direct(michalewicz, a=np.array([0.0, 0.0]), b=np.array([4.0, 4.0]), eps=eps, k_max=50)
        assert np.abs(michalewicz(x) - f_min) < eps
        assert np.all(np.abs(x - x_min) < eps)

        # Flower Function
        x = direct(flower, a=np.array([-3.0, -3.0]), b=np.array([3.0, 3.0]), eps=eps, k_max=50)
        assert np.all(np.abs(x - np.zeros(2)) < eps)

        # Rosenbrock's Banana Function
        f_min, x_min = rosenbrock.global_min()
        x = direct(rosenbrock, a=np.array([-2.0, -2.0]), b=np.array([2.0, 2.0]), eps=eps, k_max=50)
        assert np.abs(rosenbrock(x) - f_min) < eps
        assert np.all(np.abs(x - x_min) < eps)

        # Wheeler's Ridge
        f_min, x_min = wheeler.global_min()
        x = direct(wheeler, a=np.array([-5.0, -2.0]), b=np.array([25.0, 6.0]), eps=eps, k_max=50)
        assert np.abs(wheeler(x) - f_min) < eps
        assert np.all(np.abs(x - x_min) < eps)
