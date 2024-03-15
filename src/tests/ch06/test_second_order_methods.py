import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch06 import newtons_method, secant_method, DFP, BFGS, LimitedMemoryBFGS
from TestFunctions import booth, branin, rosenbrock, wheeler

class TestSecondOrderMethods():
    def test_newtons_method(self, eps=1e-8):
        f_min, x_min = booth.global_min()
        x = np.array([9.0, 8.0])
        x_prime = newtons_method(booth.grad, booth.hess, x, eps=1e-5, k_max=1)
        assert np.abs(booth(x_prime) - f_min) < eps
        assert np.all(np.abs(x_prime - x_min) < eps)

    def test_secant_method(self, eps=1e-8):
        def f(x): return np.exp(x) + np.exp(-x) - 3*x + 2
        def f_prime(x): return np.exp(x) - np.exp(-x) - 3
        x_min = np.log((3 + np.sqrt(13))/2)
        f_min = f(x_min)

        x0, x1 = -4, -3
        x = secant_method(f_prime, x0, x1, eps)
        assert np.abs(x - x_min) < eps
        assert np.abs(f(x) - f_min) < eps

    def test_DFP(self, eps=1e-8):
        M = DFP()
        self.run_on(booth, max_steps=2, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=7, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(rosenbrock, max_steps=10, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(wheeler, max_steps=10, x=np.zeros(2), M=M, eps=eps)

    def test_BFGS(self, eps=1e-8):
        M = BFGS()
        self.run_on(booth, max_steps=2, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=7, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(rosenbrock, max_steps=10, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(wheeler, max_steps=10, x=np.zeros(2), M=M, eps=eps)

    def test_limited_memory_BFGS(self, eps=1e-4):
        for m in range(1, 4):
            M = LimitedMemoryBFGS(m)
            self.run_on(booth, max_steps=2, x=np.array([-5.0, 5.0]), M=M, eps=eps)
            self.run_on_branin(max_steps=7, x=np.ones(2)*-5, M=M, eps=eps)
            self.run_on(rosenbrock, max_steps=10, x=np.ones(2)*-5, M=M, eps=eps)
            with np.errstate(over="ignore", invalid="ignore"):
                self.run_on(wheeler, max_steps=6, x=np.ones(2)*5, M=M, eps=eps)

    def run_on(self, f, max_steps, x, M, eps):
        f_min, x_min = f.global_min()
        M.initialize(f, f.grad, x)
        for _ in range(max_steps):
            x = M.step(f, f.grad, x)
        assert np.abs(f(x) - f_min) < eps
        assert np.all(np.abs(x - x_min) < eps)

    def run_on_branin(self, max_steps, x, M, eps):
        f_min, x_min = branin.global_min()
        M.initialize(branin, branin.grad, x)
        for _ in range(max_steps):
            x = M.step(branin, branin.grad, x)
        assert np.abs(branin(x) - f_min[0]) < eps
        assert np.any([np.all(np.abs(x - x_min_i) < eps) for x_min_i in x_min.T])
