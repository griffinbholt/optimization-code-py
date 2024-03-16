import sys; sys.path.append('./src/'); sys.path.append('../../')

import cvxpy as cp
import numpy as np
import warnings

from ch05 import GradientDescent, Adam, HyperNesterovMomentum
from ch08 import *
from TestFunctions import booth, branin, rosenbrock, wheeler


class TestStochasticMethods():
    def test_noisy_descent(self, eps=1e-8):
        np.random.seed(42)
        def sigma(k): return 1/(k**3)

        M = NoisyDescent(GradientDescent(alpha=0.001), sigma)
        self.run_on(booth, max_steps=100000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(rosenbrock, max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)

        M = NoisyDescent(Adam(alpha=0.001, gamma_v=0.9, gamma_s=0.999, eps=1e-8), sigma)
        self.run_on(booth, max_steps=100000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=100000, x=np.ones(2)*-5, M=M, eps=1e-4)
        self.run_on(rosenbrock, max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)

        M = NoisyDescent(HyperNesterovMomentum(alpha_0=0.01, mu=0.000001, beta=0.9), sigma)
        self.run_on(wheeler, max_steps=1000, x=np.zeros(2), M=M, eps=eps)

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

    def test_rand_positive_spanning_set(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        for alpha in [1.0, 0.25, 0.25/4, 0.25/16]:
            for n in [2, 3, 5, 10, 100]:
                D = rand_positive_spanning_set(alpha, n).T
                assert np.linalg.matrix_rank(D) == n  # full row rank

                x = cp.Variable(n + 1)
                constraints = [D @ x == -D @ np.ones(n + 1), x >= 0]
                problem = cp.Problem(cp.Minimize(0), constraints)
                problem.solve()
                assert problem.status == "optimal"  # Dx = -D1, x >= 0 is feasible

    def test_mesh_adaptive_direct_search(self):
        f_min, x_min = wheeler.global_min()
        x = np.array([0.7, 0.9])
        x = mesh_adaptive_direct_search(wheeler, x, eps=1e-8)
        assert np.abs(wheeler(x) - f_min) < 1e-6
        assert np.all(np.abs(x - x_min) < 1e-3)

    def test_simulated_annealing(self):
        pass





    def test_corana_update(self):
        pass

    def test_adaptive_simulated_annealing(self):
        pass

    def test_cross_entropy_method(self):
        pass

    def test_natural_evolution_strategies(self):
        pass

    def test_covariance_matrix_adaptation(self):
        pass

test = TestStochasticMethods()
test.test_mesh_adaptive_direct_search()