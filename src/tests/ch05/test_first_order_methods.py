import sys; sys.path.append('./src/'); sys.path.append('../../')

import numpy as np

from ch05 import *
from TestFunctions import booth, branin, michalewicz, rosenbrock, wheeler

class TestFirstOrderMethods():
    def test_gradient_descent(self, eps: float = 1e-8):
        M = GradientDescent(alpha=0.001)
        self.run_on(booth, max_steps=100000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(rosenbrock, max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)

    def test_conjugate_gradient(self, eps: float = 1e-6):
        M = ConjugateGradientDescent()
        self.run_on(booth, max_steps=2, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=10, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(michalewicz, max_steps=5, x=np.ones(2), M=M, eps=1e-4)
        self.run_on(rosenbrock, max_steps=10, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(wheeler, max_steps=10, x=np.zeros(2), M=M, eps=eps)

    def test_momentum(self, eps: float = 1e-8):
        M = Momentum(alpha=0.001, beta=0.9)
        self.run_on(booth, max_steps=1000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=1000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(rosenbrock, max_steps=10000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(wheeler, max_steps=10000, x=np.zeros(2), M=M, eps=eps)   

    def test_nesterov_momentum(self, eps: float = 1e-8):
        M = NesterovMomentum(alpha=0.001, beta=0.9)
        self.run_on(booth, max_steps=1000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=1000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(rosenbrock, max_steps=10000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(wheeler, max_steps=10000, x=np.zeros(2), M=M, eps=eps)

    def test_adagrad(self, eps: float = 1e-8):
        M = Adagrad(alpha=0.1, eps=1e-3)
        self.run_on(booth, max_steps=100000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)
        M = Adagrad(alpha=1.0, eps=1e-3)
        self.run_on(rosenbrock, max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(wheeler, max_steps=1000, x=np.zeros(2), M=M, eps=eps)

    def test_rmsprop(self, eps: float = 1e-3):
        M = RMSProp(alpha=0.001, gamma=0.9, eps=1e-3)
        self.run_on(booth, max_steps=10000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=10000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(rosenbrock, max_steps=10000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(wheeler, max_steps=10000, x=np.zeros(2), M=M, eps=1e-2)

    def test_adadelta(self, eps: float = 1e-8):
        M = Adadelta(gamma_s=0.95, gamma_x=0.95, eps=1e-3)
        self.run_on_branin(max_steps=1000, x=np.ones(2)*-5, M=M, eps=1e-3)

    def test_adam(self, eps: float = 1e-8):
        M = Adam(alpha=0.001, gamma_v=0.9, gamma_s=0.999, eps=1e-8)
        self.run_on(booth, max_steps=100000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        self.run_on_branin(max_steps=100000, x=np.ones(2)*-5, M=M, eps=1e-4)
        self.run_on(rosenbrock, max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)
        self.run_on(wheeler, max_steps=100000, x=np.zeros(2), M=M, eps=eps)
        self.run_on(michalewicz, max_steps=100000, x=np.ones(2), M=M, eps=1e-4)

    def test_hypergradient_descent(self, eps: float = 1e-8):
        M = HyperGradientDescent(alpha_0=0.00001, mu=0.00001)
        self.run_on(booth, max_steps=1000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        M = HyperGradientDescent(alpha_0=0.000001, mu=0.000001)
        self.run_on_branin(max_steps=1000, x=np.ones(2)*-5, M=M, eps=eps)
        M = HyperGradientDescent(alpha_0=0.0001, mu=0.0000000001)
        self.run_on(rosenbrock, max_steps=100000, x=np.ones(2)*-5, M=M, eps=eps)
        M = HyperGradientDescent(alpha_0=0.0001, mu=0.00001)
        self.run_on(wheeler, max_steps=100000, x=np.zeros(2), M=M, eps=eps)

    def test_hypernesterov_momentum(self, eps: float = 1e-8):
        M = HyperNesterovMomentum(alpha_0=0.000001, mu=0.000001, beta=0.9)
        self.run_on(booth, max_steps=1000, x=np.array([-5.0, 5.0]), M=M, eps=eps)
        M = HyperNesterovMomentum(alpha_0=0.0000001, mu=0.0000001, beta=0.9)
        self.run_on_branin(max_steps=1000, x=np.ones(2)*-5, M=M, eps=eps)
        M = HyperNesterovMomentum(alpha_0=0.0001, mu=0.0000000001, beta=0.9)
        self.run_on(rosenbrock, max_steps=10000, x=np.ones(2)*-5, M=M, eps=eps)
        M = HyperNesterovMomentum(alpha_0=0.01, mu=0.000001, beta=0.9)
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
