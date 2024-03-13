import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numdifftools as nd
import numpy as np

from TestFunctions import ScalarValuedTestFunction, MichalewiczFunction, VectorValuedTestFunction,\
                          ackley, booth, branin, flower, michalewicz, rosenbrock,\
                          wheeler, circle


class TestTestFunctions():
    scalar_functions = [ackley, booth, branin, flower,
                        michalewicz, rosenbrock, wheeler]
    vector_functions = [circle]

    def test_gradients(self):
        np.random.seed(42)
        for test_function in self.scalar_functions:
            self.run_gradient_test(test_function)

    def run_gradient_test(self, test_function: ScalarValuedTestFunction, eps: float = 1e-9, n_trials: int = 100):
        for _ in range(n_trials):
            x = np.random.rand(test_function.d if test_function.d is not None else 10)
            num_grad = nd.Gradient(test_function)(x)
            test_grad = test_function.grad(x)
            assert np.all(np.abs(num_grad - test_grad) < eps), test_function.__class__.__name__ + " Gradient failed"

    def test_hessians(self):
        np.random.seed(42)
        for test_function in self.scalar_functions:
            if isinstance(test_function, MichalewiczFunction):
                self.run_hessian_test(test_function, eps=0.5)
            else:
                self.run_hessian_test(test_function)

    def run_hessian_test(self, test_function: ScalarValuedTestFunction, eps: float = 1e-3, n_trials: int = 100):
        for _ in range(n_trials):
            x = np.random.rand(2)
            num_hess = nd.Hessian(test_function)(x)
            test_hess = test_function.hess(x)
            assert np.all(np.abs(num_hess - test_hess) < eps), test_function.__class__.__name__ + " Hessian failed"

    def test_jacobians(self):
        np.random.seed(42)
        for test_function in self.vector_functions:
            self.run_jacobian_test(test_function)

    def run_jacobian_test(self, test_function: VectorValuedTestFunction, eps: float = 1e-9, n_trials: int = 100):
        for _ in range(n_trials):
            x = np.random.rand(2)
            num_jac = nd.Jacobian(test_function)(x)
            test_jac = test_function.jac(x)
            assert np.all(np.abs(num_jac - test_jac) < eps), test_function.__class__.__name__ + " Jacobian failed"
