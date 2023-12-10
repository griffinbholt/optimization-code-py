import numpy as np

class RosenbrockFunction():
    def __call__(self, x: np.ndarray, a: float = 1, b: float = 5) -> float:
        """The Rosenbrock function with two-dimensional input vector `x`
        and two optinal parameters."""
        return (a - x[0])**2 + b*(x[1] - (x[0]**2))**2

    def grad(self, x: np.ndarray, a: float = 1, b: float = 5) -> float:
        dx1 = 2 * ((2*b*(x[0]**3)) - (2*b*x[0]*x[1]) + x[0] - a)
        dx2 = 2 * b * (x[1] - (x[0]**2))
        return np.array([dx1, dx2])

    def hess(self, x: np.ndarray, a: float = 1, b: float = 5) -> float:
        dx1dx1 = 12*b*(x[0]**2) - 4*b*x[1] + 2
        dx1dx2 = -4*b*x[0]
        dx2dx2 = 2*b
        return np.array([[dx1dx1, dx1dx2], [dx1dx2, dx2dx2]])

rosenbrock = RosenbrockFunction()