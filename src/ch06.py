"""Chapter 6: Second-Order Methods"""

import numpy as np

from typing import Callable

from ch05 import DescentMethod


def newtons_method():
    """
    Newton's method, which takes the gradient of the function `grad_f`,
    the Hessian of the objective function `H`, an initial point `x`, a step size
    tolerance `eps`, and a maximum number of iterations `k_max`.
    """
    pass  # TODO


def secant_method():
    """
    The secant method for univariate function minimization. The inputs are the
    first derivative `f_prime` of the target function, two initial points `x0`
    and `x1`, and the desired tolerance `eps`. The final x-coordinate is
    returned.
    """
    pass  # TODO


class QuasiNewtonMethod(DescentMethod):
    """
    Just as the secant method approximates f'' in the univariate case,
    quasi-Newton methods approximate the inverse Hessian.
    """
    pass


class DFP(QuasiNewtonMethod):
    """The Davidon-Fletcher-Powell descent method"""
    def __init__(self, Q: np.ndarray):
        self.Q = Q  # approximate inverse Hessian

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        m = len(x)
        self.Q = np.eye(m)

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        pass  # TODO


class BFGS(DFP):
    """The Broyden-Fletcher-Goldfarb-Shanno descent method"""
    def __init__(self, Q: np.ndarray):
        super().__init__(Q)

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        pass  # TODO


class LimitedMemoryBFGS(QuasiNewtonMethod):
    """
    The Limited-memory BFGS descent method, which avoids storing the approximate
    inverse Hessian. The parameter `m` determines the history size. It also
    stores the step differences `deltas`, the gradient changes `gammas`, and
    storage vectors `qs`.
    """
    def __init__(self):
        pass  # TODO

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        pass  # TODO

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        pass  # TODO
