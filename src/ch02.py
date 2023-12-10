"""Chapter 2: Derivatives and Gradients"""

import numpy as np

from typing import Callable


def diff_forward(f: Callable[[float | np.ndarray], float | np.ndarray], 
                 x: float | np.ndarray, 
                 h: float | np.ndarray = np.sqrt(np.finfo(np.float64).eps)) -> float | np.ndarray:
    """Forward difference method for estimating the derivative of a
    function `f` at `x` with finite difference `h`. The default step size is
    the square root of the machine precision for floating point values. This
    step size balances machine round-off error with step size error.

    `np.finfo(np.float64).eps` provides the step size between 1.0 and the next
    larger representable floating-point value.
    """
    return (f(x + h) - f(x)) / h


def diff_central(f: Callable[[float | np.ndarray], float | np.ndarray],
                 x: float | np.ndarray,
                 h: float | np.ndarray = np.cbrt(np.finfo(np.float64).eps)) -> float | np.ndarray:
    """Central difference method for estimating the derivative of a
    function `f` at `x` with finite difference `h`. The default step size is
    the cube root of the machine precision for floating point values.
    """
    return (f(x + (h/2)) - f(x - (h/2))) / h


def diff_backward(f: Callable[[float | np.ndarray], float | np.ndarray],
                  x: float | np.ndarray,
                  h: float | np.ndarray = np.sqrt(np.finfo(np.float64).eps)) -> float | np.ndarray:
    """Backward difference method for estimating the derivative of a
    function `f` at `x` with finite difference `h`. The default step size is
    the square root of the machine precision for floating point values.
    """
    return (f(x) - f(x - h)) / h


def diff_complex(f: Callable[[float | np.ndarray], float | np.ndarray],
                 x: float | np.ndarray,
                 h: float | np.ndarray = 1e-20) -> float | np.ndarray:
    """The complex step method for estimating the derivative of a function `f`
    at `x` with finite difference `h`."""
    return np.imag(f(x + h*1j)) / h
