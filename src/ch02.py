"""Chapter 2: Derivatives and Gradients"""

import numpy as np

from typing import Callable


def diff_forward(f: Callable[[np.ndarray | float], np.ndarray | float], 
                 x: np.ndarray | float, 
                 h: np.ndarray | float = np.sqrt(np.finfo(np.float64).eps)) -> np.ndarray | float:
    """Forward difference method for estimating the derivative of a
    function `f` at `x` with finite difference `h`. The default step size is
    the square root of the machine precision for floating point values. This
    step size balances machine round-off error with step size error.

    `np.finfo(np.float64).eps` provides the step size between 1.0 and the next
    larger representable floating-point value.
    """
    return (f(x + h) - f(x)) / h


def diff_central(f: Callable[[np.ndarray | float], np.ndarray | float],
                 x: np.ndarray | float,
                 h: np.ndarray | float = np.cbrt(np.finfo(np.float64).eps)) -> np.ndarray | float:
    """Central difference method for estimating the derivative of a
    function `f` at `x` with finite difference `h`. The default step size is
    the cube root of the machine precision for floating point values.
    """
    return (f(x + (h/2)) - f(x - (h/2))) / h


def diff_backward(f: Callable[[np.ndarray | float], np.ndarray | float],
                  x: np.ndarray | float,
                  h: np.ndarray | float = np.sqrt(np.finfo(np.float64).eps)) -> np.ndarray | float:
    """Backward difference method for estimating the derivative of a
    function `f` at `x` with finite difference `h`. The default step size is
    the square root of the machine precision for floating point values.
    """
    return (f(x) - f(x - h)) / h


def diff_complex(f: Callable[[np.ndarray | float], np.ndarray | float],
                 x: np.ndarray | float,
                 h: np.ndarray | float = 1e-20) -> np.ndarray | float:
    """The complex step method for estimating the derivative of a function `f`
    at `x` with finite difference `h`."""
    return np.imag(f(x + h*1j)) / h
