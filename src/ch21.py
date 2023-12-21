"""Chapter 21: Multidisciplinary Optimization"""

import numpy as np

from copy import deepcopy
from typing import Callable


def gauss_seidel(Fs: list[Callable[[dict[str, float]]]],
                 A: dict[str, float],
                 k_max: int = 100,
                 eps: float = 1e-4) -> tuple[dict[str, float], bool]:
    """
    The Gauss-Seidel algorithm for conducting a multidiciplinary analysis.
    Here, `Fs` is a list of disciplinary analysis functions that take and modify
    an assignment `A`. There are two optional arguments: the maximum number of
    iterations `k_max` and the relative error tolerance `eps`. The method
    returns the modified assignment and whether it converged.
    """
    k, converged = 0, False
    while (not converged) and (k <= k_max):
        k += 1
        A_old = deepcopy(A)
        for F in Fs:
            F(A)
        converged = np.all([np.isclose(A[v], A_old[v], rtol=eps) for v in A])
    return (A, converged)
