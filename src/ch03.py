"""Chapter 3: Bracketing"""

import numpy as np

from typing import Callable


PHI = (1 + np.sqrt(5))/2  # golden ratio


def bracket_minimum(f: Callable[[float], float],
                    x: float = 0.0,
                    s: float = 1e-2,
                    k: float = 2.0) -> tuple[float, float]:
    """
    An algorithm for bracketing an interval in which a local minimum must exist.
    It takes as input a univariate function `f` and starting position `x`, which
    defaults to 0.0. The starting step size `s` and the expansion factor `k` can
    be specified. It returns a tuple containg the new interval [a, b].
    """
    a, y_a = x, f(x)
    b, y_b = a + s, f(a + s)
    if y_b > y_a:
        a, b, = b, a
        y_a, y_b = y_b, y_a
        s = -s
    while True:
        c, y_c = b + s, f(b + s)
        if y_c > y_b:
            return (a, c) if a < c else (c, a)
        a, y_a, b, y_b = b, y_b, c, y_c
        s *= k


def fibonacci_search(f: Callable[[float], float],
                     a: float,
                     b: float,
                     n: int,
                     eps: float = 0.01) -> tuple[float, float]:
    """
    Fibonacci search to be run on univariate function `f`, with bracketing
    interval `[a, b]` for n > 1 function evaluations. It returns the new
    interval [a, b]. The optimal parameter `eps` controls the lowest-level
    interval.
    """
    s = (1 - np.sqrt(5)) / (1 + np.sqrt(5))
    p = 1 / ((PHI*(1 - (s**(n + 1)))) / (1 - (s**n)))
    d = p*b + (1 - p)*a
    y_d = f(d)
    for i in range(1, n):
        if i == n - 1:
            c = eps*a + (1 - eps)*d
        else:
            c = p*a + (1 - p)*b
        y_c = f(c)
        if y_c < y_d:
            b, d, y_d = d, c, y_c
        else:
            a, b = b, c
        p = 1 / ((PHI*(1 - (s**(n - i + 1)))) / (1 - (s**(n - i))))
    return (a, b) if a < b else (b, a)


def golden_section_search(f: Callable[[float], float],
                          a: float,
                          b: float,
                          n: int) -> tuple[float, float]:
    """
    Golden section search to be run on a univariate function `f`, with
    bracketing interval [a, b], for n > 1 function evaluations. It returns the
    new interval (a, b). Guaranteeing convergence to within `eps` requires
    n = (b - a)/(eps*ln(PHI)) iterations. 
    """
    p = PHI - 1
    d = p*b + (1 - p)*a
    y_d = f(d)
    for _ in range(1, n):
        c = p*a + (1 - p)*b
        y_c = f(c)
        if y_c < y_d:
            b, d, y_d = d, c, y_c
        else:
            a, b = b, c
    return (a, b) if a < b else (b, a)


def quadratic_fit_search(f: Callable[[float], float],
                         a: float,
                         b: float,
                         c: float,
                         n: int) -> tuple[float, float, float]:
    """
    Quadratic fit search to be run on univariate function `f`, with bracketing
    interval [a, c] with a < b < c. The method will run for `n` function
    evaluations. It returns the new bracketing values as a tuple, `(a, b, c)`.
    """
    y_a, y_b, y_c = f(a), f(b), f(c)
    for i in range(1, n - 2):
        x = 0.5 * (y_a*(b**2 - c**2) + y_b*(c**2 - a**2) + y_c*(a**2 - b**2)) /\
                  (y_a*(b - c) + y_b*(c - a) + y_c*(a - b))
        y_x = f(x)
        if x > b:
            if y_x > y_b:
                c, y_c = x, y_x
            else:
                a, y_a, b, y_b = b, y_b, x, y_x
        elif x < b:
            if y_x > y_b:
                a, y_a = x, y_x
            else:
                c, y_c, b, y_b = b, y_b, x, y_x
    return (a, b, c)


def shubert_piyavskii(f: Callable[[float], float],
                      a: float,
                      b: float,
                      l: float,
                      eps: float,
                      delta: float = 0.01) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    The Shubert-Piyavskii method to be run on univariate function `f`, with
    bracketing interval `a` < `b` and Lipschitz constant `l`. The algorithm runs
    until the update is less than the tolerance `eps`. Both the best point and
    the set of uncertainty intervals are returned. The uncertainty intervals are
    returned as an array of `(a, b)` tuples. The parameter `delta` is a
    tolerance used to merge the uncertainty intervals.
    """
    def _get_sp_intersection(A: np.ndarray, B: np.ndarray, l: float) -> np.ndarray:
        t = ((A[1] - B[1]) - l*(A[0] - B[0])) / (2*l)
        return np.array([A[0] + t, A[1] - t*l])

    m = (a + b) / 2
    A, M, B = np.array([a, f(a)]), np.array([m, f(m)]), np.array([b, f(b)])
    pts = np.array([A, _get_sp_intersection(A, M, l),
                    M, _get_sp_intersection(M, B, l),
                    B])
    Delta = np.inf
    while Delta > eps:
        i = np.argmin(pts[:, 1])
        P = np.array([pts[i, 0], f(pts[i, 0])])
        Delta = P[1] - pts[i, 1]

        P_prev = _get_sp_intersection(pts[i - 1], P, l)
        P_next = _get_sp_intersection(P, pts[i + 1], l)

        pts = np.delete(pts, i)
        pts = np.insert(pts, i, P_next)
        pts = np.insert(pts, i, P)
        pts = np.insert(pts, i, P_prev)

    intervals = []
    P_min = pts[2 * np.argmin(pts[::2, 1])]
    y_min = P_min[1]
    for i in range(2, len(pts) + 1, 2):
        if pts[i, 1] < y_min:
            dy = y_min - pts[i, 1]
            x_lo = np.maximum(a, pts[i, 0] - (dy/l))
            x_hi = np.minimum(b, pts[i, 0] + (dy/l))
            if (len(intervals) != 0) and (intervals[-1][1] + delta >= x_lo):
                intervals[-1] = (intervals[-1][0], x_hi)
            else:
                intervals.append((x_lo, x_hi))
    
    return (P_min, intervals)


def bisection(f_prime: Callable[[float], float],
              a: float,
              b: float,
              eps: float) -> tuple[float, float]:
    """
    The bisection algorithm where `f_prime` is the derivative of the univariate
    function we seek to optimize. We have a < b that bracket a zero of `f_prime`.
    The interval width tolerance is `eps`. Calling `bisection` returns the new
    bracketed interval [a, b] as a tuple.
    """
    a, b = (b, a) if a > b else (a, b)  # ensure a < b

    y_a, y_b = f_prime(a), f_prime(b)
    b = a if y_a == 0 else b
    a = b if y_b == 0 else a

    while (b - a > eps):
        x = (a + b) / 2
        y = f_prime(x)
        if y == 0:
            a, b = x, x
        elif np.sign(y) == np.sign(y_a):
            a = x
        else:
            b = x
    
    return (a, b)


def bracket_sign_change(f_prime: Callable[[float], float],
                        a: float,
                        b: float,
                        k: float = 2.0) -> tuple[float, float]:
    """
    An algorithm for finding an interval in which a sign change occurs. The
    inputs are the real-valued function `f_prime` defined on the real numbers,
    and starting interval [a, b]. It returns the new interval as a tuple by
    expanding the interval width until there is a sign change between the
    function evaluated at the interval bounds. The expansion default factor `k`
    defaults to 2.0.
    """
    a, b = (b, a) if a > b else (a, b)  # ensure a < b

    center, half_width = (b + a) / 2, (b - a) / 2
    while (f_prime(a) * f_prime(b) > 0):
        half_width *= k
        a = center - half_width
        b = center + half_width
    
    return (a, b)
