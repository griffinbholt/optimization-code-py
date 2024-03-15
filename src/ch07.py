"""Chapter 7: Direct Methods"""

import numpy as np

from abc import ABC, abstractmethod
from collections import OrderedDict
from queue import PriorityQueue
from typing import Callable

from ch04 import line_search


def basis(i: int, n: int) -> np.ndarray:
    """A function for constructing the `i`th basis vector (zero-indexed) of length `n`"""
    return np.array([1.0 if k == i else 0.0 for k in range(n)])


def cyclic_coordinate_descent(f: Callable[[np.ndarray], float], 
                              x: np.ndarray,
                              eps: float,
                              with_acceleration: bool = False) -> np.ndarray:
    """
    The cyclic coordinate descent method (with or without acceleration) takes as
    input the objective function `f` and a starting point `x`, and it runs until
    the step size over a full cycle is less than a given tolerance `eps`.
    """
    delta, n = np.inf, len(x)
    while delta > eps:
        x_prev = x.copy()
        for i in range(n):
            d = basis(i, n)
            x = line_search(f, x, d)
        if with_acceleration:
            x = line_search(f, x, x - x_prev)  # acceleration step
        delta = np.linalg.norm(x - x_prev)
    return x


def powell(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float) -> np.ndarray:
    """
    Powell's method, which takes the objective function `f`, a starting point `x`,
    and a tolerance `eps`.
    """
    n = len(x)
    U = np.eye(n)
    delta = np.inf
    while delta > eps:
        x_prime = x.copy()
        for i in range(n):
            d = U[i]
            x_prime = line_search(f, x_prime, d)
        for i in range(n - 1):
            U[i] = U[i + 1]
        U[n - 1] = d = x_prime - x
        x_prime = line_search(f, x_prime, d)
        delta = np.linalg.norm(x_prime - x)
        x = x_prime
    return x


def hooke_jeeves(f: Callable[[np.ndarray], float], 
                 x: np.ndarray, 
                 alpha: float, 
                 eps: float, 
                 gamma: float = 0.5) -> np.ndarray:
    """
    The Hooke-Jeeves method, which takes the target function `f`, a starting point
    `x`, a starting step size `alpha`, a tolerance `eps`, and a step decay `gamma`.
    The method runs until the step size is less than `eps` and the points sampled
    along the coordinate directions do not provide an improvement. 

    Based on the implementation from A.F. Kaupe Jr, "Algorithm 178: Direct Search,"
    Communications of the ACM, vol. 6, no. 6, pp. 313-314, 1963.
    """
    y, n = f(x), len(x)
    while alpha > eps:
        improved = False
        x_best, y_best = x, y
        for i in range(n):
            for sgn in [-1, 1]:
                x_prime = x + sgn*alpha*basis(i, n)
                y_prime = f(x_prime)
                if y_prime < y_best:
                    x_best, y_best, improved = x_prime, y_prime, True
        x, y = x_best, y_best
        if not improved:
            alpha *= gamma
    return x


def generalized_pattern_search(f: Callable[[np.ndarray], float],
                               x: np.ndarray,
                               alpha: float,
                               D: np.ndarray,
                               eps: float,
                               gamma: float = 0.5) -> np.ndarray:
    """
    Generalized pattern search, which takes the target function `f`, a starting
    point `x`, a starting step size `alpha`, a set of search directions `D`, a
    tolerance `eps`, and a step decay `gamma`. The method runs until the step
    size is less than `eps` and the points sampled along the coordinate directions
    do not provide an improvement.
    """
    y = f(x)
    while alpha > eps:
        improved = False
        for i, d in enumerate(D):
            x_prime = x + alpha * d
            y_prime = f(x_prime)
            if y_prime < y:
                x, y, improved = x_prime, y_prime, True
                D = np.insert(np.delete(D, i, axis=0), 0, d, axis=0)
                break
        if not improved:
            alpha *= gamma
    return x


def nelder_mead(f: Callable[[np.ndarray], float],
                S: np.ndarray,
                eps: float,
                alpha: float = 1.0,
                beta: float = 2.0,
                gamma: float = 0.5) -> np.ndarray:
    """
    The Nelder-Mead simplex method, which takes the objective function `f`, a
    starting simplex `S` consisting of a list of vectors, and a tolerance `eps`.
    The Nelder-Mead parameters can be specified as well and default to recommended
    values.
    """
    delta, y_arr = np.inf, np.apply_along_axis(f, 1, S)
    while delta > eps:
        p = np.argsort(y_arr)         # sort lowest to highest
        S, y_arr = S[p], y_arr[p]
        xl, yl = S[0], y_arr[0]       # lowest
        xh, yh = S[-1], y_arr[-1]     # highest
        xs, ys = S[-2], y_arr[-2]     # second-highest
        xm = np.mean(S[:-1], axis=0)  # centroid
        xr = xm + alpha * (xm - xh)   # reflection point
        yr = f(xr)

        if yr < yl:
            xe = xm + beta * (xr - xm)   # expansion point
            ye = f(xe)
            S[-1], y_arr[-1] = (xe, ye) if ye < yr else (xr, yr)
        elif yr >= ys:
            if yr < yh:
                xh, yh, S[-1], y_arr[-1] = xr, yr, xr, yr
            xc = xm + gamma * (xh - xm)  # contraction point
            yc = f(xc)
            if yc > yh:
                for i in range(1, len(y_arr)):
                    S[i] = (S[i] + xl) / 2
                    y_arr[i] = f(S[i])
            else:
                S[-1], y_arr[-1] = xc, yc
        else:
            S[-1], y_arr[-1] = xr, yr
        
        delta = np.std(y_arr)
    return S[np.argmin(y_arr)]


def direct(f: Callable[[np.ndarray], float],
           a: np.ndarray,
           b: np.ndarray,
           eps: float,
           k_max: int) -> np.ndarray:
    """
    DIRECT, which takes the multidimensional objective function `f`, vector of
    lower bounds `a`, vector of upper bounds `b`, tolerance parameter `eps`, and
    number of iterations `k_max`. It returns the best coordinate.
    """
    g = reparametrize_to_unit_hypercube(f, a, b)
    intervals = Intervals()
    n = len(a)
    c = np.full(n, 0.5)
    interval = Interval(c, g(c), np.zeros(n))
    intervals.add_interval(interval)
    c_best, y_best = np.copy(interval.c), interval.y

    for _ in range(k_max):
        S = intervals.get_opt_intervals(eps, y_best)  # TODO - Why is y_best needed?
        to_add = []
        for interval in S:
            to_add.extend(interval.divide(g))
            intervals[interval.vertex_dist()].get()
        for interval in to_add:
            intervals.add_interval(interval)
            if interval.y < y_best:
                c_best, y_best = np.copy(interval.c), interval.y
    
    return rev_unit_hypercube_parametrization(c_best, a, b)


def rev_unit_hypercube_parametrization(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x * (b - a) + a


def reparametrize_to_unit_hypercube(f: Callable[[np.ndarray], float], a: np.ndarray, b: np.ndarray) -> Callable[[np.ndarray], float]:
    """
    A function that creates a function defined over the unit hypercube that
    is a reparametrized version of the function `f` defined over the
    hypercube with lower and upper bounds `a` and `b`.
    """
    Delta = b - a
    return lambda x: f(x * Delta + a)


class Interval():
    """
    `Interval` has three fields: the interval center `c`, the center point
    `y = f(c)`, and the number of divisions in each dimension `depths`.
    """
    def __init__(self, c: np.ndarray, y: float, depths: np.ndarray):
        self.c = c
        self.y = y
        self.depths = depths

    def __lt__(self, other: 'Interval'):
        return self.y < other.y
    
    def min_depth(self):
        return np.min(self.depths)

    def vertex_dist(self):
        return np.linalg.norm(0.5 * (3.0**(-self.depths)))

    def divide(self, f: Callable[[np.ndarray], float]) -> list['Interval']:
        """The `divide` routine for dividing an interval, where `f` is the
        objective function and `self` is the interval to be divided. It
        returns a list of the resulting smaller intervals."""
        c, d, n = self.c, self.min_depth(), len(self.c)
        dirs = np.where(self.depths == d)[0]
        cs = np.array([[c + (3.0**(-d-1)) * basis(i, n), 
                        c - (3.0**(-d-1)) * basis(i, n)] for i in dirs])
        vs = np.apply_along_axis(f, 2, cs)
        minvals = np.min(vs, axis=1)

        intervals = []
        depths = np.copy(self.depths)
        for j in np.argsort(minvals):
            depths[dirs[j]] += 1
            C, V = cs[j], vs[j]
            intervals.append(Interval(C[0], V[0], np.copy(depths)))
            intervals.append(Interval(C[1], V[1], np.copy(depths)))
        intervals.append(Interval(c, self.y, np.copy(depths)))
        return intervals

class Intervals(OrderedDict[float, PriorityQueue[tuple[float, Interval]]]):
    """The data structure used in DIRECT"""
    def add_interval(self, interval: Interval):
        """Inserts a new `Interval` into the data structure."""
        d = interval.vertex_dist()
        if d not in self.keys():
            self[d] = PriorityQueue()
        self[d].put((interval.y, interval))

    def get_opt_intervals(self, eps: float, y_best: float) -> list[Interval]:  # TODO - y_best isn't used?
        """A routine for obtaining the potentially optimal intervals, where `eps`
        is a tolerance parameter and `y_best` is the best function evaluation."""
        stack = []
        for (x, pq) in self.items():
            if not pq.empty():
                interval = pq.queue[0][1]
                y = interval.y

                while len(stack) > 1:
                    interval1 = stack[-1]
                    interval2 = stack[-2]
                    x1, y1 = interval1.vertex_dist(), interval1.y
                    x2, y2 = interval2.vertex_dist(), interval2.y
                    l = (y2 - y) / (x2 - x)
                    if (y1 <= l * (x1 - x) + y + eps): # TODO: and (y1 <= l * x1 + y_best - eps*np.abs(y_best)):
                        break
                    stack.pop()  # remove previous interval
                
                if (len(stack) != 0) and (interval.y > stack[-1].y + eps):
                    continue  # skip new interval

                stack.append(interval)  # add new interval
        return stack
