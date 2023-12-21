"""Chapter 19: Discrete Optimization"""

import networkx as nx
import numpy as np

from itertools import combinations
from queue import PriorityQueue

from ch11 import LinearProgram
from convenience import normalize


class MixedIntegerProgram():
    """
    A mixed integer linear program type that reflects the following equation:

    minimize    c'x
    subject to: Ax = b
                x >= 0
                x_D \in Z^||D||

    Here, `D` is the set of design indices constrained to be discrete.
    """
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, D: np.ndarray):
        self.A = A
        self.b = b
        self.c = c
        self.D = D

    def relax(self) -> LinearProgram:
        """A method for relaxing a mixed integer linear program into a linear program"""
        return LinearProgram(self.A, self.b, self.c)
    
    def round(self) -> np.ndarray:
        """
        A method for solving a mixed integer linear program by rounding.
        The solution obtained by rounding may be suboptimal or infeasible.
        """
        x = self.relax().minimize()
        for i in self.D:
            x[i] = round(x[i])
        return x

    def is_totally_unimodular(self) -> bool:
        """Method for determining whether a mixed integer program is totally unimodular"""
        return is_totally_unimodular(self.A) and\
               np.all(isintegral(self.b)) and np.all(isintegral(self.c))


def isintegral(x: float | np.ndarray, eps=1e-10) -> bool | np.ndarray:
    """Returns true if the given value is integral"""
    return np.abs(np.round(x) - x) <= eps


def is_totally_unimodular(A: np.ndarray) -> bool:
    """Method for determining whether matrices `A` are totally unimodular"""
    # all entries must be in [0, 1, -1]
    if np.any([a not in [0, -1, 1] for a in A]):
        return False
    # brute force check every subdeterminant
    r, c = A.shape
    for i in range(1, min(r, c) + 1):
        for a in combinations(range(r), i):
            for b in combinations(range(c), i):
                B = A[a, b]
                if np.linalg.det(B) not in [0, -1, 1]:  # TODO Check this closer (for approximate values)
                    return False
    return True


def frac(x: float):
    """Returns the fractional part of a number"""
    return np.modf(x)[0]


def cutting_plane(MIP: MixedIntegerProgram) -> np.ndarray:
    """
    The cutting plane method solves a given mixed integer program `MIP` and
    returns an optimal design vector. An error is thrown if no feasible solution
    exists.
    """
    LP = MIP.relax()
    x, b_inds, v_inds = LP.minimize(return_idcs=True)
    n_orig = len(x)
    D = np.copy(MIP.D)
    while not np.all(isintegral(x[D])):
        AB, AV = LP.A[:, b_inds], LP.A[:, v_inds]
        Abar = np.linalg.solve(AB, AV)
        b = 0
        for i in D:
            if not isintegral(x[i]):
                b += 1
                A2 = np.vstack([np.hstack([LP.A, np.zeros((len(LP.A), 1))]),
                                np.zeros((1, LP.A.shape[1] + 1))])
                A2[-1, -1] = 1
                A2[-1, v_inds] = np.floor(Abar[b,:]) - Abar[b,:]
                b2 = np.append(LP.b, -frac(x[i]))
                c2 = np.append(LP.c, 0)
                LP = LinearProgram(A2, b2, c2)
        x, b_inds, v_inds = LP.minimize(return_idcs=True)
    return x[:n_orig]


def branch_and_bound(MIP: MixedIntegerProgram) -> np.ndarray:
    """
    The branch and bound algorithm for solving a mixed integer program `MIP`.
    More sophisticated implementations will drop variables whose solutions are
    known in order to speed computation.

    The `PriorityQueue` type is provided by the Python `queue` library.
    """
    LP = MIP.relax()
    x, y = LP.minimize_lp_and_y()
    n = len(x)
    x_best, y_best, Q = np.copy(x), np.inf, PriorityQueue()
    Q.put((y, (LP, x, y)))
    while not Q.empty():
        LP, x, y = Q.get()
        if np.any(np.isnan(x)) or np.all(isintegral(x[MIP.D])):
            if y < y_best:
                x_best, y_best = x[:n], y
        else:
            i = np.argmax(np.abs(x[MIP.D] - np.round(x[MIP.D])))  # TODO - Not convinced this gets the right index
            A, b, c = LP.A, LP.b, LP.c
            c2 = np.append(c, 0)
            for r in [1, -1]:  # x_i <= floor(x_i), then x_i >= ceil(x_i)
                A2 = np.vstack([np.hstack([A, np.zeros((len(A), 1))]),
                                np.array([[j == i for j in range(A.shape[1])] + [r]])])
                b2 = np.append(b, np.floor(x[i]) if r == 1 else np.ceil(x[i]))
                LP2 = LinearProgram(A2, b2, c2)
                x2, y2 = LP2.minimize_lp_and_y()
                if y2 <= y_best:
                    Q.put((y2, (LP2, x2, y2)))
    return x_best


def padovan_topdown(n: int, P: dict[int, int] = dict()) -> int:
    """Computing the Padovan sequence using dynamic programming, with the top-down approach"""
    if n not in P:
        P[n] = 1 if n < 3 else padovan_topdown(n - 2, P) + padovan_topdown(n - 3, P)
    return P[n]


def padovan_bottomup(n: int) -> int:
    """Computing the Padovan sequence using dynamic programming, with the bottom-up approach"""
    P = {0:1, 1:1, 2:1}
    for i in range(3, n + 1):
        P[i] = P[i - 2] + P[i - 3]
    return P[n]


def knapsack(v: np.ndarray, w: np.ndarray, w_max: float) -> np.ndarray:
    """
    A method for solving the 0-1 knapsack problem with item values `v`,
    integral item weights `w`, and integral capacity `w_max`. Recovering the
    design vector from the cached solutions requires additional iteration.
    """
    n = len(v)
    y = {(0, j): 0.0 for j in range(w_max + 1)}
    for i in range(n):
        for j in range(w_max + 1):
            y[(i, j)] = y[(i - 1, j)] if w[i] > j else max(y[(i - 1, j)], y[(i - 1, j - w[i])] + v[i])
    
    # recover solution
    x, j = np.full(n, False), w_max
    for i in range(n - 1, -1, -1):
        if (w[i] <= j) and (y[(i, j)] - y[(i - 1, j - w[i])] == v[i]):
            # the ith element is in the knapsack
            x[i] = True
            j -= w[i]
    return x

class AntColonyOptimization():
    """
    Ant colony optimization, which takes a directed or undirected graph `G`
    from `networkx` and a dictionary of edge tuples ot path lengths `lengths`.
    Ants start at the first node in the graph. Optional parameters include the
    number of ants per iteration `m`, the number of iterations `k_max`, the
    pheromone exponent `alpha`, the prior exponent `beta`, the evaporation
    scalar `rho`, and a dictionary of prior edge weights `eta`.
    """
    def __call__(self,
                 G: nx.Graph | nx.DiGraph,
                 lengths: dict[tuple[int, int], float],
                 m: int = 1000,
                 k_max: int = 100,
                 alpha: float = 1.0,
                 beta: float = 5.0,
                 rho: float = 0.5,
                 eta: dict[tuple[int, int], float] = None) -> list[int]:
        tau = {e: 1.0 for e in G.edges}
        x_best, y_best = [], np.inf
        for k in range(k_max):
            A = self.edge_attractiveness(G, tau, eta, alpha, beta)
            for (e, v) in tau.items():
                tau[e] = (1 - rho)*v
            for ant in range(m):
                x_best, y_best = self.run_ant(G, lengths, tau, A, x_best, y_best)
        return x_best

    def edge_attractiveness(self,
                            graph: nx.Graph | nx.DiGraph,
                            tau: dict[tuple[int, int], float],
                            eta: dict[tuple[int, int], float],
                            alpha: float = 1.0,
                            beta: float = 5.0) -> dict[tuple[int, int], float]:
        """
        A method for computing the edge attractiveness table given graph `graph`,
        pheromone levels `tau`, prior edge weights `eta`, pheromone exponent `alpha`,
        and prior exponent `beta`.
        """
        A = dict()
        for src in graph:
            neighbors = graph.neighbors(src)
            for dst in neighbors:
                v = (tau[(src, dst)]**alpha) * (eta[(src, dst)]**beta)
                A[(src, dst)] = v
        return A

    def run_ant(self,
                G: nx.Graph | nx.DiGraph,
                lengths: dict[tuple[int, int], float],
                tau: dict[tuple[int, int], float],
                A: dict[tuple[int, int], float],
                x_best: list[int],
                y_best: float) -> tuple[list[int], float]:
        """
        A method for simulating a single ant on a traveling salesman problem
        in which the ant starts at the first node and attempts to visit each node
        exactly once. Pheromone levels are increased at the end of a successful
        tour. The parameters are the graph `G`, edge lengths `lengths`, pheromone
        levels `tau`, edge attractiveness `A`, the best solution found thus far
        `x_best`, and its value `y_best`.
        """
        x = [1]
        while len(x) < len(G):
            src = x[-1]
            neighbors = np.setdiff1d(G.neighbors(src), x).tolist()
            if len(neighbors) == 0:  # ant got stuck
                return (x_best, y_best)
            
            attractiveness = [A[(src, dst)] for dst in neighbors]
            x.append(neighbors[np.random.choice(len(neighbors), p=normalize(attractiveness, 1))])
        
        l = np.sum([lengths[(x[i - 1], x[i])] for i in range(1, len(x))])
        for i in range(1, len(x)):
            tau[(x[i - 1], x[i])] += 1/l
        if l < y_best:
            return (x, l)
        return (x_best, y_best)
