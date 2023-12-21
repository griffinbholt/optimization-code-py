"""Chapter 11: Linear Constrainted Optimization"""

import numpy as np

class LinearProgram():
    """
    A linear program in equality form:

    minimize    c'x
    subject to: Ax = b
                x >= 0
    """
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray):
        self.A = A
        self.b = b
        self.c = c

    def get_vertex(self, B: np.ndarray) -> np.ndarray:
        """A method for extracting the vertex associated with a partition `B` and an LP `self`"""
        b_inds = np.sort(B)
        AB = self.A[:, b_inds]
        xB = np.linalg.solve(AB, self.b)
        x = np.zeros(len(self.c))
        x[b_inds] = xB
        return x

    def edge_transition(self, B: np.ndarray, q: int) -> tuple[int, float]:
        """
        A method for computing the index `p` and the new coordinate value `x_q_prime`
        obtained by increasing index `q` of the vertex defined by the partition
        `B` in the equality-form linear program.
        """
        A, b = self.A, self.b
        n = A.shape[1]
        b_inds = np.sort(B)
        n_inds = np.setdiff1d(np.arange(n), B)
        AB = A[:, b_inds]
        d, xB = np.linalg.solve(AB, A[:, n_inds[q]]), np.linalg.solve(AB, b)

        p, xq_prime = 0, np.inf
        for i in range(len(d)):
            if d[i] > 0:
                v = xB[i] / d[i]
                if v < xq_prime:
                    p, xq_prime = i, v

        return (p, xq_prime)

    def step(self, B: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        A single iteration of the simplex algorithm in which the set `B`
        is moved from one vertex to a neighbor while maximally decreasing the
        objective function. The function takes a partition defined by `B`.
        """
        A, b, c = self.A, self.b, self.c
        n = A.shape[1]
        b_inds = np.sort(B)
        n_inds = np.setdiff1d(np.arange(n), B)
        AB, AV = A[:, b_inds], A[:, n_inds]
        # xB = np.linalg.solve(AB, b) # TODO - never used?
        cB = c[b_inds]
        lam = np.linalg.solve(AB.T, cB)
        cV = c[n_inds]
        muV = cV - AV.T @ lam

        q, p, xq_prime, delta = 0.0, 0.0, np.inf, np.inf
        for i in range(len(muV)):
            if muV[i] < 0:
                pi, xi_prime = self.edge_transition(B, i)
                if muV[i] * xi_prime < delta:
                    q, p, xq_prime, delta = i, pi, xi_prime, muV[i]*xi_prime
            if q == 0:
                return (B, True)  # optimal point found

        if np.isinf(xq_prime):
            raise ValueError("unbounded")

        j = np.where(B == b_inds[p])[0][0]
        B[j] = n_inds[q]   # swap indices
        return (B, False)  # new vertex but not optimal

    def minimize_given_vertex_partition(self, B: np.ndarray) -> np.ndarray:
        """Minimizing a linear program given a vertex partition defined by `B`."""
        done = False
        while not done:
            B, done = self.step(B)
        return B

    def minimize(self, return_idcs=False) -> np.ndarray:
        """
        The simplex algorithm for solving linear programs in equality form
        when an initial partition is not known.
        """
        A, b, c = self.A, self.b, self.c  # TODO - c is not necessary?
        m, n = A.shape
        z = np.ones(m)
        Z = np.diag([1 if j >= 0 else -1 for j in b])

        A_prime = np.hstack([A, Z])
        b_prime = b
        c_prime = np.concatenate((np.zeros(n), z))
        LP_init = LinearProgram(A_prime, b_prime, c_prime)
        B = np.arange(1, m + 1) + n
        B = LP_init.minimize_given_vertex_partition(B)

        if np.any(B > n):
            raise ValueError("infeasible")

        A_prime_prime = np.vstack([np.hstack([A, np.eye(m)]),
                                   np.hstack([np.zeros((m, n)), np.eye(m)])])
        b_prime_prime = np.concatenate((b, np.zeros(m)))
        c_prime_prime = c_prime
        LP_opt = LinearProgram(A_prime_prime, b_prime_prime, c_prime_prime)
        B = LP_opt.minimize_given_vertex_partition(B)
        x_opt = LP_opt.get_vertex(B)[:n]
        if return_idcs:
            b_inds = np.sort(B)
            n_inds = np.setdiff1d(np.arange(n), B)
            return x_opt, b_inds, n_inds
        return x_opt

    def dual_certificate(self, x: np.ndarray, lam: np.ndarray, eps: float = 1e-6) -> bool:
        """
        A method for checking whether a candidate solution given by design point
        `x` and dual point `lam` for the linear program is optimal. The
        parameter `eps` controls the tolerance for the equality constraint.
        """
        A, b, c = self.A, self.b, self.c
        primal_feasible = np.all(x >= 0) and np.all(np.isclose(A @ x, b))
        dual_feasible = np.all(A.T @ lam <= c)
        return primal_feasible and dual_feasible and np.isclose(np.dot(c, x), np.dot(b, lam), atol=eps)

    def minimize_lp_and_y(self) -> tuple[np.ndarray, float]:
        """
        (From Chapter 19) Solves an LP and returns both the solutions and its
        value. An infeasible LP produces a `NaN` solution and an `np.inf` value. 
        """
        try:
            x = self.minimize()
            return (x, np.dot(x, self.c))
        except ValueError:
            return (np.full(len(self.c), np.nan), np.inf)