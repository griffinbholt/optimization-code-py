import sys; sys.path.append("./src"); sys.path.append("../../")

import numpy as np

from ch21 import gauss_seidel


class TestGaussSeidel():
    def F1(A):
        A["y1"] = A["y2"] - A["x"]
        return A

    def F2(A):
        A["y2"] = np.sin(A["y1"] + A["y3"])
        return A
    
    def F3(A):
        A["y3"] = np.cos(A["x"] + A["y2"] + A["y1"])

    def test(self):
        A = {"x": 1.0, "y1": 1.0, "y2": 1.0, "y3": 1.0}
        A, converged = gauss_seidel([TestGaussSeidel.F1, TestGaussSeidel.F2, TestGaussSeidel.F3], A, k_max=100)
        assert converged
        assert np.isclose(A["y1"], -1.8795201143545137, atol=1e-8)
        assert np.isclose(A["y2"], -0.8795468970115342, atol=1e-8)
        assert np.isclose(A["y3"], -0.1871604183537351, atol=1e-8)
