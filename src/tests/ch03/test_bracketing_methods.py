import sys; sys.path.append('./src/'); sys.path.append('../../')

import pytest

from ch03 import *


class TestBracketingMethods():
    @pytest.fixture(autouse=True)
    def run_before(self):
        self.f = lambda x: 2*(x**4) + 5*(x**3) - 3*x
        self.f_prime = lambda x: 8*(x**3) + 15*(x**2) - 3
        self.x_local_min = 0.40550
        self.x_global_min = -1.75296
    
    def test_bracket_minimum(self):
        a, b = bracket_minimum(self.f, x=0.0)
        assert (a <= self.x_local_min) and (self.x_local_min <= b)

        a, b = bracket_minimum(self.f, x=-1.0)
        assert (a <= self.x_global_min) and (self.x_global_min <= b)

    def test_fibonacci_search(self):
        a, b = fibonacci_search(self.f, a=-5, b=5, n=10)
        assert (a <= self.x_global_min) and (self.x_global_min <= b)

        a, b = fibonacci_search(self.f, a=-5, b=0, n=10)
        assert (a <= self.x_global_min) and (self.x_global_min <= b)

        a, b = fibonacci_search(self.f, a=0, b=5, n=10)
        assert (a <= self.x_local_min) and (self.x_local_min <= b)

    def test_golden_section_search(self, eps=1e-5):
        a, b = golden_section_search(self.f, a=-5, b=5, n=10)
        assert (a <= self.x_global_min) and (self.x_global_min <= b)

        a, b = -5, 5
        n = np.ceil((b - a)/(eps*np.log(PHI))).astype(int)
        a, b = golden_section_search(self.f, a, b, n)
        assert np.abs(self.x_global_min - a) < eps
        assert np.abs(self.x_global_min - b) < eps

        a, b = golden_section_search(self.f, a=-5, b=0, n=10)
        assert (a <= self.x_global_min) and (self.x_global_min <= b)

        a, b = -5, 0
        n = np.ceil((b - a)/(eps*np.log(PHI))).astype(int)
        a, b = golden_section_search(self.f, a, b, n)
        assert np.abs(self.x_global_min - a) < eps
        assert np.abs(self.x_global_min - b) < eps

        a, b = golden_section_search(self.f, a=0, b=5, n=10)
        assert (a <= self.x_local_min) and (self.x_local_min <= b)

        a, b = 0, 5
        n = np.ceil((b - a)/(eps*np.log(PHI))).astype(int)
        a, b = golden_section_search(self.f, a, b, n)
        assert np.abs(self.x_local_min - a) < eps
        assert np.abs(self.x_local_min - b) < eps

    def test_quadratic_fit_search(self, eps=1e-5):
        a, b, c = quadratic_fit_search(self.f, a=-5, b=0, c=5, n=1000)
        assert (a - eps <= self.x_global_min) and (self.x_global_min <= c + eps)
        assert np.abs(self.x_global_min - b) <= eps

        a, b, c = quadratic_fit_search(self.f, a=-10, b=-5, c=0, n=1000)
        assert (a - eps <= self.x_global_min) and (self.x_global_min <= c + eps)
        assert np.abs(self.x_global_min - b) <= eps

        a, b, c = quadratic_fit_search(self.f, a=0.1, b=5, c=10, n=1000)
        assert (a - eps <= self.x_local_min) and (self.x_local_min <= c + eps)
        assert np.abs(self.x_local_min - b) <= eps

    def test_shubert_piyavskii(self, eps=1e-5):
        P_min, intervals = shubert_piyavskii(self.f, a=-2, b=1, l=20, eps=eps)
        assert np.abs(P_min[0] - self.x_global_min) <= eps
        assert len(intervals) > 0
        assert self.in_an_interval(self.x_global_min, intervals)

        P_min, intervals = shubert_piyavskii(self.f, a=0, b=1, l=20, eps=eps)
        assert np.abs(P_min[0] - self.x_local_min) <= eps
        assert len(intervals) > 0
        assert self.in_an_interval(self.x_local_min, intervals)

        P_min, intervals = shubert_piyavskii(lambda x: np.sin(x) - 0.5*x, a=-5, b=7, l=1.5, eps=eps)
        assert np.abs(P_min[0] - (5*np.pi/3)) <= eps
        assert len(intervals) > 0
        assert self.in_an_interval(5*np.pi/3, intervals)

    def in_an_interval(self, x_min: float, intervals: list[tuple[float, float]]) -> bool:
        in_interval = False
        for interval in intervals:
            in_interval = in_interval or (interval[0] <= x_min and x_min <= interval[1])
        return in_interval

    def test_bisection(self, eps=1e-5):
        a, b = bisection(self.f_prime, a=-5, b=5, eps=eps/10)
        assert (np.abs(self.x_global_min - a) <= eps) or (np.abs(self.x_local_min - a) <= eps)
        assert (np.abs(self.x_global_min - b) <= eps) or (np.abs(self.x_local_min - b) <= eps)
    
        a, b = bisection(self.f_prime, a=-5, b=-0.5, eps=eps/10)
        assert (a - eps <= self.x_global_min) and (self.x_global_min <= b + eps)
        assert np.abs(self.x_global_min - a) <= eps
        assert np.abs(self.x_global_min - b) <= eps

        a, b = bisection(self.f_prime, a=0, b=5, eps=eps/10)
        assert (a - eps <= self.x_local_min) and (self.x_local_min <= b + eps)
        assert np.abs(self.x_local_min - a) <= eps
        assert np.abs(self.x_local_min - b) <= eps

    def test_bracket_sign_change(self):
        a, b = bracket_sign_change(self.f_prime, a=-5, b=0)
        assert self.f_prime(a) * self.f_prime(b) <= 0

        a, b = bracket_sign_change(self.f_prime, a=-5, b=5)
        assert self.f_prime(a) * self.f_prime(b) <= 0

        a, b = bracket_sign_change(self.f_prime, a=0, b=5)
        assert self.f_prime(a) * self.f_prime(b) <= 0
