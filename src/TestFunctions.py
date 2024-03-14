import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod

from convenience import plot_surface, plot_contour

# TODO - Think about having default values available through the class

class ScalarValuedTestFunction(ABC):
    def __init__(self, d: int):
        self.d = d  # dimension of input (None, if it can be arbitrary)

    @abstractmethod
    def __call__(self, x: np.ndarray, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def grad(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def hess(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def global_min(self, *args, **kwargs) -> tuple[float, np.ndarray]:
        pass

    @abstractmethod
    def plot(self):
        pass


class AckleysFunction(ScalarValuedTestFunction):
    def __init__(self):
        super().__init__(d=None) # arbitrary dimensional input

    def __call__(self, x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> float:
        """Ackley's function with d-dimensional input vector `x` and three optional parameters."""
        d = len(x)
        tmp = b / np.sqrt(d)
        norm = np.linalg.norm(x, axis=0)
        return -a*np.exp(-tmp * norm) - np.exp(np.sum(np.cos(c*x), axis=0)/d) + a + np.e

    def grad(self, x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> np.ndarray:
        d = len(x)
        tmp = b / np.sqrt(d)
        norm = np.linalg.norm(x)
        part_1 = x * (((a*tmp) * np.exp(-tmp * norm)) / norm)
        part_2 = np.sin(c*x) * (c/d) * np.exp(np.sum(np.cos(c*x))/d)
        return part_1 + part_2

    def hess(self, x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> np.ndarray:
        d = len(x)
        tmp = b / np.sqrt(d)
        tmp2 = c / d
        norm = np.linalg.norm(x)
        norm_sq = np.dot(x, x)
        norm_cb = norm**3
        inner1 = -tmp * norm
        exp_inner1 = np.exp(inner1)
        sin_cx = np.sin(c * x)
        cos_cx = np.cos(c * x)
        inner2 = np.sum(cos_cx) / d
        exp_inner2 = np.exp(inner2)

        hess = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                prod = x[i] * x[j]
                if i == j:
                    hess[i, i] += a * tmp * exp_inner1 * ((-tmp * (prod / norm_sq)) + ((norm_sq - prod) / norm_cb))
                    hess[i, i] += tmp2 * exp_inner2 * (-tmp2 * ((sin_cx[i])**2) + c * cos_cx[i])
                else:
                    hess[i, j] += a * tmp * exp_inner1 * ((-tmp * (prod / norm_sq)) - (prod / norm_cb))
                    hess[i, j] += - (tmp2**2) * sin_cx[i] * sin_cx[j] * exp_inner2
        
        return hess

    def global_min(self, d: int = 2, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros(d)

    def plot(self):
        f_min, x_min = self.global_min(d=2)
        lim = (-30.1, 30.1)
        zlim = (-1, 30)
        stride = 0.01
        levels = [i for i in range(25)]
        fig = plt.figure(figsize=(15, 5))
        surf_ax = plot_surface(fig, self, lim, lim, zlim, stride, stride, subplot_coords=(1,3,1))
        surf_ax.scatter([x_min[0]], [x_min[1]], [f_min], c='black', s=0.5)
        surf_ax.view_init(azim=-60)
        contf_ax = plot_contour(fig, self, lim, lim, stride, stride, levels=levels, filled=True, subplot_coords=(1,3,2))
        contf_ax.scatter([x_min[0]], [x_min[1]], c='black', s=0.5)
        cont_ax = plot_contour(fig, self, lim, lim, stride, stride, levels=levels, subplot_coords=(1,3,3))
        cont_ax.scatter([x_min[0]], [x_min[1]], c='black', s=0.5)
        plt.suptitle("Ackley's Function")
        plt.subplots_adjust(wspace=0.5)
        plt.show()


class BoothsFunction(ScalarValuedTestFunction):
    H = np.array([[10.0, 8.0], [8.0, 10.0]])
    F_MIN = 0.0
    X_MIN = np.array([1.0, 3.0])

    def __init__(self):
        super().__init__(d=2)  # two-dimensional input

    def __call__(self, x: np.ndarray) -> float:
        """Booth's function with two-dimensional input vector `x`."""
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def grad(self, x: np.ndarray) -> np.ndarray:
        dx1 = 10*x[0] + 8*x[1] - 34
        dx2 = 8*x[0] + 10*x[1] - 38
        return np.array([dx1, dx2])

    def hess(self, x: np.ndarray) -> np.ndarray:
        return self.H

    def global_min(self) -> tuple[float, np.ndarray]:
        return self.F_MIN, self.X_MIN

    def plot(self):
        f_min, x_min = self.global_min()
        xlim = (-10.1, 10.1)
        ylim = (-10.1, 10.1)
        zlim = (-1, 1000)
        stride = 0.01
        levels = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000]
        fig = plt.figure(figsize=(15, 5))
        surf_ax = plot_surface(fig, self, xlim, ylim, zlim, stride, stride, subplot_coords=(1,3,1))
        surf_ax.scatter([x_min[0]], [x_min[1]], [f_min], c='black')
        surf_ax.view_init(azim=-60)
        contf_ax = plot_contour(fig, self, xlim, ylim, stride, stride, levels=levels, filled=True, subplot_coords=(1,3,2))
        contf_ax.scatter([x_min[0]], [x_min[1]], c='black')
        cont_ax = plot_contour(fig, self, xlim, ylim, stride, stride, levels=levels, subplot_coords=(1,3,3))
        cont_ax.scatter([x_min[0]], [x_min[1]], c='black')
        plt.suptitle("Booth's Function")
        plt.subplots_adjust(wspace=0.5)
        plt.show()


class BraninFunction(ScalarValuedTestFunction):
    X1_MIN = np.pi * np.array([-1.0, 1.0, 3.0, 5.0])  # x1 = pi + 2*pi*m for integral m

    def __init__(self):
        super().__init__(d=2)  # two-dimensional input

    def __call__(self, 
                 x: np.ndarray,
                 a: float = 1,
                 b: float = 5.1/(4*(np.pi**2)),
                 c: float = 5/np.pi,
                 r: float = 6,
                 s: float = 10,
                 t: float = 1/(8*np.pi)) -> float:
        """The Branin function with two-dimensional input vector `x` and six optional parameters."""
        return a * ((x[1] - b*(x[0]**2) + c*x[0] - r)**2) + s * (1 - t) * np.cos(x[0]) + s

    def grad(self, 
             x: np.ndarray,
             a: float = 1,
             b: float = 5.1/(4*(np.pi**2)),
             c: float = 5/np.pi,
             r: float = 6,
             s: float = 10,
             t: float = 1/(8*np.pi)) -> np.ndarray:
        dx2 = 2 * a * (x[1] - b*(x[0]**2) + c*x[0] - r)
        dx1 = dx2 * (c - 2*b*x[0]) - s * (1 - t) * np.sin(x[0])
        return np.array([dx1, dx2])

    def hess(self, 
             x: np.ndarray,
             a: float = 1,
             b: float = 5.1/(4*(np.pi**2)),
             c: float = 5/np.pi,
             r: float = 6,
             s: float = 10,
             t: float = 1/(8*np.pi)) -> np.ndarray:
        dx2dx2 = 2 * a
        dx1dx1 = dx2dx2 * (6*((b*x[0])**2) - 6*c*b*x[0] + c**2 - 2*b*x[1] + 2*b*r) - s * (1 - t) * np.cos(x[0])
        dx1dx2 = dx2dx2 * (c - 2*b*x[0])
        return np.array([[dx1dx1, dx1dx2], [dx1dx2, dx2dx2]])

    def global_min(self,
                   a: float = 1,
                   b: float = 5.1/(4*(np.pi**2)),
                   c: float = 5/np.pi,
                   r: float = 6,
                   s: float = 10,
                   t: float = 1/(8*np.pi)) -> tuple[float, np.ndarray]:
        x2 = b*(self.X1_MIN**2) - c*self.X1_MIN + r
        x = np.array([self.X1_MIN, x2]).T
        f = np.array([self(x_opt, a, b, c, r, s, t) for x_opt in x])
        return f, x.T

    def plot(self):
        f_min, x_min = self.global_min()
        xlim = (2*np.pi - 12, 2*np.pi + 12)
        ylim = (-3, 22)
        zlim = (-1, 200)
        stride = 0.01
        levels = [0, 1, 2, 3, 5, 10, 20, 50, 100]
        fig = plt.figure(figsize=(15, 5))
        surf_ax = plot_surface(fig, self, xlim, ylim, zlim, stride, stride, subplot_coords=(1,3,1))
        surf_ax.scatter(x_min[0], x_min[1], f_min, c='black')
        surf_ax.view_init(elev=25, azim=-100)
        contf_ax = plot_contour(fig, self, xlim, ylim, stride, stride, levels=levels, filled=True, subplot_coords=(1,3,2))
        contf_ax.scatter(x_min[0], x_min[1], c='black')
        cont_ax = plot_contour(fig, self, xlim, ylim, stride, stride, levels=levels, subplot_coords=(1,3,3))
        cont_ax.scatter(x_min[0], x_min[1], c='black')
        plt.suptitle("Branin Function")
        plt.subplots_adjust(wspace=0.5)
        plt.show()


class FlowerFunction(ScalarValuedTestFunction):
    def __init__(self):
        super().__init__(d=2)  # two-dimensional input

    def __call__(self, x: np.ndarray, a: float = 1, b: float = 1, c: float = 4) -> float:
        """The flower function with two-dimensional input vector `x` and three optional parameters."""
        return a * np.linalg.norm(x, axis=0) + b * np.sin(c * np.arctan2(x[1], x[0]))

    def grad(self, x: np.ndarray, a: float = 1, b: float = 1, c: float = 4) -> np.ndarray:
        norm = np.linalg.norm(x)
        tmp = (a * x) / norm
        tmp2 = b * c * np.cos(c * np.arctan2(x[1], x[0])) / (norm**2)
        return tmp + tmp2 * np.array([-x[1], x[0]])

    def hess(self, x: np.ndarray, a: float = 1, b: float = 1, c: float = 4) -> np.ndarray:
        inner = c * np.arctan2(x[1], x[0])
        sin_inner = np.sin(inner)
        cos_inner = np.cos(inner)
        norm = np.linalg.norm(x)
        norm_2 = norm**2
        norm_3 = norm**3
        norm_4 = norm**4
        tmp = b * c / norm_4
        prod = x[0] * x[1]
        x_sq = x**2

        dx1dx1 = a * ((norm_2 - (x_sq[0])) / norm_3) + tmp * (-x_sq[1] * c * sin_inner + 2*prod*cos_inner)
        dx1dx2 = a * (-prod / norm_3) + tmp * (prod * c * sin_inner + (norm_2 - 2*x_sq[0]) * cos_inner)
        dx2dx2 = a * ((norm_2 - (x_sq[1])) / norm_3) + tmp * (-x_sq[0] * c * sin_inner - 2*prod*cos_inner)

        return np.array([[dx1dx1, dx1dx2], [dx1dx2, dx2dx2]])

    def global_min(self, a: float = 1, b: float = 1, c: float = 4):
        return None, None  # NOTE: The Flower function has no global minimum

    def plot(self):
        lim = (-3.1, 3.1)
        zlim = (0, 6)
        stride = 0.01
        levels = np.arange(-0.1, 5, 0.5)
        fig = plt.figure(figsize=(15, 5))
        surf_ax = plot_surface(fig, self, lim, lim, zlim, stride, stride, subplot_coords=(1,3,1))
        surf_ax.view_init(elev=45, azim=-60)
        plot_contour(fig, self, lim, lim, stride, stride, levels=levels, filled=True, subplot_coords=(1,3,2))
        plot_contour(fig, self, lim, lim, stride, stride, levels=levels, subplot_coords=(1,3,3))
        plt.suptitle("Flower Function")
        plt.subplots_adjust(wspace=0.5)
        plt.show()


class MichalewiczFunction(ScalarValuedTestFunction):
    X_MIN_M10 = np.array([2.2029, np.pi/2])

    def __init__(self):
        super().__init__(d=None)  # arbitrary dimensional input

    """The Michalewicz function with input vector `x` and optional steepness parameter `m`."""
    def __call__(self, x: np.ndarray, m: float = 10) -> float:
        i = np.arange(len(x)) + 1
        if len(x.shape) == 3:
            inner = (i[:,None,None]*(x**2)) / np.pi
        else:
            inner = (i*(x**2)) / np.pi
        sin_inner = np.sin(inner)
        exp = 2*m
        return -np.sum(np.sin(x)*(sin_inner**exp), axis=0)

    def grad(self, x: np.ndarray, m: float = 10) -> np.ndarray:
        i = np.arange(len(x)) + 1
        inner = (i*(x**2)) / np.pi
        sin_inner = np.sin(inner)
        cos_inner = np.cos(inner) 
        return - (sin_inner**(2*m - 1)) * (np.cos(x) * sin_inner + ((4*m*i*x)/ np.pi) * np.sin(x) * cos_inner)

    def hess(self, x: np.ndarray, m: float = 10) -> np.ndarray:
        i = np.arange(len(x)) + 1
        inner = (i*(x**2)) / np.pi
        sin_x = np.sin(x)
        cos_x = np.cos(x)
        sin_inner = np.sin(inner)
        cos_inner = np.cos(inner)
        const = (2 * i * x) / np.pi
        const2 = 2 * m * const

        tmp1 = -(2*m - 1) * (sin_inner**(2*m - 2)) * cos_inner * const
        tmp2 = cos_x * sin_inner + const2 * sin_x * cos_inner
        first = tmp1 * tmp2

        tmp3 = cos_x * cos_inner * const - sin_x * sin_inner
        tmp4 = cos_x * cos_inner - sin_x * sin_inner * const
        second = - (sin_inner**(2*m - 1)) * (tmp3 + const2 * tmp4)

        hess = first + second
        return np.diag(hess)

    def global_min(self) -> tuple[float, np.ndarray]:
        x = self.X_MIN_M10
        f = self(x)
        return f, x

    def plot(self):
        f_min, x_min = self.global_min()
        lim = (-0.1, 4.1)
        zlim = (-2, 1.2)
        stride = 0.01
        levels = np.arange(-2, 1.2, 0.1)
        fig = plt.figure(figsize=(15, 5))
        surf_ax = plot_surface(fig, self, lim, lim, zlim, stride, stride, subplot_coords=(1,3,1))
        surf_ax.scatter([x_min[0]], [x_min[1]], [f_min], c='black', s=0.5)
        surf_ax.view_init(azim=-70)
        contf_ax = plot_contour(fig, self, lim, lim, stride, stride, levels=levels, filled=True, subplot_coords=(1,3,2))
        contf_ax.scatter([x_min[0]], [x_min[1]], c='black', s=0.5)
        cont_ax = plot_contour(fig, self, lim, lim, stride, stride, levels=levels, subplot_coords=(1,3,3))
        cont_ax.scatter([x_min[0]], [x_min[1]], c='black', s=0.5)
        plt.suptitle("Michalewicz Function")
        plt.subplots_adjust(wspace=0.5)
        plt.show()


class RosenbrockFunction(ScalarValuedTestFunction):
    def __init__(self):
        super().__init__(d=2)  # two-dimensional input

    def __call__(self, x: np.ndarray, a: float = 1, b: float = 5) -> float:
        """The Rosenbrock function with two-dimensional input vector `x` and
        two optinal parameters."""
        return (a - x[0])**2 + b*(x[1] - (x[0]**2))**2

    def grad(self, x: np.ndarray, a: float = 1, b: float = 5) -> np.ndarray:
        dx1 = 2 * ((2*b*(x[0]**3)) - (2*b*x[0]*x[1]) + x[0] - a)
        dx2 = 2 * b * (x[1] - (x[0]**2))
        return np.array([dx1, dx2])

    def hess(self, x: np.ndarray, a: float = 1, b: float = 5) -> np.ndarray:
        dx1dx1 = 12*b*(x[0]**2) - 4*b*x[1] + 2
        dx1dx2 = -4*b*x[0]
        dx2dx2 = 2*b
        return np.array([[dx1dx1, dx1dx2], [dx1dx2, dx2dx2]])

    def global_min(self, a: float = 1, b: float = 5) -> tuple[float, np.ndarray]:
        x = np.array([a, a**2])
        f = self(x, a, b)
        return f, x

    def plot(self):
        f_min, x_min = self.global_min()
        lim = (-2.1, 2.1)
        zlim = (-5, 105)
        stride = 0.01
        levels = [0, 1, 2, 3, 5, 9, 25, 50, 100]
        fig = plt.figure(figsize=(15, 5))
        surf_ax = plot_surface(fig, self, lim, lim, zlim, stride, stride, subplot_coords=(1,3,1))
        surf_ax.scatter([x_min[0]], [x_min[1]], [f_min], c='black')
        surf_ax.view_init(azim=-100)
        contf_ax = plot_contour(fig, self, lim, lim, stride, stride, levels=levels, filled=True, subplot_coords=(1,3,2))
        contf_ax.scatter([x_min[0]], [x_min[1]], c='black')
        cont_ax = plot_contour(fig, self, lim, lim, stride, stride, levels=levels, subplot_coords=(1,3,3))
        cont_ax.scatter([x_min[0]], [x_min[1]], c='black')
        plt.suptitle("Rosenbrock's Banana Function")
        plt.subplots_adjust(wspace=0.5)
        plt.show()


class WheelersRidge(ScalarValuedTestFunction):
    def __init__(self):
        super().__init__(d=2)  # two-dimensional input

    def __call__(self, x: np.ndarray, a: float = 1.5) -> float:
        """Wheeler's ridge, which takes in a two-dimensional design point `x` and an optional scalar parameter `a`."""
        return -np.exp(-(x[0]*x[1] - a)**2 - (x[1] - a)**2)

    def grad(self, x: np.ndarray, a: float = 1.5) -> np.ndarray:
        f = self(x, a)
        tmp = -2 * (x[0]*x[1] - a)
        dx1 = f * tmp * x[1]
        dx2 = f * (tmp * x[0] - 2 * (x[1] - a))
        return np.array([dx1, dx2])

    def hess(self, x: np.ndarray, a: float = 1.5) -> np.ndarray:
        f = self(x, a)
        tmp = -2 * (x[0]*x[1] - a)
        tmp_1 = tmp * x[1]
        tmp_2 = tmp * x[0] - 2 * (x[1] - a)
        dx1, dx2 = self.grad(x, a)
        dx1dx1 = dx1 * tmp_1 + f * (-2 * (x[1]**2))
        dx2dx1 = dx2 * tmp_1 + f * (-2 * (2*x[0]*x[1] - a))
        dx2dx2 = dx2 * tmp_2 + f * (-2 * (x[0]**2)  - 2)
        return np.array([[dx1dx1, dx2dx1], [dx2dx1, dx2dx2]])

    def global_min(self, a: float = 1.5) -> tuple[float, np.ndarray]:
        x = np.array([1, a])
        f = self(x, a)
        return f, x

    def plot(self):
        f_min, x_min = self.global_min()
        xlim = (-9, 26)
        ylim = (-3, 7)
        zlim = (-1.1, 0.5)
        stride = 0.01
        fig = plt.figure(figsize=(15, 5))
        surf_ax = plot_surface(fig, self, xlim, ylim, zlim, stride, stride,
            subplot_coords=(1,3,1)  
        )
        surf_ax.scatter([x_min[0]], [x_min[1]], [f_min], c='black', s=0.5)
        contf_ax = plot_contour(fig, self, xlim, ylim, stride, stride,
            levels=np.arange(-1, 0.1, 0.01),
            filled=True,
            subplot_coords=(1,3,2)
        )
        contf_ax.scatter([x_min[0]], [x_min[1]], c='black', s=0.5)
        cont_ax = plot_contour(fig, self,
            xlim=(-0.1, 3),
            ylim=(-0.1, 3),
            xstride=stride,
            ystride=stride,
            levels=np.arange(-1, -0.0, 0.1),
            subplot_coords=(1,3,3)
        )
        cont_ax.scatter([x_min[0]], [x_min[1]], c='black', s=0.5)
        plt.suptitle("Wheeler's Ridge")
        plt.subplots_adjust(wspace=0.5)
        plt.show()


class VectorValuedTestFunction(ABC):
    def __init__(self, in_d: int, out_d: int):
        self.in_d = in_d    # dimension of input (None, if it can be arbitrary)
        self.out_d = out_d  # dimension of output (None, if it can be arbitrary)

    @abstractmethod
    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def jac(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass


class CircleFunction(VectorValuedTestFunction):
    def __init__(self):
        super().__init__(in_d=2, out_d=2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The circle function, which takes in a two-dimensional design point `x`
        and produces a two-dimensional objective value."""
        theta = x[0]
        r = 0.5 + (x[1]/(1 + (x[1]**2)))
        y1 = 1 - r * np.cos(theta)
        y2 = 1 - r * np.sin(theta)
        return np.array([y1, y2])

    def jac(self, x: np.ndarray) -> np.ndarray:
        theta = x[0]
        y2 = x[1]**2
        denom = 1 + y2
        denom_2 = denom**2
        y2_sub_1 = y2 - 1
        r = 0.5 + (x[1]/denom)
        dy1dx1 = np.sin(theta) * r
        dy1dx2 = (np.cos(theta) * y2_sub_1) / denom_2
        dy2dx1 = -np.cos(theta) * r
        dy2dx2 = (np.sin(theta) * y2_sub_1) / denom_2
        return np.array([[dy1dx1, dy1dx2], [dy2dx1, dy2dx2]])


ackley = AckleysFunction()
booth = BoothsFunction()
branin = BraninFunction()
flower = FlowerFunction()
michalewicz = MichalewiczFunction()
rosenbrock = RosenbrockFunction()
wheeler = WheelersRidge()
circle = CircleFunction()
