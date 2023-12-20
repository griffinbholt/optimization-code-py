"""Chapter 5: First-Order Methods"""

import numpy as np

from abc import ABC, abstractmethod
from typing import Callable

from ch04 import line_search


class DescentMethod(ABC):
    @abstractmethod
    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        pass

    @abstractmethod
    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        pass


class GradientDescent(DescentMethod):
    """
    The gradient descent method, which follows the direction of gradient descent
    with a fixed learning rate. The `step` function produces the next iterate
    whereas the `initialize` function does nothing.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha  # learning rate

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        pass
    
    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        return x - self.alpha * g


class ConjugateGradientDescent(DescentMethod):
    """
    The conjugate gradient method with the Polak-Ribiere update, where `d`
    is the previous search direction and `g` is the previous gradient.
    """
    def __init__(self, d: np.ndarray, g: np.ndarray):
        self.d = d  # previous search direction
        self.g = g  # previous gradient

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.g = grad_f(x)
        self.d = -self.g
    
    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g_prime = grad_f(x)
        beta = np.maximum(0, np.dot(g_prime, g_prime - self.g) / np.dot(self.g, self.g))
        d_prime = -g_prime + beta*self.d
        x_prime = line_search(f, x, d_prime)
        self.d, self.g = d_prime, g_prime
        return x_prime
    

class Momentum(GradientDescent):
    """The momentum method for accelerated descent."""
    def __init__(self, alpha: float, beta: float, v: np.ndarray):
        super().__init__(alpha)  # learning rate
        self.beta = beta         # momentum decay
        self.v = v               # momentum
    
    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.v = np.zeros(len(x))

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        self.v = self.beta*self.v - self.alpha*g
        return x + self.v


class NesterovMomentum(Momentum):
    """Nesterov's momentum method of accelerated descent."""
    def __init__(self, alpha: float, beta: float, v: np.ndarray):
        super().__init__(alpha, beta, v)

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x + self.beta*self.v)
        self.v = self.beta*self.v - self.alpha*g
        return x + self.v


class Adagrad(GradientDescent):
    """The Adagrad accelerated descent method."""
    def __init__(self, alpha: float, eps: float, s: np.ndarray):
        super().__init__(alpha)  # learning rate
        self.eps = eps           # small value
        self.s = s               # sum of squared gradient
    
    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.s = np.zeros(len(x))
    
    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        self.s += g**2
        return x - self.alpha * (g / (np.sqrt(self.s) + self.eps))


class RMSProp(Adagrad):
    """The RMSProp accelerated descent method."""
    def __init__(self, alpha: float, gamma: float, eps: float, s: np.ndarray):
        super().__init__(alpha, eps, s)
        self.gamma = gamma  # decay
    
    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        self.s = self.gamma*self.s + (1 - self.gamma)*(g**2)
        return x - self.alpha * (g / (np.sqrt(self.s) + self.eps))


class Adadelta(DescentMethod):
    """
    The Adadelta accelerated descent method. The small constant `eps` is
    added to the numerator as well to prevent progress from entirely decaying to
    zero and to start off the first iteration where `delta_x = 0`.
    """
    def __init__(self, gamma_s: float, gamma_x: float, eps: float, s: np.ndarray, u: np.ndarray):
        self.gamma_s = gamma_s  # gradient decay
        self.gamma_x = gamma_x  # update decay
        self.eps = eps          # small value
        self.s = s              # sum of squared gradients
        self.u = u              # sum of squared updates

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.s = np.zeros(len(x))
        self.u = np.zeros(len(x))

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        self.s = self.gamma_s*self.s + (1 - self.gamma_s)*(g**2)
        delta_x = -((np.sqrt(self.u) + self.eps) / (np.sqrt(self.s) + self.eps)) * g
        self.u = self.gamma_x*self.u + (1 - self.gamma_x)*(delta_x**2)
        return x + delta_x


class Adam(GradientDescent):
    """The Adam accelerated descent method."""
    def __init__(self, alpha: float, gamma_v: float, gamma_s: float, eps: float, k: int, v: np.ndarray, s: np.ndarray):
        super().__init__(alpha)  # learning rate
        self.gamma_v = gamma_v   # 1st moment decay
        self.gamma_s = gamma_s   # 2nd moment decay
        self.eps = eps           # small value
        self.k = k               # step counter
        self.v = v               # 1st moment estimate
        self.s = s               # 2nd moment estimate
    
    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.k = 0
        self.v = np.zeros(len(x))
        self.s = np.zeros(len(x))
    
    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        self.v = self.gamma_v*self.v + (1 - self.gamma_v)*g
        self.s = self.gamma_s*self.s + (1 - self.gamma_s)*(g**2)
        self.k += 1
        v_hat = self.v / (1 - (self.gamma_v**self.k))
        s_hat = self.s / (1 - (self.gamma_s**self.k))
        return x - self.alpha * (v_hat / (np.sqrt(s_hat) + self.eps))


class HyperGradientDescent(GradientDescent):
    """The hypergradient form of gradient descent."""
    def __init__(self, alpha_0: float, mu: float, alpha: float, g_prev: np.ndarray):
        super().__init__(alpha)  # current learning rate
        self.alpha_0 = alpha_0   # initial learning rate
        self.mu = mu             # learning rate of the learning rate
        self.g_prev = g_prev     # previous gradient

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.alpha = self.alpha_0
        self.g_prev = np.zeros(len(x))

    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        self.alpha += self.mu * np.dot(g, self.g_prev)
        self.g_prev = g
        return x - self.alpha * g


class HyperNesterovMomentum(NesterovMomentum):
    """The hypergradient form of the Nesterov momentum descent method."""
    def __init__(self, alpha_0: float, mu: float, beta: float, v: np.ndarray, alpha: float, g_prev: np.ndarray):
        super().__init__(alpha, beta, v)  # current learning rate, momentum decay, momentum
        self.alpha_0 = alpha_0            # initial learning rate
        self.mu = mu                      # learning rate of the learning rate
        self.g_prev = g_prev              # previous gradient

    def initialize(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray):
        self.alpha = self.alpha_0
        self.v = np.zeros(len(x))
        self.g_prev = np.zeros(len(x))
    
    def step(self, f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        g = grad_f(x)
        self.alpha += self.mu * np.dot(g, self.g_prev + self.beta*self.v)
        self.v = g + self.beta*self.v
        self.g_prev = g
        return x - self.alpha * (g + self.beta*self.v)  # TODO - Ask Mykel if this is a typo
