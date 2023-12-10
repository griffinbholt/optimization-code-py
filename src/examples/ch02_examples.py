import numpy as np
import sympy as sp

from typing import Union


def example_2_1():
    """Example 2.1: Symbolic differentiation provides analytical derivatives."""
    x = sp.Symbol('x')
    f = x**2 + x/2 - sp.sin(x)/x
    print(sp.diff(f, x))

def example_2_4():
    """Example 2.4: The complex step method for estimating derivatives"""
    def f(x): return np.sin(x**2)
    v = f(np.pi/2 + 0.001j)
    print("f(x) = real(v) = ", np.real(v))
    print("f'(x) = imag(v)/0.001 = ", np.imag(v)/0.001)

def example_2_5():
    """Example 2.5: An implementation of dual numbers allows for automatic forward accumulation"""
    class Dual():
        def __init__(self, v: float, d: float):
            self.v = v
            self.d = d

        def __repr__(self) -> str:
            return 'Dual(' + str(self.v) + ',' + str(self.d) + ')'
        
        def __add__(self, other: 'Dual') -> 'Dual':
            return Dual(self.v + other.v, self.d + other.d)

        def __mul__(self, other: 'Dual') -> 'Dual':
            return Dual(self.v * other.v, self.v * other.d + other.v * self.d)
    
        @staticmethod
        def log(a: 'Dual') -> 'Dual':
            return Dual(np.log(a.v), a.d / a.v)

        @staticmethod
        def max(a: 'Dual', b: Union['Dual', int]) -> 'Dual':
            if isinstance(b, Dual):
                v = np.maximum(a.v, b.v)
                d = a.d if a.v > b.v else (b.d if a.v < b.v else np.nan)
            else:  # isinstance(b, int)
                v = np.maximum(a.v, b)
                d = a.d if a.v > b else (0 if a.v < b else np.nan)
            return Dual(v, d)

    a = Dual(3, 1)
    b = Dual(2, 0)
    print(Dual.log(a*b + Dual.max(a, 2)))

def example_2_6():
    """
    Example 2.6: Automatic differentiation using the Tensorflow package.
    We find that the gradient at [3, 2] is [1/3, 1/3]
    """
    import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Disables Tensorflow CPU warning
    import tensorflow as tf

    @tf.function
    def f(a, b): return tf.math.log(a*b + tf.math.maximum(a, 2))

    x = tf.Variable(3.0)
    y = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        z = f(x, y)
    print([deriv.numpy() for deriv in tape.gradient(z, [x, y])])
