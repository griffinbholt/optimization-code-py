import sys; sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np

from ch05 import GradientDescent, ConjugateGradientDescent
from ch06 import newtons_method


def exercise_6_3(x0: float):
    """Exercise 6.3: Applying Newton's Method to f(x) = x^2"""
    def f(x): return x**2
    def grad_f(x): return 2*x
    def H(x): return 2
    x = x0

    # Single Iteration of Newton's Method (for univariate function)
    Delta = grad_f(x) / H(x)
    x -= Delta

    print("After 1 iteration, x = ", x)
    print("Gradient at x: ", grad_f(x))
    print("=> Only 1 step of Newton's Method is needed to minimize f(x) = x^2.")


def exercise_6_4():
    """
    Exercise 6.4: Applying Newton's Method, Gradient Descent, and the 
    Conjugate Gradient Method to f(x) = (1/2)x'Hx.
    """
    def H(x): return np.array([[1.0, 0.0], [0.0, 1000.0]])
    def f(x): return 0.5 * np.dot(x, H(x) @ x)
    def grad_f(x): return H(x) @ x
    x0 = np.array([1.0, 1.0])

    # Newton's Method
    print("Newton's Method:")
    x_nm = newtons_method(grad_f, H, x0.copy(), eps=1e-5, k_max=1)
    print("After 1 iteration of Newton's Method, x = ", x_nm)
    print("Gradient at x: ", grad_f(x_nm))
    print("=> Newton's Method converges to the minimum after only 1 iteration.\n")

    # Gradient Descent
    print("Gradient Descent: (w/ unnormalized gradient)")
    M = GradientDescent(alpha=1)
    M.initialize(f, grad_f, x0.copy())
    x_gd = M.step(f, grad_f, x0.copy())
    print("After 1 iteration of Gradient Descent, x = ", x_gd)
    print("Gradient at x: ", grad_f(x_gd))
    x_gd = M.step(f, grad_f, x_gd)
    print("After 2 iterations of Gradient Descent, x = ", x_gd)
    print("Gradient at x: ", x_gd)
    print("=> Gradient Descent does not converge after 2 iterations.\n")

    # Conjugate Gradient Method
    print("Conjugate Gradient Method")
    M = ConjugateGradientDescent()
    M.initialize(f, grad_f, x0.copy())
    x_cg = M.step(f, grad_f, x0.copy())
    print("After 1 iteration of Conjugate Gradient, x = ", x_cg)
    print("Gradient at x: ", grad_f(x_cg))
    x_cg = M.step(f, grad_f, x_cg)
    print("After 2 iterations of Conjugate Gradient, x = ", x_cg)
    print("Gradient at x: ", x_cg)
    print("=> Gradient Descent converges after 2 iterations.\n")


def exercise_6_5():
    """Exercise 6.5: Comparison of Newton's Method vs. Secant Method"""
    def f(x): return x**2 + x**4
    def deriv(x): return 2*x + 4*(x**3)
    def deriv2(x): return 2 + 12*(x**2)
    n_iter = 10

    # Initialize Newton's Method
    x = -3
    newton_x = [x]
    newton_f = [f(x)]
    newton_deriv = [deriv(x)]

    # Initialize Secant Method
    x0, x1 = -4, -3
    g0 = deriv(x0)
    secant_x = [x1]
    secant_f = [f(x1)]
    secant_deriv = [deriv(x1)]
    
    for _ in range(n_iter):
        # Newton's Method (for univariate function)
        Delta_nm = deriv(x) / deriv2(x)
        x -= Delta_nm
        newton_x.append(x)
        newton_deriv.append(0)
        newton_f.append(f(x))
        newton_x.append(x)
        newton_deriv.append(deriv(x))

        # Secant Method
        g1 = deriv(x1)
        Delta_sm = ((x1 - x0) / (g1 - g0)) * g1
        x0, x1, g0 = x1, x1 - Delta_sm, g1
        secant_x.append(x1)
        secant_deriv.append(0)
        secant_f.append(f(x1))
        secant_x.append(x1)
        secant_deriv.append(deriv(x1))
    
    # Plots the results
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # f(x_k) vs iterations, k
    iters = np.arange(len(newton_f))
    ax[0].plot(iters, newton_f, color="tab:blue")
    ax[0].plot(iters, secant_f, color="tab:red")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("iterations, $k$")
    ax[0].set_ylabel("$f(x_k)$")

    # f_prime vs. x
    t = np.linspace(newton_x[0] - 0.5, newton_x[-1] + 0.5, 1000)
    f_prime = [deriv(t_i) for t_i in t]
    ax[1].plot(newton_x, newton_deriv, color="tab:blue", label="Newton")
    ax[1].plot(secant_x, secant_deriv, color="tab:red", label="secant")
    ax[1].hlines([0], xmin=newton_x[0] - 0.5, xmax=newton_x[-1] + 0.5, colors=['black'], linewidth=0.5)
    ax[1].plot(t, f_prime, color="black", linewidth=0.5)
    ax[1].set_xlabel("$x_k$")
    ax[1].set_ylabel("$f'(x_k)$")
    ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    fig.tight_layout()
    plt.show()


def exercise_6_9():
    """Exercise 6.9: Newton's Method for f(x) = (x1 + 1)^2 + (x2 + 3)^2 + 4"""
    def f(x): return (x[0] + 1)**2 + (x[1] + 3)**2 + 4
    def grad_f(x): return np.array([2*(x[0] + 1), 2*(x[1] + 3)])
    def H(x): return np.array([[2, 0], [0, 2]])

    x = np.zeros(2)
    x_prime = newtons_method(grad_f, H, x, eps=1e-5, k_max=1)
    print("After 1 step of Newton's Method, x = ", x_prime)
    print("Gradient at x: ", grad_f(x_prime))
    print("=> Newton's Method converges to the minimum after only 1 iteration.")
