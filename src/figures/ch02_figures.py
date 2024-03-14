import sys; sys.path.append('./src/'); sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from ch02 import diff_forward, diff_central, diff_complex


def figure_2_4():
    """
    Figure 2.4: A comparison of the error in derivative estimate for the
    function sin(x) at x = 1/2 as the step size is varied. The linear error
    of the forward difference method and the quadratic error of the central
    difference and complex methods can be seen by the constant slops on the
    right hand side. The complex step method avoids the subtractive cancellation
    error that occurs when differencing two function evaluations that are close
    together.
    """
    def abs_rel_error(v, v_approx): return np.abs((v - v_approx) / v)
    x = 0.5
    dfdx_true = np.cos(x)
   
    # Compute absolute relative errors for finite difference gradient approximations
    h = np.logspace(-18, 1, 100)
    error_complex = abs_rel_error(dfdx_true, diff_complex(np.sin, x, h))
    error_forward = abs_rel_error(dfdx_true, diff_forward(np.sin, x, h))
    error_central = abs_rel_error(dfdx_true, diff_central(np.sin, x, h))

    # Plot results
    plt.plot(h, error_complex, c="tab:green", label="complex")
    plt.plot(h, error_forward, c="tab:blue", label="forward")
    plt.plot(h, error_central, c="tab:red", label="central")
    plt.xlabel("step size h")
    plt.ylabel("absolute relative error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Figure 2.4")
    plt.show()
