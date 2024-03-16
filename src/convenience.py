import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm, ticker
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


VIRIDIS_REV = cm.viridis.reversed()


def normalize(x: np.ndarray,
              ord: int | float | str = 2,
              axis: int | tuple[int, int] = None,
              keepdims: bool = False) -> np.ndarray:
    nmlzd_x = np.divide(x, np.linalg.norm(x, ord, axis, keepdims))
    nmlzd_x = np.where(np.abs(nmlzd_x) < 1e-16, 0, nmlzd_x)
    return nmlzd_x


def plot_surface(fig, f, xlim, ylim, zlim, xstride, ystride, subplot_coords=None):
    X, Y, Z = _make_3d_data(f, xlim, ylim, xstride, ystride)
    if subplot_coords is not None:
        ax = fig.add_subplot(*subplot_coords, projection='3d')
    else:
        ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=VIRIDIS_REV)
    ax.set_zlim(*zlim) # Customize the z-axis
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    return ax


def plot_contour(fig, f, xlim, ylim, xstride, ystride, levels=None, filled=False, clabel=False, subplot_coords=None):
    X, Y, Z = _make_3d_data(f, xlim, ylim, xstride, ystride)
    if subplot_coords is not None:
        ax = fig.add_subplot(*subplot_coords)
    else:
        ax = fig.add_subplot()
    if filled:
        if levels is not None:
            CS = ax.contourf(X, Y, Z, levels=levels, cmap=VIRIDIS_REV, zorder=1)
        else:
            CS = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap=VIRIDIS_REV, zorder=1)
    else:
        if levels is not None:
            CS = ax.contour(X, Y, Z, levels=levels, cmap=VIRIDIS_REV, zorder=1)
        else:
            CS = ax.contour(X, Y, Z, locator=ticker.LogLocator(), cmap=VIRIDIS_REV, zorder=1)
    if clabel:
        ax.clabel(CS, inline=True, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    return ax


def _make_3d_data(f, xlim, ylim, xstride, ystride):
    X = np.arange(xlim[0], xlim[1], xstride)
    Y = np.arange(ylim[0], ylim[1], ystride)
    X, Y = np.meshgrid(X, Y)
    Z = f(np.array([X, Y]))
    return X, Y, Z


def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    mean: array-like, shape (2, )
        Mean

    cov : array-like, shape (2, 2)
        Covariance matrix

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
