import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm, ticker

VIRIDIS_REV = cm.viridis.reversed()

def plot_surface(fig, f, xlim, ylim, zlim, xstride, ystride, subplot_coords=None):
    X, Y, Z = _make_3d_data(f, xlim, ylim, xstride, ystride)
    if subplot_coords is not None:
        ax = fig.add_subplot(*subplot_coords, projection='3d')
    else:
        ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=VIRIDIS_REV)
    ax.set_zlim(*zlim) # Customize the z-axis
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    return ax

def plot_contour(fig, f, xlim, ylim, xstride, ystride, levels=None, filled=False, clabel=False, subplot_coords=None):
    X, Y, Z = _make_3d_data(f, xlim, ylim, xstride, ystride)
    if subplot_coords is not None:
        ax = fig.add_subplot(*subplot_coords)
    else:
        ax = fig.add_subplot()
    if filled:
        if levels is not None:
            CS = ax.contourf(X, Y, Z, levels=levels, cmap=VIRIDIS_REV)
        else:
            CS = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap=VIRIDIS_REV)
    else:
        if levels is not None:
            CS = ax.contour(X, Y, Z, levels=levels, cmap=VIRIDIS_REV)
        else:
            CS = ax.contour(X, Y, Z, locator=ticker.LogLocator(), cmap=VIRIDIS_REV)
    if clabel:
        ax.clabel(CS, inline=True, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    return ax

def _make_3d_data(f, xlim, ylim, xstride, ystride):
    X = np.arange(xlim[0], xlim[1], xstride)
    Y = np.arange(ylim[0], ylim[1], ystride)
    X, Y = np.meshgrid(X, Y)
    Z = f(np.array([X, Y]))
    return X, Y, Z
