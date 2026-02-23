"""
Integration helpers and utility functions.
"""

import numpy as np
from scipy import integrate


def simps_log10(y, log10_x, axis=0):
    """
    Simpson's integration over log10-spaced x values.

    Computes ∫ y dlog10(x) using Simpson's rule, matching the original code's
    ``intg.simps(y, x=np.log10(x), axis=..., even='avg')``.

    Parameters
    ----------
    y : array_like
        Integrand values.
    log10_x : array_like
        log10 of the x-values (1-D).
    axis : int
        Axis of y along which to integrate.

    Returns
    -------
    result : ndarray
        Definite integral approximation.
    """
    return integrate.simpson(y, x=log10_x, axis=axis)


def simps(y, x, axis=0):
    """
    Simpson's integration wrapper.

    Parameters
    ----------
    y : array_like
        Integrand values.
    x : array_like
        Abscissae (1-D).
    axis : int
        Axis along which to integrate.

    Returns
    -------
    result : ndarray
    """
    return integrate.simpson(y, x=x, axis=axis)
