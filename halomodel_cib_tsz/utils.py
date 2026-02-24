"""
Integration helpers and utility functions.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate


def simps_log10(
    y: np.ndarray, log10_x: np.ndarray, axis: int = 0
) -> np.ndarray:
    """
    Simpson's integration over log10-spaced x values.

    Computes int y dlog10(x) using Simpson's rule.

    Parameters
    ----------
    y : array_like
        Integrand values.
    log10_x : array_like
        log10 of the x-values (1-D).
    axis : int
        Axis of *y* along which to integrate.

    Returns
    -------
    result : ndarray
        Definite integral approximation.
    """
    return integrate.simpson(y, x=log10_x, axis=axis)


def simps(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
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
        Definite integral approximation.
    """
    return integrate.simpson(y, x=x, axis=axis)
