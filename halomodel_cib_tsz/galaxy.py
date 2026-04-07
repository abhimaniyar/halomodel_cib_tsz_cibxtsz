"""
Galaxy Halo Occupation Distribution (HOD) model.

Implements the More+2015 HOD parametrization with support for
CMASS, DESI_LRG, and DESI_ELG galaxy surveys.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import special
from scipy.interpolate import UnivariateSpline

from . import config
from .utils import simps


class GalaxyHOD:
    """
    Halo Occupation Distribution for galaxies.

    Parametrises the mean number of central and satellite galaxies
    per halo as a function of halo mass, following More+2015.

    Parameters
    ----------
    survey : str
        Galaxy survey name: ``'CMASS'``, ``'DESI_LRG'``, or ``'DESI_ELG'``.
    params : dict, optional
        HOD parameters. If *None*, uses defaults for the given survey.
    dndz_file : str, optional
        Path to a dN/dz data file. If *None*, uses the default file
        for the given survey from the package data directory.
    """

    def __init__(
        self,
        survey: str = 'CMASS',
        params: dict | None = None,
        dndz_file: str | None = None,
    ) -> None:
        self.survey = survey

        if params is None:
            if survey not in config.GALAXY_SURVEYS:
                raise ValueError(
                    f"Unknown survey '{survey}'. "
                    f"Valid: {list(config.GALAXY_SURVEYS.keys())}"
                )
            params = config.GALAXY_SURVEYS[survey]

        self.log10mMin = params['log10mMin']
        self.sLog10m = params['sLog10m']
        self.alpha = params['alpha']
        self.m1 = params['m1']
        self.kappa = params['kappa']
        self.mMinHod = params['mMinHod']
        self.alphaInc = params.get('alphaInc', 0.51)
        self.log10mInc = params.get('log10mInc', 13.84)

        # ELG-specific
        self._is_elg = 'Ac' in params
        if self._is_elg:
            self.Ac = params['Ac']
            self.log10mc = params['log10mc']
            self.gamma = params['gamma']
            self.m0 = params['m0']

        # Load dN/dz
        self._dndz_file = dndz_file
        self._dndz_spline = None

    # ── Occupation numbers ───────────────────────────────────────────────

    def Ncen(self, M: np.ndarray) -> np.ndarray:
        """
        Mean number of central galaxies per halo.

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun, shape ``(n_M,)``.

        Returns
        -------
        Ncen : ndarray, shape ``(n_M,)``
        """
        M = np.atleast_1d(M).astype(float)
        if self._is_elg:
            y = (np.log10(M) - self.log10mc) / self.sLog10m
            return (self.Ac / (np.sqrt(2 * np.pi) * self.sLog10m)
                    * np.exp(-0.5 * y**2)
                    * (1.0 + special.erf(self.gamma * y / np.sqrt(2))))
        else:
            x = (np.log10(M) - self.log10mMin) / self.sLog10m
            result = 0.5 * (1.0 + special.erf(x))
            fInc = np.clip(
                1.0 + self.alphaInc * (np.log10(M) - self.log10mInc), 0.0, 1.0
            )
            return result * fInc

    def Nsat(self, M: np.ndarray) -> np.ndarray:
        """
        Mean number of satellite galaxies per halo.

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun, shape ``(n_M,)``.

        Returns
        -------
        Nsat : ndarray, shape ``(n_M,)``
        """
        M = np.atleast_1d(M).astype(float)
        if self._is_elg:
            diff = M - self.m0
            return np.where(diff > 0, (diff / self.m1)**self.alpha, 0.0)
        else:
            diff = M - self.kappa * self.mMinHod
            return np.where(
                diff > 0,
                (diff / self.m1)**self.alpha * self.Ncen(M),
                0.0,
            )

    def Ntot(self, M: np.ndarray) -> np.ndarray:
        """Total mean occupation N_cen + N_sat."""
        return self.Ncen(M) + self.Nsat(M)

    # ── Mean number density ──────────────────────────────────────────────

    def nbar(self, mass: np.ndarray, hmf_arr: np.ndarray) -> np.ndarray:
        """
        Mean comoving galaxy number density at each redshift.

        Parameters
        ----------
        mass : ndarray, shape ``(n_mass,)``
            Halo mass grid in M_sun.
        hmf_arr : ndarray, shape ``(n_mass, n_z)``
            Halo mass function dn/dlog10(M) on the grid.

        Returns
        -------
        nbar : ndarray, shape ``(n_z,)``
            Mean galaxy number density in Mpc^{-3}.
        """
        Ng = self.Ntot(mass)  # (n_mass,)
        integrand = hmf_arr * Ng[:, None]  # (n_mass, n_z)
        return simps(integrand, np.log10(mass), axis=0)

    # ── Redshift distribution ────────────────────────────────────────────

    def load_dndz(self, z_grid: np.ndarray | None = None) -> callable:
        """
        Load the observed dN/dz and return an interpolating function.

        Parameters
        ----------
        z_grid : ndarray, optional
            If provided, evaluate dN/dz on this grid and return the array.
            Otherwise, return a callable.

        Returns
        -------
        dndz : callable or ndarray
            dN/dz in galaxies/sr/dz. If *z_grid* is given, returns
            an ndarray of shape ``(n_z,)``.
        """
        if self._dndz_spline is None:
            self._dndz_spline = self._build_dndz_spline()

        if z_grid is not None:
            return self._dndz_spline(z_grid)
        return self._dndz_spline

    def _build_dndz_spline(self) -> callable:
        """Build a spline interpolator for dN/dz."""
        fpath = self._dndz_file
        if fpath is None:
            if self.survey == 'CMASS':
                fname = 'dn_dz_cmass.txt'
            elif self.survey in ('DESI_LRG', 'DESI_ELG'):
                fname = f'dndz_{self.survey}.txt'
            else:
                raise ValueError(
                    f"No default dN/dz file for survey '{self.survey}'. "
                    "Provide dndz_file explicitly."
                )
            fpath = os.path.join(config.DATA_DIR, fname)

        data = np.loadtxt(fpath)

        if self.survey == 'CMASS':
            Z = data[:, 0]
            Dndz = data[:, 1]
        else:
            # DESI format: zmin, zmax, N_per_sq_deg
            Z = (data[:, 0] + data[:, 1]) / 2.0
            deltaz = data[:, 1] - data[:, 0]
            # Convert from per sq. degree to per steradian
            ngal = data[:, 2] / (np.pi / 180.0)**2
            Dndz = ngal / deltaz

        zmin, zmax = Z.min(), Z.max()
        f = UnivariateSpline(Z, Dndz, k=1, s=0, ext=1)

        def dndz(z):
            z = np.atleast_1d(z)
            return f(z) * (z >= zmin) * (z <= zmax)

        return dndz

    def window(self, z: np.ndarray) -> np.ndarray:
        """
        Normalised galaxy selection function phi(z) = (dN/dz) / N_bar.

        Parameters
        ----------
        z : ndarray, shape ``(n_z,)``
            Redshift grid.

        Returns
        -------
        phi : ndarray, shape ``(n_z,)``
            Normalised selection function (integrates to 1).
        """
        dndz = self.load_dndz(z)
        Nbar = simps(dndz, z)
        if Nbar == 0:
            return np.zeros_like(z)
        return dndz / Nbar

    def shot_noise(self) -> float:
        """
        Galaxy shot noise C_ℓ^{shot} = 1/n_gal_tot.

        Returns
        -------
        shot : float
            Shot noise power in sr.
        """
        dndz_fn = self.load_dndz()

        if self.survey == 'CMASS':
            data = np.loadtxt(
                self._dndz_file or os.path.join(config.DATA_DIR, 'dn_dz_cmass.txt')
            )
            Z, Dndz = data[:, 0], data[:, 1]
            ngal_tot = simps(Dndz, Z)
        else:
            fname = self._dndz_file or os.path.join(
                config.DATA_DIR, f'dndz_{self.survey}.txt'
            )
            data = np.loadtxt(fname)
            ngal = data[:, 2] / (np.pi / 180.0)**2
            ngal_tot = np.sum(ngal)

        return 1.0 / ngal_tot if ngal_tot > 0 else 0.0
