"""
tSZ halo model: pressure profile, y_ell, power spectra.

Matches the physics of Cell_tSZ.py from the original code.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import integrate

from . import config
from .utils import simps
from .sed import tsz_spectral_fn


class tSZModel:
    """
    tSZ angular power spectrum model.

    Uses the Arnaud+2010 generalised NFW pressure profile with a
    pre-tabulated y_ell integration kernel.

    Parameters
    ----------
    halo_model : HaloModel
        Pre-initialised halo model instance.
    freqs : list of float
        Observed frequencies in GHz.
    experiment : str
        Experiment name (for bandpassed f_nu values).
    B : float
        Hydrostatic mass bias factor. ``M_true = M_500 / B``.
    mdef : str
        Mass definition (default ``'500c'``).

    Raises
    ------
    ValueError
        If *B* is not positive.
    """

    def __init__(
        self,
        halo_model,
        freqs: list[float],
        experiment: str = 'Planck',
        B: float = config.TSZ_B,
        mdef: str = '500c',
    ) -> None:
        if B <= 0:
            raise ValueError(f"Mass bias B must be > 0, got {B}")

        self.hm = halo_model
        self.freqs = freqs
        self.nfreq = len(freqs)
        self.experiment = experiment
        self.B = B
        self.mdef = mdef

        # Grids
        self.mass = halo_model.mass
        self.z = halo_model.z
        self.ell = halo_model.ell
        self.log10_mass = halo_model.log10_mass
        self.n_mass = halo_model.n_mass
        self.n_z = halo_model.n_z
        self.n_ell = halo_model.n_ell

        # Tilded mass (true mass corrected for mass bias)
        self.M_tilde = self.mass / self.B  # (n_mass,)

        # Pressure profile params (Arnaud+2010)
        pp = config.PRESSURE_PROFILE
        self.P_0 = pp['P_0']
        self.c_500 = pp['c_500']
        self.gamma = pp['gamma']
        self.alpha = pp['alpha']
        self.beta = pp['beta']

        # Pre-compute halo quantities
        print("tSZ: pre-computing HMF, bias, NFW FT...")
        pre = halo_model.precompute(mdef)
        self.hmf_arr = pre['hmf']     # (n_mass, n_z)
        self.bias_arr = pre['bias']   # (n_mass, n_z)
        # NFW not needed for tSZ (we use y_ell instead)

        # Load pre-tabulated y_ell integration
        self._load_y_ell_table()

        # Pre-compute y_ell
        print("tSZ: computing y_ell...")
        self._y_ell = self._compute_y_ell()  # (n_ell, n_mass, n_z)

        # tSZ spectral function
        self._f_nu = tsz_spectral_fn(freqs, experiment)

    # ── Pressure profile ─────────────────────────────────────────────────

    def _r_delta(self, M: np.ndarray, z: float) -> np.ndarray:
        """R_500 of tilded mass in Mpc. M is the observed mass, M_tilde = M/B."""
        M_t = np.atleast_1d(M) / self.B
        return self.hm.r_delta(M_t, z, self.mdef)

    def _ell_delta(self, z: float) -> np.ndarray:
        """ell_500 = d_A / r_500 for each (mass, z). Shape ``(n_mass, n_z)``."""
        da = self.hm._da  # (n_z,)
        r500 = np.zeros((self.n_mass, self.n_z))
        for j in range(self.n_z):
            r500[:, j] = self._r_delta(self.mass, self.z[j])
        return da[None, :] / r500  # (n_mass, n_z)

    def _C_factor(self, z: np.ndarray) -> np.ndarray:
        """
        Pressure normalisation C(M, z) from Arnaud+2010.

        Parameters
        ----------
        z : ndarray
            Redshift grid.

        Returns
        -------
        C : ndarray, shape ``(n_mass, n_z)``
            Pressure normalisation in eV/cm^3.
        """
        M_t = self.M_tilde  # (n_mass,)
        Ez = self.hm.E_z(self.z)  # (n_z,)
        h70 = self.hm.h / 0.7

        a = 1.65 * h70**2 * Ez**(8.0 / 3.0)       # (n_z,)
        b = (h70 * M_t / 3e14)**(2.0 / 3.0 + 0.12)  # (n_mass,)

        return np.outer(b, a)  # (n_mass, n_z) in eV/cm^3

    # ── y_ell from tabulated integration ─────────────────────────────────

    def _load_y_ell_table(self) -> None:
        """Load pre-tabulated y_ell integration kernel."""
        fpath = os.path.join(config.DATA_DIR, 'y_ell_integration.txt')
        data = np.loadtxt(fpath)
        self._yl_lnx = data[:, 0]   # ln(ell/ell_500)
        self._yl_lny = data[:, 1]   # ln(integral value)

    def _compute_y_ell(self) -> np.ndarray:
        """
        Compute y_ell(ell, M, z) using pre-tabulated integration.

        Returns
        -------
        y_ell : ndarray, shape ``(n_ell, n_mass, n_z)``
        """
        # r500 and ell_500 for each (mass, z)
        r500 = np.zeros((self.n_mass, self.n_z))
        l500 = np.zeros((self.n_mass, self.n_z))
        for j in range(self.n_z):
            r500[:, j] = self._r_delta(self.mass, self.z[j])
            l500[:, j] = self.hm._da[j] / r500[:, j]

        r500_m = r500 * config.Mpc_to_m  # metres
        fact = (config.sigma_T / (config.m_e * (config.c_light * 1e3)**2)) * \
               (4.0 * np.pi * r500_m / l500**2)  # (n_mass, n_z)

        # Pressure normalisation C in SI (J/m^3)
        C_t = self._C_factor(self.z) * config.eV_to_J / config.cm_to_m**3  # (n_mass, n_z)

        y_ell = np.zeros((self.n_ell, self.n_mass, self.n_z))

        for i in range(self.n_ell):
            for j in range(self.n_mass):
                l_over_l500 = self.ell[i] / l500[j, :]  # (n_z,)
                y_int = np.interp(np.log(l_over_l500), self._yl_lnx, self._yl_lny)
                y_ell[i, j, :] = np.exp(y_int)

        return self.P_0 * y_ell * fact * C_t  # (n_ell, n_mass, n_z), dimensionless

    # ── Power spectra ────────────────────────────────────────────────────

    def f_nu(self) -> np.ndarray:
        """
        tSZ spectral function values.

        Returns
        -------
        f_nu : ndarray, shape ``(n_freq,)``
        """
        return self._f_nu

    def cl_1h(self) -> np.ndarray:
        """
        1-halo tSZ power spectrum.

        Returns
        -------
        Cl_1h : ndarray, shape ``(n_freq, n_freq, n_ell)``
            In Jy^2/sr (after Kcmb_MJy conversion).
        """
        if self.experiment == 'Planck':
            Kcmb_MJy = config.PLANCK['Kcmb_MJy']
        else:
            Kcmb_MJy = np.ones(self.nfreq)

        Cl_1h = np.zeros((self.nfreq, self.nfreq, self.n_ell))

        # dVc/dz = c * chi^2 / (H0 * E(z))
        dVc_dz = (config.c_light * self.hm._chi**2 /
                  (self.hm.H0 * self.hm.E_z(self.z)))  # (n_z,)

        y_l = self._y_ell      # (n_ell, n_mass, n_z)
        y_l2 = y_l**2
        hmf = self.hmf_arr     # (n_mass, n_z)

        # ∫ dlog10(M) hmf * y_l^2, then ∫ dz dVc_dz * [result]
        intgral1 = hmf * y_l2  # (n_ell, n_mass, n_z)
        intgn1 = simps(intgral1, self.log10_mass, axis=1)  # (n_ell, n_z)
        intgral2 = dVc_dz * intgn1  # (n_ell, n_z)
        res = simps(intgral2, self.z, axis=1)  # (n_ell,)

        fnu = self._f_nu * 1e6 * Kcmb_MJy  # (n_freq,)
        for f in range(self.nfreq):
            Cl_1h[f, :, :] = np.outer(fnu, res) * fnu[f]

        return Cl_1h

    def cl_2h(self) -> np.ndarray:
        """
        2-halo tSZ power spectrum.

        Returns
        -------
        Cl_2h : ndarray, shape ``(n_freq, n_freq, n_ell)``
            In Jy^2/sr.
        """
        if self.experiment == 'Planck':
            Kcmb_MJy = config.PLANCK['Kcmb_MJy']
        else:
            Kcmb_MJy = np.ones(self.nfreq)

        Cl_2h = np.zeros((self.nfreq, self.nfreq, self.n_ell))

        dVc_dz = (config.c_light * self.hm._chi**2 /
                  (self.hm.H0 * self.hm.E_z(self.z)))

        y_l = self._y_ell      # (n_ell, n_mass, n_z)
        hmf = self.hmf_arr     # (n_mass, n_z)
        b = self.bias_arr      # (n_mass, n_z)

        # ∫ dlog10(M) hmf * b * y_l → squared
        intgrl = hmf * b * y_l  # (n_ell, n_mass, n_z)
        res_m = simps(intgrl, self.log10_mass, axis=1)  # (n_ell, n_z)
        ylhmfbias2 = res_m**2   # (n_ell, n_z)

        # ∫ dz dVc_dz * P(k) * [hmf*b*y_l]^2
        Pk = self.hm._Pk_limber  # (n_ell, n_z)
        intgrl2 = dVc_dz * Pk * ylhmfbias2  # (n_ell, n_z)
        res = simps(intgrl2, self.z, axis=1)  # (n_ell,)

        fnu = self._f_nu * 1e6 * Kcmb_MJy
        for f in range(self.nfreq):
            Cl_2h[f, :, :] = np.outer(fnu, res) * fnu[f]

        return Cl_2h
