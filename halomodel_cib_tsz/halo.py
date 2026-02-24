"""
Core halo model class wrapping colossus for all halo physics.

ALL colossus calls and h-unit conversions happen here. No other module
should import from colossus directly.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sici
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline

from colossus.cosmology import cosmology
from colossus.lss import mass_function, bias, peaks
from colossus.halo import concentration as conc_module

from . import config


# ── Original Tinker08 HMF (spline-interpolated parameters) ─────────────

_TINKER08_PARAMS = {
    "A_200": 1.858659e-01, "A_300": 1.995973e-01, "A_400": 2.115659e-01,
    "A_600": 2.184113e-01, "A_800": 2.480968e-01, "A_1200": 2.546053e-01,
    "A_1600": 2.600000e-01, "A_2400": 2.600000e-01, "A_3200": 2.600000e-01,
    "a_200": 1.466904, "a_300": 1.521782, "a_400": 1.559186,
    "a_600": 1.614585, "a_800": 1.869936, "a_1200": 2.128056,
    "a_1600": 2.301275, "a_2400": 2.529241, "a_3200": 2.661983,
    "b_200": 2.571104, "b_300": 2.254217, "b_400": 2.048674,
    "b_600": 1.869559, "b_800": 1.588649, "b_1200": 1.507134,
    "b_1600": 1.464374, "b_2400": 1.436827, "b_3200": 1.405210,
    "c_200": 1.193958, "c_300": 1.270316, "c_400": 1.335191,
    "c_600": 1.446266, "c_800": 1.581345, "c_1200": 1.795050,
    "c_1600": 1.965613, "c_2400": 2.237466, "c_3200": 2.439729,
}
_DELTA_VIRS = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
_T08_A = InterpolatedUnivariateSpline(
    _DELTA_VIRS, [_TINKER08_PARAMS[f"A_{d}"] for d in _DELTA_VIRS])
_T08_a = InterpolatedUnivariateSpline(
    _DELTA_VIRS, [_TINKER08_PARAMS[f"a_{d}"] for d in _DELTA_VIRS])
_T08_b = InterpolatedUnivariateSpline(
    _DELTA_VIRS, [_TINKER08_PARAMS[f"b_{d}"] for d in _DELTA_VIRS])
_T08_c = InterpolatedUnivariateSpline(
    _DELTA_VIRS, [_TINKER08_PARAMS[f"c_{d}"] for d in _DELTA_VIRS])


def _tinker08_fsigma(sigma, z, delta_halo):
    """Tinker08 f(sigma) with cubic spline interpolation of parameters.

    This matches the original implementation in Maniyar+2021, which uses
    ``InterpolatedUnivariateSpline`` to interpolate the Tinker08 fitting
    parameters at the overdensity ``delta_halo`` (w.r.t. mean density).
    Colossus uses a different interpolation scheme, leading to 10-15%
    differences at z > 0 near M_eff.
    """
    A_0 = float(_T08_A(delta_halo))
    a_0 = float(_T08_a(delta_halo))
    b_0 = float(_T08_b(delta_halo))
    c_0 = float(_T08_c(delta_halo))
    A = A_0 * (1.0 + z) ** (-0.14)
    a = a_0 * (1.0 + z) ** 0.06
    alpha = 10.0 ** (-(0.75 / np.log10(delta_halo / 75.0)) ** 1.2)
    b = b_0 * (1.0 + z) ** alpha
    return A * ((sigma / b) ** (-a) + 1.0) * np.exp(-c_0 / sigma ** 2)


class HaloModel:
    """
    Wrapper around colossus providing HMF, bias, concentration, NFW FT,
    P(k), and cosmological quantities — all in physical units (M_sun, Mpc).

    Parameters
    ----------
    cosmo_name : str
        Colossus cosmology name (default ``'planck18'``).
    hmf_model : str
        HMF model name for colossus (default ``'tinker08'``).
    bias_model : str
        Halo bias model name (default ``'tinker10'``).
    conc_model : str
        Concentration-mass relation (default ``'duffy08'``).
    mass_range : tuple of float
        ``(M_min, M_max)`` in M_sun.
    n_mass : int
        Number of log-spaced mass points.
    z_range : tuple of float
        ``(z_min, z_max)``.
    n_z : int
        Number of redshift points.
    ell_range : tuple of float
        ``(ell_min, ell_max)``.
    n_ell : int
        Number of log-spaced multipoles.

    Attributes
    ----------
    mass : ndarray
        Log-spaced halo mass grid in M_sun, shape ``(n_mass,)``.
    z : ndarray
        Linearly-spaced redshift grid, shape ``(n_z,)``.
    ell : ndarray
        Log-spaced multipole grid, shape ``(n_ell,)``.
    h : float
        Dimensionless Hubble parameter h = H0/100.
    """

    def __init__(
        self,
        cosmo_name: str = config.COSMO_NAME,
        hmf_model: str = config.HMF_MODEL,
        bias_model: str = config.BIAS_MODEL,
        conc_model: str = config.CONC_MODEL,
        mass_range: tuple[float, float] = config.MASS_RANGE,
        n_mass: int = config.N_MASS,
        z_range: tuple[float, float] = config.Z_RANGE,
        n_z: int = config.N_Z,
        ell_range: tuple[float, float] = config.ELL_RANGE,
        n_ell: int = config.N_ELL,
    ) -> None:
        # Set colossus cosmology (global state)
        self._cosmo = cosmology.setCosmology(cosmo_name)
        self.h = self._cosmo.H0 / 100.0
        self.H0 = self._cosmo.H0  # km/s/Mpc
        self.Om0 = self._cosmo.Om0
        self.Ob0 = self._cosmo.Ob0
        self.Ode0 = self._cosmo.Ode0

        self.hmf_model = hmf_model
        self.bias_model = bias_model
        self.conc_model = conc_model

        # Build grids
        self.mass = np.geomspace(mass_range[0], mass_range[1], n_mass)
        self.log10_mass = np.log10(self.mass)
        self.z = np.linspace(z_range[0], z_range[1], n_z)
        self.ell = np.geomspace(ell_range[0], ell_range[1], n_ell)
        self.n_mass = n_mass
        self.n_z = n_z
        self.n_ell = n_ell

        # Pre-compute cosmological distances
        self._chi = self.comoving_distance(self.z)          # [n_z] Mpc
        self._da = self.angular_diameter_distance(self.z)    # [n_z] Mpc
        self._dchi_dz = self.dchi_dz(self.z)                 # [n_z] Mpc

        # Pre-compute P(k,z) interpolator
        self._build_pk_interpolator()

        # P(k) at Limber wavenumbers: k = (ell + 0.5) / chi
        self._Pk_limber = np.zeros((self.n_ell, self.n_z))
        for i in range(self.n_ell):
            k_limber = (self.ell[i] + 0.5) / self._chi
            for j in range(self.n_z):
                self._Pk_limber[i, j] = self._pk_interp_eval(k_limber[j], self.z[j])

    # ── P(k,z) interpolation ────────────────────────────────────────────

    def _build_pk_interpolator(self) -> None:
        """Build 2D interpolator for P(k,z) in physical units."""
        k_grid = np.geomspace(1e-4, 200.0, 300)  # 1/Mpc
        Pk_table = np.zeros((len(k_grid), self.n_z))
        for j in range(self.n_z):
            Pk_table[:, j] = self._pk_colossus(k_grid, self.z[j])
        self._pk_k_grid = k_grid
        self._pk_interp = RectBivariateSpline(
            np.log(k_grid), self.z, np.log(np.clip(Pk_table, 1e-30, None))
        )

    def _pk_colossus(self, k_phys: np.ndarray, z: float) -> np.ndarray:
        """Call colossus P(k) with unit conversion. k in 1/Mpc, returns Mpc^3."""
        k_col = k_phys / self.h  # h/Mpc
        Pk_col = self._cosmo.matterPowerSpectrum(k_col, z=z)
        return Pk_col / self.h**3  # Mpc^3

    def _pk_interp_eval(self, k_phys: float, z: float) -> float:
        """Evaluate interpolated P(k,z) at scalar k, z. Returns float."""
        lnk = np.log(np.clip(k_phys, self._pk_k_grid[0], self._pk_k_grid[-1]))
        z_clip = np.clip(z, self.z[0], self.z[-1])
        # RectBivariateSpline returns 2D array; use grid=False for scalar eval
        return float(np.exp(self._pk_interp(lnk, z_clip, grid=False)))

    def power_spectrum(self, k: np.ndarray | float, z: float) -> np.ndarray | float:
        """
        Linear matter power spectrum P(k,z).

        Parameters
        ----------
        k : float or array_like
            Wavenumber in 1/Mpc.
        z : float
            Redshift.

        Returns
        -------
        P : float or ndarray
            Power spectrum in Mpc^3.
        """
        k = np.atleast_1d(k)
        lnk = np.log(np.clip(k, self._pk_k_grid[0], self._pk_k_grid[-1]))
        z_clip = np.clip(z, self.z[0], self.z[-1])
        result = np.exp(self._pk_interp(lnk, z_clip, grid=False))
        return result if len(result) > 1 else float(result)

    # ── HMF, bias, concentration ─────────────────────────────────────────

    def hmf(self, M: np.ndarray, z: float, mdef: str = '200c') -> np.ndarray:
        """
        Halo mass function dn/dlog10(M).

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun.
        z : float
            Redshift.
        mdef : str
            Mass definition (e.g. ``'200c'``, ``'500c'``).

        Returns
        -------
        dndlog10M : ndarray
            Halo mass function in Mpc^{-3}, per unit log10(M).

        Notes
        -----
        When ``hmf_model='tinker08_original'``, uses the original
        Maniyar+2021 implementation with cubic-spline interpolation of
        Tinker08 parameters. This gives 10-15% higher HMF at z > 0 near
        M_eff compared to the colossus ``'tinker08'`` implementation.
        """
        M = np.atleast_1d(M)

        if self.hmf_model == 'tinker08_original':
            return self._hmf_tinker08_original(M, z, mdef)

        M_col = M * self.h  # M_sun/h
        mf = mass_function.massFunction(
            M_col, z, q_in='M', q_out='dndlnM',
            mdef=mdef, model=self.hmf_model
        )
        return mf * self.h**3 * np.log(10)  # dn/dlnM → dn/dlog10M

    def _hmf_tinker08_original(
        self, M: np.ndarray, z: float, mdef: str = '200c'
    ) -> np.ndarray:
        """Original Tinker08 HMF with cubic spline parameter interpolation.

        Uses colossus for sigma(M) but applies the original Tinker08
        f(sigma) formula with ``InterpolatedUnivariateSpline`` for
        parameter interpolation at delta_mean = delta / Omega_m(z).
        """
        h = self.h

        # Mean matter density at z=0 in physical Msun/Mpc^3
        rho_m0 = self.Om0 * 2.775e11 * h**2

        # Lagrangian radius in Mpc/h (colossus units)
        M_col = M * h
        R_col = peaks.lagrangianR(M_col)

        # sigma(M, z) from colossus
        sigma = self._cosmo.sigma(R_col, z=z)

        # d(ln sigma)/d(ln M) via finite differences
        eps = 0.005
        sig_p = self._cosmo.sigma(R_col * (1.0 + eps), z=z)
        sig_m = self._cosmo.sigma(R_col * (1.0 - eps), z=z)
        dlns_dlnR = (np.log(sig_p) - np.log(sig_m)) / (2.0 * eps)
        dlns_dlnm = dlns_dlnR / 3.0

        # Convert delta_c to delta_mean = delta / Omega_m(z)
        delta = float(mdef[:-1])
        Om_z = self.Om0 * (1.0 + z)**3 / (self.Om0 * (1.0 + z)**3 + self.Ode0)
        delta_halo = delta / Om_z

        # f(sigma) with original spline interpolation
        f = _tinker08_fsigma(sigma, z, delta_halo)

        # dn/dlog10M = f(sigma) * rho_m0 * |dlns/dlnM| / M * ln(10)
        return f * rho_m0 * np.abs(dlns_dlnm) / M * np.log(10)

    def halo_bias(self, M: np.ndarray, z: float, mdef: str = '200c') -> np.ndarray:
        """
        Halo bias b(M,z), dimensionless.

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun.
        z : float
            Redshift.
        mdef : str
            Mass definition.

        Returns
        -------
        b : ndarray
            Dimensionless halo bias.
        """
        M = np.atleast_1d(M)
        M_col = M * self.h
        return bias.haloBias(M_col, model=self.bias_model, z=z, mdef=mdef)

    def concentration(self, M: np.ndarray, z: float, mdef: str = '200c') -> np.ndarray:
        """
        Halo concentration c(M,z), dimensionless.

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun.
        z : float
            Redshift.
        mdef : str
            Mass definition.

        Returns
        -------
        c : ndarray
            Halo concentration.
        """
        M = np.atleast_1d(M)
        M_col = M * self.h
        return conc_module.concentration(M_col, mdef, z, model=self.conc_model)

    # ── Halo radius and NFW Fourier transform ────────────────────────────

    def rho_crit(self, z: float) -> float:
        """
        Critical density at redshift *z*.

        Returns
        -------
        rho_c : float
            Critical density in M_sun / Mpc^3.
        """
        # colossus rho_c is in (M_sun/h) / (kpc/h)^3 = M_sun h^2 / kpc^3
        # Physical: (value/h) / (value/h)^3 → × h^2, then × 1e9 for kpc^3→Mpc^3
        rho_col = self._cosmo.rho_c(z)  # M_sun h^2 / kpc^3
        return rho_col * 1e9 * self.h**2  # physical M_sun / Mpc^3

    def r_delta(self, M: np.ndarray, z: float, mdef: str = '200c') -> np.ndarray:
        """
        Halo radius R_Delta in Mpc.

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun.
        z : float
            Redshift.
        mdef : str
            Mass definition (e.g. ``'200c'``).

        Returns
        -------
        r : ndarray
            Halo radius in Mpc.
        """
        M = np.atleast_1d(M)
        delta = float(mdef[:-1])
        rho_ref = self.rho_crit(z)  # M_sun / Mpc^3
        r3 = 3.0 * M / (4.0 * np.pi * delta * rho_ref)
        return r3 ** (1.0 / 3.0)

    def nfw_fourier(
        self, k: np.ndarray, M: np.ndarray, z: float, mdef: str = '200c'
    ) -> np.ndarray:
        """
        Normalised Fourier transform of the truncated NFW profile.

        Parameters
        ----------
        k : array_like
            Wavenumber in 1/Mpc, shape ``(n_k,)``.
        M : array_like
            Halo mass in M_sun, shape ``(n_M,)``.
        z : float
            Redshift.
        mdef : str
            Mass definition.

        Returns
        -------
        u : ndarray
            NFW Fourier transform, shape ``(n_M, n_k)``. Values between 0
            and 1.
        """
        k = np.atleast_1d(k)
        M = np.atleast_1d(M)

        c = self.concentration(M, z, mdef)       # (n_M,)
        r_d = self.r_delta(M, z, mdef)            # (n_M,)
        r_s = r_d / c                              # (n_M,)

        # eta = k * r_s, shape (n_M, n_k)
        eta = np.outer(r_s, k)                     # (n_M, n_k)
        eta_c = eta * (1.0 + c[:, None])           # (n_M, n_k)

        # Amplitude factor: 1 / [ln(1+c) - c/(1+c)]
        amp = 1.0 / (np.log(1.0 + c) - c / (1.0 + c))  # (n_M,)

        # Si/Ci integrals
        Si_eta, Ci_eta = sici(eta)
        Si_eta_c, Ci_eta_c = sici(eta_c)

        u = amp[:, None] * (
            np.sin(eta) * (Si_eta_c - Si_eta)
            + np.cos(eta) * (Ci_eta_c - Ci_eta)
            - np.sin(c[:, None] * eta) / eta_c
        )

        # Handle k→0 limit
        u = np.where(eta < 1e-10, 1.0, u)
        return u

    # ── Cosmological distances ───────────────────────────────────────────

    def comoving_distance(self, z: np.ndarray | float) -> np.ndarray:
        """
        Comoving distance chi(z) in Mpc.

        Parameters
        ----------
        z : float or array_like
            Redshift(s).

        Returns
        -------
        chi : ndarray
            Comoving distance in Mpc.
        """
        z = np.atleast_1d(z)
        chi_col = np.array([self._cosmo.comovingDistance(0.0, zi) for zi in z])
        return chi_col / self.h  # Mpc

    def angular_diameter_distance(self, z: np.ndarray | float) -> np.ndarray:
        """
        Angular diameter distance d_A(z) in Mpc.

        Parameters
        ----------
        z : float or array_like
            Redshift(s).

        Returns
        -------
        d_A : ndarray
            Angular diameter distance in Mpc.
        """
        z = np.atleast_1d(z)
        chi = self.comoving_distance(z)
        return chi / (1.0 + z)

    def luminosity_distance(self, z: np.ndarray | float) -> np.ndarray:
        """
        Luminosity distance d_L(z) in Mpc.

        Parameters
        ----------
        z : float or array_like
            Redshift(s).

        Returns
        -------
        d_L : ndarray
            Luminosity distance in Mpc.
        """
        z = np.atleast_1d(z)
        chi = self.comoving_distance(z)
        return chi * (1.0 + z)

    def dchi_dz(self, z: np.ndarray | float) -> np.ndarray:
        """
        Derivative of comoving distance, dchi/dz = c / H(z), in Mpc.

        Parameters
        ----------
        z : float or array_like
            Redshift(s).

        Returns
        -------
        dchi_dz : ndarray
            dchi/dz in Mpc.
        """
        z = np.atleast_1d(z)
        Hz = np.array([self._cosmo.Hz(zi) for zi in z])  # km/s/Mpc
        return config.c_light / Hz  # Mpc

    def E_z(self, z: np.ndarray | float) -> np.ndarray:
        """
        Dimensionless Hubble parameter E(z) = H(z) / H0.

        Parameters
        ----------
        z : float or array_like
            Redshift(s).

        Returns
        -------
        E : ndarray
        """
        z = np.atleast_1d(z)
        return np.sqrt(self.Om0 * (1.0 + z)**3 + self.Ode0)

    def baryon_fraction(self, z: float) -> float:
        """
        Cosmic baryon fraction Omega_b / Omega_m (constant in LCDM).

        Parameters
        ----------
        z : float
            Redshift (unused, included for API consistency).

        Returns
        -------
        f_b : float
        """
        return self.Ob0 / self.Om0

    # ── Pre-compute arrays for CIB/tSZ ──────────────────────────────────

    def precompute(self, mdef: str = '200c') -> dict[str, np.ndarray]:
        """
        Pre-compute HMF, bias, NFW FT arrays on the stored grids.

        Parameters
        ----------
        mdef : str
            Mass definition.

        Returns
        -------
        arrays : dict
            ``'hmf'``: shape ``(n_mass, n_z)``,
            ``'bias'``: shape ``(n_mass, n_z)``,
            ``'nfw'``: shape ``(n_mass, n_ell, n_z)`` — NFW FT at Limber k.
        """
        hmf_arr = np.zeros((self.n_mass, self.n_z))
        bias_arr = np.zeros((self.n_mass, self.n_z))
        nfw_arr = np.zeros((self.n_mass, self.n_ell, self.n_z))

        for j in range(self.n_z):
            z_j = self.z[j]
            hmf_arr[:, j] = self.hmf(self.mass, z_j, mdef)
            bias_arr[:, j] = self.halo_bias(self.mass, z_j, mdef)

            # NFW FT at Limber k for each ell
            k_limber = (self.ell + 0.5) / self._chi[j]  # (n_ell,)
            nfw_arr[:, :, j] = self.nfw_fourier(k_limber, self.mass, z_j, mdef)

        return {
            'hmf': hmf_arr,
            'bias': bias_arr,
            'nfw': nfw_arr,
        }
