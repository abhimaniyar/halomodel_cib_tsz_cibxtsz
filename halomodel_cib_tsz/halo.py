"""
Core halo model class wrapping colossus for all halo physics.

ALL colossus calls and h-unit conversions happen here. No other module
should import from colossus directly.
"""

import numpy as np
from scipy.special import sici
from scipy.interpolate import RectBivariateSpline

from colossus.cosmology import cosmology
from colossus.lss import mass_function, bias
from colossus.halo import concentration as conc_module

from . import config


class HaloModel:
    """
    Wrapper around colossus providing HMF, bias, concentration, NFW FT,
    P(k), and cosmological quantities — all in physical units (M_sun, Mpc).

    Parameters
    ----------
    cosmo_name : str
        Colossus cosmology name (default 'planck18').
    hmf_model : str
        HMF model name for colossus (default 'tinker08').
    bias_model : str
        Halo bias model name (default 'tinker10').
    conc_model : str
        Concentration-mass relation (default 'duffy08').
    mass_range : tuple
        (M_min, M_max) in M_sun.
    n_mass : int
        Number of log-spaced mass points.
    z_range : tuple
        (z_min, z_max).
    n_z : int
        Number of redshift points.
    ell_range : tuple
        (ell_min, ell_max).
    n_ell : int
        Number of log-spaced multipoles.
    """

    def __init__(self, cosmo_name=config.COSMO_NAME,
                 hmf_model=config.HMF_MODEL,
                 bias_model=config.BIAS_MODEL,
                 conc_model=config.CONC_MODEL,
                 mass_range=config.MASS_RANGE, n_mass=config.N_MASS,
                 z_range=config.Z_RANGE, n_z=config.N_Z,
                 ell_range=config.ELL_RANGE, n_ell=config.N_ELL):

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

    def _build_pk_interpolator(self):
        """Build 2D interpolator for P(k,z) in physical units."""
        k_grid = np.geomspace(1e-4, 200.0, 300)  # 1/Mpc
        Pk_table = np.zeros((len(k_grid), self.n_z))
        for j in range(self.n_z):
            Pk_table[:, j] = self._pk_colossus(k_grid, self.z[j])
        self._pk_k_grid = k_grid
        self._pk_interp = RectBivariateSpline(
            np.log(k_grid), self.z, np.log(np.clip(Pk_table, 1e-30, None))
        )

    def _pk_colossus(self, k_phys, z):
        """Call colossus P(k) with unit conversion. k in 1/Mpc, returns Mpc^3."""
        k_col = k_phys / self.h  # h/Mpc
        Pk_col = self._cosmo.matterPowerSpectrum(k_col, z=z)
        return Pk_col / self.h**3  # Mpc^3

    def _pk_interp_eval(self, k_phys, z):
        """Evaluate interpolated P(k,z) at scalar k, z. Returns float."""
        lnk = np.log(np.clip(k_phys, self._pk_k_grid[0], self._pk_k_grid[-1]))
        z_clip = np.clip(z, self.z[0], self.z[-1])
        # RectBivariateSpline returns 2D array; use grid=False for scalar eval
        return float(np.exp(self._pk_interp(lnk, z_clip, grid=False)))

    def power_spectrum(self, k, z):
        """
        Linear matter power spectrum P(k,z).

        Parameters
        ----------
        k : float or array
            Wavenumber in 1/Mpc.
        z : float
            Redshift.

        Returns
        -------
        P : float or array
            Power spectrum in Mpc^3.
        """
        k = np.atleast_1d(k)
        lnk = np.log(np.clip(k, self._pk_k_grid[0], self._pk_k_grid[-1]))
        z_clip = np.clip(z, self.z[0], self.z[-1])
        result = np.exp(self._pk_interp(lnk, z_clip, grid=False))
        return result if len(result) > 1 else float(result)

    # ── HMF, bias, concentration ─────────────────────────────────────────

    def hmf(self, M, z, mdef='200c'):
        """
        Halo mass function dn/dlnM in Mpc^{-3}.

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun.
        z : float
            Redshift.
        mdef : str
            Mass definition (e.g., '200c', '500c').
        """
        M = np.atleast_1d(M)
        M_col = M * self.h  # M_sun/h
        mf = mass_function.massFunction(
            M_col, z, q_in='M', q_out='dndlnM',
            mdef=mdef, model=self.hmf_model
        )
        return mf * self.h**3  # Mpc^{-3}

    def halo_bias(self, M, z, mdef='200c'):
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
        """
        M = np.atleast_1d(M)
        M_col = M * self.h
        return bias.haloBias(M_col, model=self.bias_model, z=z, mdef=mdef)

    def concentration(self, M, z, mdef='200c'):
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
        """
        M = np.atleast_1d(M)
        M_col = M * self.h
        return conc_module.concentration(M_col, mdef, z, model=self.conc_model)

    # ── Halo radius and NFW Fourier transform ────────────────────────────

    def rho_crit(self, z):
        """Critical density in M_sun / Mpc^3."""
        # colossus rho_crit is in M_sun h^2 / kpc^3
        # convert: (M_sun h^2 / kpc^3) * (kpc/Mpc)^3 / h^2 = M_sun / Mpc^3
        rho_col = self._cosmo.rho_c(z)  # M_sun h^2 / kpc^3
        return rho_col * 1e9 / self.h**2  # (1e3)^3 = 1e9 kpc^3/Mpc^3, / h^2

    def r_delta(self, M, z, mdef='200c'):
        """
        Halo radius R_Delta in Mpc.

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun.
        z : float
            Redshift.
        mdef : str
            Mass definition (e.g. '200c').
        """
        M = np.atleast_1d(M)
        delta = float(mdef[:-1])
        rho_ref = self.rho_crit(z)  # M_sun / Mpc^3
        r3 = 3.0 * M / (4.0 * np.pi * delta * rho_ref)
        return r3 ** (1.0 / 3.0)

    def nfw_fourier(self, k, M, z, mdef='200c'):
        """
        Normalised Fourier transform of the truncated NFW profile.

        Parameters
        ----------
        k : array_like
            Wavenumber in 1/Mpc. Shape (n_k,).
        M : array_like
            Halo mass in M_sun. Shape (n_M,).
        z : float
            Redshift.
        mdef : str
            Mass definition.

        Returns
        -------
        u : ndarray
            Shape (n_M, n_k). Values between 0 and 1.
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

    def comoving_distance(self, z):
        """Comoving distance chi(z) in Mpc."""
        z = np.atleast_1d(z)
        chi_col = np.array([self._cosmo.comovingDistance(0.0, zi) for zi in z])
        return chi_col / self.h  # Mpc

    def angular_diameter_distance(self, z):
        """Angular diameter distance d_A(z) in Mpc."""
        z = np.atleast_1d(z)
        chi = self.comoving_distance(z)
        return chi / (1.0 + z)

    def luminosity_distance(self, z):
        """Luminosity distance d_L(z) in Mpc."""
        z = np.atleast_1d(z)
        chi = self.comoving_distance(z)
        return chi * (1.0 + z)

    def dchi_dz(self, z):
        """dchi/dz = c / H(z) in Mpc."""
        z = np.atleast_1d(z)
        Hz = np.array([self._cosmo.Hz(zi) for zi in z])  # km/s/Mpc
        return config.c_light / Hz  # Mpc

    def E_z(self, z):
        """E(z) = H(z) / H0."""
        z = np.atleast_1d(z)
        return np.sqrt(self.Om0 * (1.0 + z)**3 + self.Ode0)

    def baryon_fraction(self, z):
        """Omega_b(z) / Omega_m(z). Constant in LCDM."""
        return self.Ob0 / self.Om0

    # ── Pre-compute arrays for CIB/tSZ ──────────────────────────────────

    def precompute(self, mdef='200c'):
        """
        Pre-compute HMF, bias, NFW FT arrays on the stored grids.

        Parameters
        ----------
        mdef : str
            Mass definition.

        Returns
        -------
        dict with keys:
            'hmf'  : (n_mass, n_z)
            'bias' : (n_mass, n_z)
            'nfw'  : (n_mass, n_ell, n_z)   NFW FT at Limber k values
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
