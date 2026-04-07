"""
Tracer framework for generic halo model cross-correlations.

Each LSS tracer (CIB, tSZ, galaxies, lensing, ...) is represented as a
:class:`Tracer` subclass that provides:

* **central_weight / satellite_weight** — per-halo weights (without HMF).
  The 1-halo cross-spectrum uses ``c_A*s_B + s_A*c_B + s_A*s_B``
  (excludes the central×central shot-noise term).
  For continuous-field tracers (tSZ, lensing) the full profile goes
  into ``satellite_weight`` and ``central_weight`` returns zeros.

* **window** — radial window function ``W(z)`` for Limber projection.

The angular power spectrum (computed by :class:`AngularCrossSpectrum`)
uses the universal Limber formula:

.. math::

    C_\\ell^{AB} = \\int dz\\,\\frac{d\\chi/dz}{\\chi^2}\\,
    W_A(z)\\, W_B(z)\\; P^{AB}_{\\rm hm}(k=(\\ell+0.5)/\\chi,\\, z)

Physical window functions (following Maniyar+2022, Eqs. A15-A17):

* **CIB**: ``W(z) = a(z) × j̄(ν,z)`` where ``a = 1/(1+z)``.
  With un-normalised halo weights (used here), this simplifies to
  ``W(z) = a(z) = 1/(1+z)`` and the weights carry the full emissivity.

* **Galaxy**: ``W(z) = (dz/dχ) × (dN/dz) / N̄ = φ(z) / (dχ/dz)``
  where ``φ(z)`` is the normalised redshift distribution.

* **tSZ**: ``W(z) = χ(z)²`` so that ``(dχ/dz)/χ² × W² = dV_c/dz``.

* **CMB lensing** (future): standard lensing kernel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from . import config
from .utils import simps
from .cib import CIBModel
from .tsz import tSZModel
from .galaxy import GalaxyHOD


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class Tracer(ABC):
    """
    Abstract base class for an LSS tracer in the halo model.

    Parameters
    ----------
    halo_model : HaloModel
        Shared halo model instance.
    mdef : str
        Halo mass definition (e.g. ``'200c'``, ``'500c'``).
    """

    def __init__(self, halo_model, mdef: str = '200c') -> None:
        self.hm = halo_model
        self.mdef = mdef
        self.mass = halo_model.mass
        self.z = halo_model.z
        self.ell = halo_model.ell
        self.log10_mass = halo_model.log10_mass
        self.n_mass = halo_model.n_mass
        self.n_z = halo_model.n_z
        self.n_ell = halo_model.n_ell

    # ── Interface ────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def n_fields(self) -> int:
        """Number of observable fields (1 for scalar, n_freq for CIB/tSZ)."""

    @abstractmethod
    def central_weight(self, ell_idx: int) -> np.ndarray:
        """
        Per-halo weight for the central component (without HMF).

        Parameters
        ----------
        ell_idx : int
            Index into the multipole grid.

        Returns
        -------
        w_c : ndarray, shape ``(n_fields, n_mass, n_z)``
        """

    @abstractmethod
    def satellite_weight(self, ell_idx: int) -> np.ndarray:
        """
        Per-halo weight for the satellite component (without HMF).

        For discrete tracers (CIB, galaxies) this includes the NFW
        Fourier transform ``u(k|M,z)``.  For continuous-field tracers
        (tSZ, lensing) this is the full halo profile.

        Parameters
        ----------
        ell_idx : int
            Index into the multipole grid.

        Returns
        -------
        w_s : ndarray, shape ``(n_fields, n_mass, n_z)``
        """

    @abstractmethod
    def window(self) -> np.ndarray:
        """
        Radial window function ``W(z)`` for Limber projection.

        Returns
        -------
        W : ndarray, shape ``(n_z,)`` or ``(n_fields, n_z)``
            For tracers whose window is frequency-independent (CIB, tSZ,
            galaxies), return shape ``(n_z,)``.
            For tracers whose window depends on the observed frequency
            (e.g. CO line intensity mapping, where each frequency channel
            probes a different redshift slice), return shape
            ``(n_fields, n_z)``.
        """


# ═══════════════════════════════════════════════════════════════════════════
# CIB tracer
# ═══════════════════════════════════════════════════════════════════════════

class CIBTracer(Tracer):
    """
    CIB tracer wrapping an existing :class:`CIBModel`.

    The per-halo emissivity weights are ``dj / hmf`` (central and
    satellite), already including ``(1+z) χ² snu / KC`` factors
    and colour/flux corrections.

    Window function: ``W(z) = a(z) = 1/(1+z)``.

    This is the un-normalised convention.  In the paper convention
    (Maniyar+2022 Eq. A16), ``W(ν,z) = a(z) × j̄(ν,z)`` and the halo
    weights are normalised by ``j̄``.  Both give identical ``C_ℓ``.

    Parameters
    ----------
    cib_model : CIBModel
        Initialised CIB model.
    """

    def __init__(self, cib_model: CIBModel) -> None:
        super().__init__(cib_model.hm, cib_model.mdef)
        self._cib = cib_model

        # Reuse pre-computed halo arrays
        self.hmf_arr = cib_model.hmf_arr
        self.bias_arr = cib_model.bias_arr
        self.nfw_arr = cib_model.nfw_arr

        # Per-halo emissivities (remove HMF factor)
        # dj_cen, dj_sub have shape (n_freq, n_mass, n_z) and include hmf
        self._j_cen = cib_model._dj_cen / self.hmf_arr  # (n_freq, n_mass, n_z)
        self._j_sub = cib_model._dj_sub / self.hmf_arr   # (n_freq, n_mass, n_z)

        # Colour × flux correction
        self._fcxcc = (cib_model.fc * cib_model.cc)  # (n_freq,)

    @property
    def n_fields(self) -> int:
        return self._cib.nfreq

    def central_weight(self, ell_idx: int) -> np.ndarray:
        return self._j_cen * self._fcxcc[:, None, None]

    def satellite_weight(self, ell_idx: int) -> np.ndarray:
        u_i = self.nfw_arr[:, ell_idx, :]  # (n_mass, n_z)
        return self._j_sub * u_i * self._fcxcc[:, None, None]

    def window(self) -> np.ndarray:
        """``W(z) = a(z) = 1/(1+z)``."""
        return 1.0 / (1.0 + self.z)


# ═══════════════════════════════════════════════════════════════════════════
# tSZ tracer
# ═══════════════════════════════════════════════════════════════════════════

class tSZTracer(Tracer):
    """
    tSZ (Compton-y) tracer wrapping an existing :class:`tSZModel`.

    The tSZ is a continuous field — the full ``y_ell`` profile goes
    into ``satellite_weight``; ``central_weight`` returns zeros.
    The spectral function ``f(ν)`` and unit conversions are folded
    into the weight.

    Window function: ``W(z) = χ(z)²`` so that
    ``(dχ/dz)/χ² × W² = dV_c/dz = dχ/dz × χ²``.

    Parameters
    ----------
    tsz_model : tSZModel
        Initialised tSZ model.
    """

    def __init__(self, tsz_model: tSZModel) -> None:
        super().__init__(tsz_model.hm, tsz_model.mdef)
        self._tsz = tsz_model

        # Reuse halo arrays
        self.hmf_arr = tsz_model.hmf_arr
        self.bias_arr = tsz_model.bias_arr

        # Spectral factor: f_nu * 10^6 * Kcmb_MJy
        if tsz_model.experiment == 'Planck':
            Kcmb_MJy = config.PLANCK['Kcmb_MJy']
        else:
            Kcmb_MJy = np.ones(tsz_model.nfreq)
        self._fnu_factor = tsz_model._f_nu * 1e6 * Kcmb_MJy  # (n_freq,)

    @property
    def n_fields(self) -> int:
        return self._tsz.nfreq

    def central_weight(self, ell_idx: int) -> np.ndarray:
        return np.zeros((self.n_fields, self.n_mass, self.n_z))

    def satellite_weight(self, ell_idx: int) -> np.ndarray:
        y_i = self._tsz._y_ell[ell_idx, :, :]  # (n_mass, n_z)
        return y_i[None, :, :] * self._fnu_factor[:, None, None]

    def window(self) -> np.ndarray:
        """``W(z) = χ(z)²``."""
        return self.hm._chi**2


# ═══════════════════════════════════════════════════════════════════════════
# Galaxy tracer
# ═══════════════════════════════════════════════════════════════════════════

class GalaxyTracer(Tracer):
    """
    Galaxy number density tracer using an HOD model.

    The galaxy field is the angular overdensity ``δ_g``.
    Per-halo weights are ``N_cen(M) / n̄(z)`` (central) and
    ``N_sat(M) × u_NFW(k,M,z) / n̄(z)`` (satellite).

    Window function: ``W(z) = φ(z) / (dχ/dz)``
    where ``φ(z) = (dN/dz) / N̄`` is the normalised selection function
    (Maniyar+2022, Eq. A17).

    Parameters
    ----------
    halo_model : HaloModel
        Shared halo model instance.
    hod : GalaxyHOD
        Galaxy HOD model.
    mdef : str
        Halo mass definition.
    centrals_nfw : bool
        If *True*, central galaxies follow the NFW profile (off-centre).
        If *False* (default), centrals are point-like (u=1 in Fourier
        space).
    """

    def __init__(
        self,
        halo_model,
        hod: GalaxyHOD,
        mdef: str = config.MDEF_GAL,
        centrals_nfw: bool = False,
    ) -> None:
        super().__init__(halo_model, mdef)
        self.hod = hod
        self.centrals_nfw = centrals_nfw

        # Pre-compute halo quantities
        print("Galaxy: pre-computing HMF, bias, NFW FT...")
        pre = halo_model.precompute(mdef)
        self.hmf_arr = pre['hmf']    # (n_mass, n_z)
        self.bias_arr = pre['bias']  # (n_mass, n_z)
        self.nfw_arr = pre['nfw']    # (n_mass, n_ell, n_z)

        # HOD occupation numbers (mass-dependent, z-independent for More+15)
        self._Ncen = hod.Ncen(self.mass)   # (n_mass,)
        self._Nsat = hod.Nsat(self.mass)   # (n_mass,)

        # Mean comoving number density at each z
        self._nbar = hod.nbar(self.mass, self.hmf_arr)  # (n_z,)

        # Galaxy selection function φ(z)
        self._phi = hod.window(self.z)  # (n_z,)

    @property
    def n_fields(self) -> int:
        return 1

    def central_weight(self, ell_idx: int) -> np.ndarray:
        # N_cen(M) / nbar(z), optionally × u_NFW
        w = self._Ncen[:, None] / self._nbar[None, :]  # (n_mass, n_z)
        if self.centrals_nfw:
            u_i = self.nfw_arr[:, ell_idx, :]  # (n_mass, n_z)
            w = w * u_i
        return w[None, :, :]  # (1, n_mass, n_z)

    def satellite_weight(self, ell_idx: int) -> np.ndarray:
        u_i = self.nfw_arr[:, ell_idx, :]  # (n_mass, n_z)
        w = self._Nsat[:, None] * u_i / self._nbar[None, :]  # (n_mass, n_z)
        return w[None, :, :]  # (1, n_mass, n_z)

    def window(self) -> np.ndarray:
        """``W(z) = φ(z) / (dχ/dz)`` (Maniyar+2022, Eq. A17)."""
        return self._phi / self.hm._dchi_dz
