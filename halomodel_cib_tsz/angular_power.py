"""
Generic angular cross-correlation power spectrum engine.

Computes 1-halo and 2-halo angular power spectra for any pair of
:class:`~halomodel_cib_tsz.tracers.Tracer` objects using the Limber
approximation:

.. math::

    C_\\ell^{AB} = \\int dz\\,\\frac{d\\chi/dz}{\\chi^2}\\,
    W_A(z)\\, W_B(z)\\; P^{AB}(k=(\\ell+0.5)/\\chi,\\, z)

where the 3D halo-model power spectrum decomposes as

.. math::

    P^{AB}_{1h} &= \\int d\\log_{10}M\\;
    \\frac{dn}{d\\log_{10}M}\\,
    \\bigl[c_A s_B + s_A c_B + s_A s_B\\bigr]

    P^{AB}_{2h} &= P_{\\rm lin}(k,z)\\,
    \\Bigl[\\int \\frac{dn}{d\\log_{10}M}\\, b\\, (c_A+s_A)\\Bigr]\\,
    \\Bigl[\\int \\frac{dn}{d\\log_{10}M}\\, b\\, (c_B+s_B)\\Bigr]

The 1-halo formula omits the central×central term (``c_A c_B``),
which contributes to shot noise rather than the clustering signal.
"""

from __future__ import annotations

import numpy as np

from .utils import simps
from .tracers import Tracer


class AngularCrossSpectrum:
    """
    Compute the angular cross-power spectrum between two tracers.

    Parameters
    ----------
    tracer_a, tracer_b : Tracer
        The two LSS tracers.  They must share the same
        :class:`~halomodel_cib_tsz.halo.HaloModel` instance (same
        grids).

    Raises
    ------
    ValueError
        If the tracers have incompatible grids.

    Notes
    -----
    For an auto-spectrum, pass the same tracer as both *tracer_a*
    and *tracer_b*.
    """

    def __init__(self, tracer_a: Tracer, tracer_b: Tracer) -> None:
        if tracer_a.hm is not tracer_b.hm:
            raise ValueError(
                "Both tracers must share the same HaloModel instance"
            )

        self.ta = tracer_a
        self.tb = tracer_b
        self.hm = tracer_a.hm

        self.n_a = tracer_a.n_fields
        self.n_b = tracer_b.n_fields

        # Cross between different multi-frequency tracers needs
        # symmetrization: C[νi,νj] = ⟨A(νi)B(νj)⟩ + ⟨B(νi)A(νj)⟩
        self._symmetrize = (
            tracer_a is not tracer_b and self.n_a == self.n_b
        )
        self.n_ell = tracer_a.n_ell
        self.n_mass = tracer_a.n_mass
        self.n_z = tracer_a.n_z

        # Common geometric factor: G(z) = dχ/dz / χ²
        self._G = self.hm._dchi_dz / self.hm._chi**2  # (n_z,)

        # Window functions — may be (n_z,) or (n_fields, n_z)
        W_a = tracer_a.window()
        W_b = tracer_b.window()

        # Expand frequency-independent windows to (n_fields, n_z)
        if W_a.ndim == 1:
            W_a = np.broadcast_to(W_a, (self.n_a, self.n_z))
        if W_b.ndim == 1:
            W_b = np.broadcast_to(W_b, (self.n_b, self.n_z))

        # Combined projection kernel: G(z) × W_A(z) × W_B(z)
        # Shape (n_a, n_b, n_z) to handle frequency-dependent windows
        self._proj = (self._G * W_a[:, None, :]
                      * W_b[None, :, :])  # (n_a, n_b, n_z)

    def cl_1h(self) -> np.ndarray:
        """
        1-halo angular cross-power spectrum.

        Returns
        -------
        Cl_1h : ndarray, shape ``(n_fields_A, n_fields_B, n_ell)``
        """
        Cl_1h = np.zeros((self.n_a, self.n_b, self.n_ell))

        z = self.hm.z
        log10m = self.hm.log10_mass
        proj = self._proj  # (n_a, n_b, n_z)

        # Use tracer_a's HMF (both should be consistent for the cross)
        hmf = self.ta.hmf_arr  # (n_mass, n_z)

        for i in range(self.n_ell):
            # Halo weights at this ell
            c_a = self.ta.central_weight(i)    # (n_a, n_mass, n_z)
            s_a = self.ta.satellite_weight(i)  # (n_a, n_mass, n_z)
            c_b = self.tb.central_weight(i)    # (n_b, n_mass, n_z)
            s_b = self.tb.satellite_weight(i)  # (n_b, n_mass, n_z)

            # 1-halo integrand: c_A*s_B + s_A*c_B + s_A*s_B
            # Vectorise over all field pairs (n_a, n_b, n_mass, n_z)
            cs = (c_a[:, None, :, :] * s_b[None, :, :, :]
                  + s_a[:, None, :, :] * c_b[None, :, :, :]
                  + s_a[:, None, :, :] * s_b[None, :, :, :])

            # Multiply by HMF and integrate over mass
            integrand = hmf * cs  # (n_a, n_b, n_mass, n_z)
            P_1h = simps(integrand, log10m, axis=2)  # (n_a, n_b, n_z)

            # Project along LOS and integrate over z
            Cl_1h[:, :, i] = simps(P_1h * proj, z, axis=-1)

        if self._symmetrize:
            Cl_1h = Cl_1h + Cl_1h.transpose(1, 0, 2)

        return Cl_1h

    def cl_2h(self) -> np.ndarray:
        """
        2-halo angular cross-power spectrum.

        Returns
        -------
        Cl_2h : ndarray, shape ``(n_fields_A, n_fields_B, n_ell)``
        """
        Cl_2h = np.zeros((self.n_a, self.n_b, self.n_ell))

        z = self.hm.z
        log10m = self.hm.log10_mass
        proj = self._proj  # (n_a, n_b, n_z)
        Pk = self.hm._Pk_limber  # (n_ell, n_z)

        hmf_a = self.ta.hmf_arr
        bias_a = self.ta.bias_arr
        hmf_b = self.tb.hmf_arr
        bias_b = self.tb.bias_arr

        for i in range(self.n_ell):
            # Total weights (central + satellite)
            w_a = (self.ta.central_weight(i)
                   + self.ta.satellite_weight(i))  # (n_a, n_mass, n_z)
            w_b = (self.tb.central_weight(i)
                   + self.tb.satellite_weight(i))  # (n_b, n_mass, n_z)

            # Bias-weighted mass integrals
            I_a = simps(
                hmf_a * bias_a * w_a, log10m, axis=1
            )  # (n_a, n_z)
            I_b = simps(
                hmf_b * bias_b * w_b, log10m, axis=1
            )  # (n_b, n_z)

            # P_2h = P_lin × I_A × I_B
            P_2h = (Pk[i, :] * I_a[:, None, :]
                    * I_b[None, :, :])  # (n_a, n_b, n_z)

            # Project and integrate over z
            Cl_2h[:, :, i] = simps(P_2h * proj, z, axis=-1)

        if self._symmetrize:
            Cl_2h = Cl_2h + Cl_2h.transpose(1, 0, 2)

        return Cl_2h

    def cl_total(self) -> np.ndarray:
        """
        Total (1-halo + 2-halo) angular cross-power spectrum.

        Returns
        -------
        Cl : ndarray, shape ``(n_fields_A, n_fields_B, n_ell)``
        """
        return self.cl_1h() + self.cl_2h()
