"""
CIB × tSZ cross-correlation power spectrum.

Both CIB and tSZ use M_500c mass definition for the cross-spectrum.
The CIB best-fit parameters differ from the CIB-only (200c) case.

Matches the physics of Cell_CIBxtSZ.py from the original code.
"""

from __future__ import annotations

import numpy as np

from . import config
from .utils import simps


class CIBxTSZModel:
    """
    CIB × tSZ cross-correlation model.

    Parameters
    ----------
    cib_model : CIBModel
        CIB model initialised with ``mdef='500c'`` and 500c best-fit params.
    tsz_model : tSZModel
        tSZ model initialised with ``mdef='500c'``.

    Raises
    ------
    ValueError
        If models use different halo model instances or incompatible grids.
    """

    def __init__(self, cib_model, tsz_model) -> None:
        if cib_model.hm is not tsz_model.hm:
            raise ValueError(
                "CIB and tSZ models must share the same HaloModel instance"
            )

        self.cib = cib_model
        self.tsz = tsz_model

        self.nfreq = cib_model.nfreq
        self.z = cib_model.z
        self.mass = cib_model.mass
        self.ell = cib_model.ell
        self.log10_mass = cib_model.log10_mass
        self.n_mass = cib_model.n_mass
        self.n_z = cib_model.n_z
        self.n_ell = cib_model.n_ell
        self.hm = cib_model.hm

        # Normalised CIB emissivities: divide out HMF * (1+z) * chi^2
        # to get the "prime" emissivities as in original code
        cosm = (1.0 + self.z) * self.hm._chi**2  # (n_z,)
        hmf_tsz = self.tsz.hmf_arr  # (n_mass, n_z)
        denom = hmf_tsz * cosm  # (n_mass, n_z)

        self._dj_cen_prime = cib_model._dj_cen / denom  # (n_freq, n_mass, n_z)
        self._dj_sub_prime = cib_model._dj_sub / denom   # (n_freq, n_mass, n_z)

    def cl_1h(self) -> np.ndarray:
        """
        1-halo CIB × tSZ cross power spectrum.

        Returns
        -------
        Cl_1h : ndarray, shape ``(n_freq, n_freq, n_ell)``
            In Jy^2/sr.
        """
        if self.cib.hm._cosmo.H0 and self.tsz.experiment == 'Planck':
            Kcmb_MJy = config.PLANCK['Kcmb_MJy']
        else:
            Kcmb_MJy = np.ones(self.nfreq)

        Cl_1h = np.zeros((self.nfreq, self.nfreq, self.n_ell))

        u_nfw = self.cib.nfw_arr     # (n_mass, n_ell, n_z) — CIB NFW at 500c
        dj_c = self._dj_cen_prime    # (n_freq, n_mass, n_z)
        dj_s = self._dj_sub_prime    # (n_freq, n_mass, n_z)
        cc = self.cib.cc             # (n_freq,)
        f_v = self.tsz.f_nu() * 1e6 * Kcmb_MJy  # (n_freq,)
        y_ell = self.tsz._y_ell      # (n_ell, n_mass, n_z)
        hmf = self.tsz.hmf_arr       # (n_mass, n_z)

        # dVc/dz geometric factor
        dVc_dz = (config.c_light * self.hm._chi**2 /
                  (self.hm.H0 * self.hm.E_z(self.z)))  # (n_z,)

        for i in range(self.n_ell):
            u_i = u_nfw[:, i, :]  # (n_mass, n_z)
            for f in range(self.nfreq):
                a = y_ell[i, :, :] * (
                    (dj_c + dj_s * u_i) * cc[:, None, None] * f_v[f] +
                    (dj_c[f, :, :] + dj_s[f, :, :] * u_i) * cc[f] * f_v[:, None, None]
                ) * hmf  # (n_freq, n_mass, n_z)

                # Integrate over mass
                intgn_mh = simps(a, self.log10_mass, axis=1)  # (n_freq, n_z)

                # Integrate over z
                intgn_z = simps(dVc_dz * intgn_mh, self.z, axis=-1)  # (n_freq,)

                Cl_1h[f, :, i] = intgn_z

        return Cl_1h

    def cl_2h(self) -> np.ndarray:
        """
        2-halo CIB × tSZ cross power spectrum.

        Returns
        -------
        Cl_2h : ndarray, shape ``(n_freq, n_freq, n_ell)``
            In Jy^2/sr.
        """
        if self.tsz.experiment == 'Planck':
            Kcmb_MJy = config.PLANCK['Kcmb_MJy']
        else:
            Kcmb_MJy = np.ones(self.nfreq)

        Cl_2h = np.zeros((self.nfreq, self.nfreq, self.n_ell))

        u_nfw = self.cib.nfw_arr     # (n_mass, n_ell, n_z)
        dj_c = self._dj_cen_prime    # (n_freq, n_mass, n_z)
        dj_s = self._dj_sub_prime    # (n_freq, n_mass, n_z)
        cc = self.cib.cc
        f_v = self.tsz.f_nu() * 1e6 * Kcmb_MJy
        y_ell = self.tsz._y_ell      # (n_ell, n_mass, n_z)
        hmf = self.tsz.hmf_arr
        b = self.tsz.bias_arr
        bhmf = b * hmf

        dVc_dz = (config.c_light * self.hm._chi**2 /
                  (self.hm.H0 * self.hm.E_z(self.z)))
        Pk = self.hm._Pk_limber  # (n_ell, n_z)

        for i in range(self.n_ell):
            u_i = u_nfw[:, i, :]  # (n_mass, n_z)

            # tSZ integral: ∫ dlog10(M) y_ell * b * hmf
            res1 = y_ell[i, :, :] * bhmf  # (n_mass, n_z)
            intgn_mh1 = simps(res1, self.log10_mass, axis=0)  # (n_z,)

            for f in range(self.nfreq):
                # CIB integral
                res2 = (
                    (dj_c + dj_s * u_i) * f_v[f] +
                    (dj_c[f, :, :] + dj_s[f, :, :] * u_i) * f_v[:, None, None]
                ) * bhmf  # (n_freq, n_mass, n_z)

                intgn_mh2 = simps(res2, self.log10_mass, axis=1)  # (n_freq, n_z)

                # Combine with P(k) and geometric factor
                integrand = dVc_dz * Pk[i, :] * intgn_mh1 * intgn_mh2  # (n_freq, n_z)
                intgn_z = simps(integrand, self.z, axis=-1)  # (n_freq,)

                Cl_2h[f, :, i] = intgn_z

        return Cl_2h
