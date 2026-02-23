"""
CIB halo model: star formation rate, emissivities, power spectra, mean intensity.

Matches the physics of Cell_cib.py and Inu_cib.py from the original code.
"""

import numpy as np
from scipy import integrate

from . import config
from .utils import simps


class CIBModel:
    """
    CIB angular power spectrum and mean intensity model.

    Parameters
    ----------
    halo_model : HaloModel
        Pre-initialised halo model instance.
    snu : ndarray, shape (n_freq, n_z)
        Effective SED in Jy/L_sun at each frequency and redshift.
    freqs : list of float
        Observed frequencies in GHz.
    cc : ndarray
        Color corrections per frequency.
    fc : ndarray
        Flux calibration factors per frequency.
    params : dict, optional
        CIB model parameters: {Meff, eta_max, sigma_Mh, tau}.
    mdef : str
        Mass definition (default '200c').
    """

    def __init__(self, halo_model, snu, freqs, cc, fc,
                 params=None, mdef='200c'):
        self.hm = halo_model
        self.snu = snu              # (n_freq, n_z)
        self.freqs = freqs
        self.nfreq = len(freqs)
        self.cc = np.asarray(cc)
        self.fc = np.asarray(fc)
        self.mdef = mdef

        # Model parameters
        if params is None:
            params = config.CIB_PARAMS_200C if mdef == '200c' else config.CIB_PARAMS_500C
        self.Meff = params['Meff']
        self.eta_max = params['eta_max']
        self.sigma_Mh = params['sigma_Mh']
        self.tau = params['tau']

        # Grids from halo model
        self.mass = halo_model.mass
        self.z = halo_model.z
        self.ell = halo_model.ell
        self.log10_mass = halo_model.log10_mass
        self.n_mass = halo_model.n_mass
        self.n_z = halo_model.n_z
        self.n_ell = halo_model.n_ell

        # Redshift-dependent sigma width for M > Meff
        self.z_c = config.Z_C
        self.sig_z = np.array([max(self.z_c - zi, 0.0) for zi in self.z])
        self.sigpow = self.sigma_Mh - self.tau * self.sig_z  # (n_z,)

        # Pre-compute halo quantities
        print("CIB: pre-computing HMF, bias, NFW FT...")
        pre = halo_model.precompute(mdef)
        self.hmf_arr = pre['hmf']     # (n_mass, n_z)
        self.bias_arr = pre['bias']   # (n_mass, n_z)
        self.nfw_arr = pre['nfw']     # (n_mass, n_ell, n_z)

        # Pre-compute emissivities
        print("CIB: computing emissivities...")
        self._dj_cen = self._djc_dlogMh()     # (n_freq, n_mass, n_z)
        self._dj_sub = self._djsub_dlogMh()   # (n_freq, n_mass, n_z)

    # ── Star formation model ─────────────────────────────────────────────

    def eta(self, M):
        """
        SFR efficiency eta(M, z) — log-normal with asymmetric width.

        Parameters
        ----------
        M : array_like
            Halo mass in M_sun. Shape (n_M,).

        Returns
        -------
        eta : ndarray, shape (n_M, n_z)
        """
        M = np.atleast_1d(M)
        lnM = np.log(M)
        lnMeff = np.log(self.Meff)
        dlnM2 = (lnM - lnMeff)**2

        result = np.zeros((len(M), self.n_z))
        for i, m in enumerate(M):
            if m < self.Meff:
                sigma = self.sigma_Mh
            else:
                sigma = self.sigpow  # (n_z,) — varies with z
            result[i, :] = self.eta_max * np.exp(-dlnM2[i] / (2.0 * sigma**2))
        return result

    def bar(self, M):
        """
        Baryonic accretion rate (Fakhouri+2010).

        Returns
        -------
        BAR : ndarray, shape (n_M, n_z) in M_sun/yr.
        """
        M = np.atleast_1d(M)
        fb = self.hm.baryon_fraction(self.z)  # scalar for LCDM
        Ez = self.hm.E_z(self.z)  # (n_z,)
        a = 46.1 * (1.0 + 1.11 * self.z) * Ez  # (n_z,)
        b = (M / 1.0e12)**1.1  # (n_M,)
        return np.outer(b, a) * fb  # (n_M, n_z)

    def sfr(self, M):
        """
        Star formation rate SFR(M,z) = eta(M,z) * BAR(M,z).

        Returns
        -------
        sfr : ndarray, shape (n_M, n_z) in M_sun/yr.
        """
        return self.eta(M) * self.bar(M)

    # ── Emissivities ─────────────────────────────────────────────────────

    def _djc_dlogMh(self):
        """
        Central halo emissivity: dn/dlogM * SFR((1-fsub)*M) * (1+z) * chi^2 / KC * snu.

        Returns
        -------
        dj_c : ndarray, shape (n_freq, n_mass, n_z)
        """
        fsub = config.F_SUB
        chi = self.hm._chi  # (n_z,) Mpc

        sfr_cen = self.sfr(self.mass * (1.0 - fsub))  # (n_mass, n_z)
        rest = self.hmf_arr * sfr_cen * (1.0 + self.z) * chi**2 / config.KC
        # rest shape: (n_mass, n_z)

        dj_c = np.zeros((self.nfreq, self.n_mass, self.n_z))
        for f in range(self.nfreq):
            dj_c[f, :, :] = rest * self.snu[f, :]
        return dj_c

    def _djsub_dlogMh(self):
        """
        Subhalo emissivity with min(SFR_I, SFR_II) prescription.

        Returns
        -------
        dj_sub : ndarray, shape (n_freq, n_mass, n_z)
        """
        fsub = config.F_SUB
        chi = self.hm._chi  # (n_z,)

        dj_sub = np.zeros((self.nfreq, self.n_mass, self.n_z))

        for i in range(self.n_mass):
            Mh_eff = self.mass[i] * (1.0 - fsub)
            if Mh_eff <= 1e5:
                continue

            # Subhalo masses from 10^5 to Mh_eff
            log_ms = np.arange(5.0, np.log10(Mh_eff), 0.1)
            ms = 10**log_ms
            dlnmsub = log_ms[1] - log_ms[0] if len(log_ms) > 1 else 0.1

            # SFR prescription I: independent SFR of subhalo
            sfrI = self.sfr(ms)  # (n_ms, n_z)

            # SFR prescription II: proportional to parent
            sfr_parent = self.sfr(np.array([Mh_eff]))  # (1, n_z)
            sfrII = sfr_parent[0, :][None, :] * ms[:, None] / Mh_eff  # (n_ms, n_z)

            # Take minimum
            sfr_sub = np.minimum(sfrI, sfrII)  # (n_ms, n_z)

            # Subhalo mass function
            subhmf = self._subhalo_mf(self.mass[i], ms)  # (n_ms,)

            # Integrate over subhalo masses
            integrand = subhmf[:, None] * sfr_sub / config.KC  # (n_ms, n_z)
            intgn = integrate.simpson(integrand, dx=dlnmsub, axis=0)  # (n_z,)

            # Build emissivity
            for f in range(self.nfreq):
                dj_sub[f, i, :] = (self.snu[f, :] * self.hmf_arr[i, :] *
                                   (1.0 + self.z) * intgn * chi**2)

        return dj_sub

    @staticmethod
    def _subhalo_mf(Mh, ms):
        """
        Subhalo mass function dn/dlog10(ms) from Tinker & Wetzel (2010).

        Returns shape (n_ms,).
        """
        x = ms / Mh
        return 0.3 * x**(-0.7) * np.exp(-9.9 * x**2.5) * np.log(10)

    # ── Power spectra ────────────────────────────────────────────────────

    def cl_1h(self):
        """
        1-halo CIB power spectrum.

        Returns
        -------
        Cl_1h : ndarray, shape (n_freq, n_freq, n_ell)
        """
        Cl_1h = np.zeros((self.nfreq, self.nfreq, self.n_ell))
        dj_c = self._dj_cen    # (n_freq, n_mass, n_z)
        dj_s = self._dj_sub    # (n_freq, n_mass, n_z)
        u = self.nfw_arr        # (n_mass, n_ell, n_z)
        hmf = self.hmf_arr      # (n_mass, n_z)

        # Geometric factor
        dchi_dz = self.hm._dchi_dz  # (n_z,)
        chi = self.hm._chi          # (n_z,)
        geo = dchi_dz / (chi * (1.0 + self.z))**2  # (n_z,)

        fcxcc = self.fc * self.cc  # (n_freq,)

        for i in range(self.n_ell):
            u_i = u[:, i, :]  # (n_mass, n_z)
            for f in range(self.nfreq):
                # (n_freq, n_mass, n_z) for each term
                # Term structure: dj_c[f]*dj_s + dj_c*dj_s[f] + dj_s[f]*dj_s*u^2
                # all divided by hmf, integrated over mass, then over z with geo
                rest1 = (dj_c[f, :, :] * dj_s * u_i +
                         dj_c * dj_s[f, :, :] * u_i +
                         dj_s[f, :, :] * dj_s * u_i**2) / hmf
                # rest1: (n_freq, n_mass, n_z)

                # Integrate over mass
                intg_mh = simps(rest1, self.log10_mass, axis=1)  # (n_freq, n_z)

                # Integrate over redshift
                intg_z = simps(intg_mh * geo, self.z, axis=-1)  # (n_freq,)

                Cl_1h[f, :, i] = fcxcc[f] * intg_z * fcxcc

        return Cl_1h

    def cl_2h(self):
        """
        2-halo CIB power spectrum.

        Returns
        -------
        Cl_2h : ndarray, shape (n_freq, n_freq, n_ell)
        """
        Cl_2h = np.zeros((self.nfreq, self.nfreq, self.n_ell))

        # Bias-weighted emissivity J_nu: ∫ dlog10(M) (dj_c + dj_s * u) * b
        # Shape: (n_freq, n_z, n_ell)
        Jv = np.zeros((self.nfreq, self.n_z, self.n_ell))
        dj_c = self._dj_cen    # (n_freq, n_mass, n_z)
        dj_s = self._dj_sub    # (n_freq, n_mass, n_z)
        u = self.nfw_arr        # (n_mass, n_ell, n_z)
        b = self.bias_arr       # (n_mass, n_z)

        for i in range(self.n_ell):
            u_i = u[:, i, :]  # (n_mass, n_z)
            rest = (dj_c + dj_s * u_i) * b  # (n_freq, n_mass, n_z)
            Jv[:, :, i] = simps(rest, self.log10_mass, axis=1)  # (n_freq, n_z)

        # Geometric factor
        dchi_dz = self.hm._dchi_dz
        chi = self.hm._chi
        geo = dchi_dz / (chi * (1.0 + self.z))**2  # (n_z,)

        pk_geo = self.hm._Pk_limber * geo  # (n_ell, n_z)
        pkt = pk_geo.T  # (n_z, n_ell)

        fcxcc = self.fc * self.cc

        for f in range(self.nfreq):
            # Jv[f] * Jv * pkt → (n_freq, n_z, n_ell)
            rest1 = Jv * Jv[f, :, :] * pkt  # (n_freq, n_z, n_ell)
            intg_z = simps(rest1, self.z, axis=1)  # (n_freq, n_ell)
            Cl_2h[f, :, :] = fcxcc[f] * intg_z * fcxcc[:, None]

        return Cl_2h

    # ── Mean CIB intensity ───────────────────────────────────────────────

    def mean_intensity(self):
        """
        Mean CIB specific intensity <I_nu> in nW/m^2/sr.

        Returns
        -------
        I_nu : ndarray, shape (n_freq,)
        """
        dj_c = self._dj_cen    # (n_freq, n_mass, n_z)
        dj_s = self._dj_sub    # (n_freq, n_mass, n_z)

        # Total emissivity integrated over mass
        total = dj_c + dj_s  # (n_freq, n_mass, n_z)
        jnu = simps(total, self.log10_mass, axis=1)  # (n_freq, n_z)

        # dchi/dz * jnu / (1+z)
        dchi_dz = self.hm._dchi_dz  # (n_z,)
        integrand = dchi_dz * jnu / (1.0 + self.z)  # (n_freq, n_z)

        # Integrate over z
        result = self.cc * np.asarray(self.freqs) * simps(integrand, self.z, axis=-1)

        # Convert: Jy → nW/m^2/sr
        result *= config.ghz * config.nW / config.w_jy
        return result
