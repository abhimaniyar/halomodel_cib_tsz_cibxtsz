"""
Unified interface for computing CIB, tSZ, and CIB×tSZ power spectra.
"""

import numpy as np

from . import config
from .halo import HaloModel
from .sed import load_planck_seds, load_spire_seds, load_unfiltered_seds
from .cib import CIBModel
from .tsz import tSZModel
from .cross import CIBxTSZModel


def compute_spectra(components=('cib', 'tsz', 'cibxtsz'),
                    experiment='Planck',
                    cosmo='planck18',
                    cib_params=None,
                    tsz_B=config.TSZ_B,
                    mass_range=config.MASS_RANGE,
                    n_mass=config.N_MASS,
                    z_range=config.Z_RANGE,
                    n_z=config.N_Z,
                    ell_range=config.ELL_RANGE,
                    n_ell=config.N_ELL,
                    shot_noise=True):
    """
    Compute CIB, tSZ, and/or CIB×tSZ angular power spectra.

    Parameters
    ----------
    components : tuple of str
        Which spectra to compute: 'cib', 'tsz', 'cibxtsz'.
    experiment : str
        Experiment preset: 'Planck', 'Herschel-spire', or 'CCAT-p'.
    cosmo : str
        Colossus cosmology name.
    cib_params : dict, optional
        CIB model parameters {Meff, eta_max, sigma_Mh, tau}.
        If None, uses defaults for the appropriate mass definition.
    tsz_B : float
        tSZ hydrostatic mass bias factor.
    mass_range, n_mass, z_range, n_z, ell_range, n_ell : grid specs.
    shot_noise : bool
        Whether to add shot noise to CIB spectra.

    Returns
    -------
    result : dict
        Contains 'ell', 'freqs', and requested power spectra arrays.
        Each C_ell has shape (n_freq, n_freq, n_ell).
    """
    # Get experiment config
    if experiment == 'Planck':
        exp = config.PLANCK
    elif experiment == 'Herschel-spire':
        exp = config.HERSCHEL_SPIRE
    elif experiment == 'CCAT-p':
        exp = config.CCAT
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    freqs = exp['freq_cib']
    cc = exp['cc']
    fc = exp['fc']
    nfreq = len(freqs)

    # Build halo model
    hm = HaloModel(cosmo_name=cosmo,
                    mass_range=mass_range, n_mass=n_mass,
                    z_range=z_range, n_z=n_z,
                    ell_range=ell_range, n_ell=n_ell)

    # Load SEDs
    if experiment == 'Planck':
        snu = load_planck_seds(hm.z)
    elif experiment == 'Herschel-spire':
        snu = load_spire_seds(hm.z)
    else:
        snu = load_unfiltered_seds(freqs, hm.z)

    result = {
        'ell': hm.ell,
        'freqs': freqs,
    }

    # ── CIB ──────────────────────────────────────────────────────────
    if 'cib' in components:
        cib = CIBModel(hm, snu, freqs, cc, fc,
                       params=cib_params, mdef='200c')
        cl_1h = cib.cl_1h()
        cl_2h = cib.cl_2h()
        cl_tot = cl_1h + cl_2h

        if shot_noise:
            sn = _build_shot_noise_matrix(freqs, nfreq, hm.n_ell)
            cl_tot = cl_tot + sn

        result['cl_cib_1h'] = cl_1h
        result['cl_cib_2h'] = cl_2h
        result['cl_cib'] = cl_tot
        result['I_nu'] = cib.mean_intensity()

    # ── tSZ ──────────────────────────────────────────────────────────
    if 'tsz' in components:
        tsz = tSZModel(hm, freqs, experiment=experiment, B=tsz_B)
        cl_1h = tsz.cl_1h()
        cl_2h = tsz.cl_2h()
        result['cl_tsz_1h'] = cl_1h
        result['cl_tsz_2h'] = cl_2h
        result['cl_tsz'] = cl_1h + cl_2h

    # ── CIB×tSZ ──────────────────────────────────────────────────────
    if 'cibxtsz' in components:
        # Cross uses 500c for both
        cib_500 = CIBModel(hm, snu, freqs, cc, fc,
                           params=config.CIB_PARAMS_500C, mdef='500c')
        if 'tsz' not in components:
            tsz = tSZModel(hm, freqs, experiment=experiment, B=tsz_B)

        cross = CIBxTSZModel(cib_500, tsz)
        cl_1h = cross.cl_1h()
        cl_2h = cross.cl_2h()
        result['cl_cibxtsz_1h'] = cl_1h
        result['cl_cibxtsz_2h'] = cl_2h
        result['cl_cibxtsz'] = cl_1h + cl_2h

    return result


def _build_shot_noise_matrix(freqs, nfreq, n_ell):
    """Build shot noise array from config, shape (nfreq, nfreq, n_ell)."""
    sn = np.zeros((nfreq, nfreq, n_ell))
    for i in range(nfreq):
        for j in range(nfreq):
            key = (int(freqs[i]), int(freqs[j]))
            key_rev = (int(freqs[j]), int(freqs[i]))
            if key in config.SHOT_NOISE:
                sn[i, j, :] = config.SHOT_NOISE[key]
            elif key_rev in config.SHOT_NOISE:
                sn[i, j, :] = config.SHOT_NOISE[key_rev]
    return sn
