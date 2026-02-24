"""
Unified interface for computing CIB, tSZ, and CIB×tSZ power spectra.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import config
from .halo import HaloModel
from .sed import load_planck_seds, load_spire_seds, load_unfiltered_seds
from .cib import CIBModel
from .tsz import tSZModel
from .cross import CIBxTSZModel

_VALID_COMPONENTS = {'cib', 'tsz', 'cibxtsz'}
_VALID_EXPERIMENTS = {'Planck', 'Herschel-spire', 'CCAT-p'}


def compute_spectra(
    components: tuple[str, ...] = ('cib', 'tsz', 'cibxtsz'),
    experiment: str = 'Planck',
    cosmo: str = 'planck18',
    cib_params: Optional[dict] = None,
    tsz_B: float = config.TSZ_B,
    mass_range: tuple[float, float] = config.MASS_RANGE,
    n_mass: int = config.N_MASS,
    z_range: tuple[float, float] = config.Z_RANGE,
    n_z: int = config.N_Z,
    ell_range: tuple[float, float] = config.ELL_RANGE,
    n_ell: int = config.N_ELL,
    shot_noise: bool = True,
) -> dict:
    """
    Compute CIB, tSZ, and/or CIB×tSZ angular power spectra.

    Parameters
    ----------
    components : tuple of str
        Which spectra to compute. Valid values: ``'cib'``, ``'tsz'``,
        ``'cibxtsz'``.
    experiment : str
        Experiment preset: ``'Planck'``, ``'Herschel-spire'``, or
        ``'CCAT-p'``.
    cosmo : str
        Colossus cosmology name (e.g. ``'planck18'``).
    cib_params : dict, optional
        CIB model parameters ``{Meff, eta_max, sigma_Mh, tau}``.
        If *None*, uses defaults for the appropriate mass definition.
    tsz_B : float
        tSZ hydrostatic mass bias factor.
    mass_range : tuple of float
        ``(M_min, M_max)`` in M_sun.
    n_mass : int
        Number of log-spaced mass grid points.
    z_range : tuple of float
        ``(z_min, z_max)``.
    n_z : int
        Number of redshift grid points.
    ell_range : tuple of float
        ``(ell_min, ell_max)``.
    n_ell : int
        Number of log-spaced multipole points.
    shot_noise : bool
        Whether to add shot noise to CIB spectra.

    Returns
    -------
    result : dict
        Contains ``'ell'``, ``'freqs'``, and requested power spectra arrays.
        Each C_ell array has shape ``(n_freq, n_freq, n_ell)``.

    Raises
    ------
    ValueError
        If any argument is out of range or invalid.
    """
    # ── Validate inputs ───────────────────────────────────────────────
    components = tuple(c.lower() for c in components)
    invalid = set(components) - _VALID_COMPONENTS
    if invalid:
        raise ValueError(
            f"Invalid component(s): {invalid}. "
            f"Valid options: {_VALID_COMPONENTS}"
        )

    if experiment not in _VALID_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Valid options: {_VALID_EXPERIMENTS}"
        )

    if cib_params is not None:
        required = {'Meff', 'eta_max', 'sigma_Mh', 'tau'}
        missing = required - set(cib_params.keys())
        if missing:
            raise ValueError(
                f"cib_params missing required keys: {missing}. "
                f"Expected: {required}"
            )

    if mass_range[0] >= mass_range[1]:
        raise ValueError(
            f"mass_range[0] ({mass_range[0]}) must be < mass_range[1] ({mass_range[1]})"
        )
    if mass_range[0] <= 0:
        raise ValueError(f"mass_range[0] must be > 0, got {mass_range[0]}")

    if z_range[0] >= z_range[1]:
        raise ValueError(
            f"z_range[0] ({z_range[0]}) must be < z_range[1] ({z_range[1]})"
        )
    if z_range[0] < 0:
        raise ValueError(f"z_range[0] must be >= 0, got {z_range[0]}")

    if ell_range[0] >= ell_range[1]:
        raise ValueError(
            f"ell_range[0] ({ell_range[0]}) must be < ell_range[1] ({ell_range[1]})"
        )
    if ell_range[0] < 1:
        raise ValueError(f"ell_range[0] must be >= 1, got {ell_range[0]}")

    for name, val in [('n_mass', n_mass), ('n_z', n_z), ('n_ell', n_ell)]:
        if val < 5:
            raise ValueError(f"{name} must be >= 5, got {val}")

    if tsz_B <= 0:
        raise ValueError(f"tsz_B must be > 0, got {tsz_B}")

    # ── Get experiment config ─────────────────────────────────────────
    if experiment == 'Planck':
        exp = config.PLANCK
    elif experiment == 'Herschel-spire':
        exp = config.HERSCHEL_SPIRE
    elif experiment == 'CCAT-p':
        exp = config.CCAT

    freqs = exp['freq_cib']
    cc = exp['cc']
    fc = exp['fc']
    nfreq = len(freqs)

    # ── Build halo model ──────────────────────────────────────────────
    hm = HaloModel(cosmo_name=cosmo,
                    mass_range=mass_range, n_mass=n_mass,
                    z_range=z_range, n_z=n_z,
                    ell_range=ell_range, n_ell=n_ell)

    # ── Load SEDs ─────────────────────────────────────────────────────
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

    # ── CIB ───────────────────────────────────────────────────────────
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

    # ── tSZ ───────────────────────────────────────────────────────────
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


def _build_shot_noise_matrix(
    freqs: list[float], nfreq: int, n_ell: int
) -> np.ndarray:
    """Build shot noise array from config, shape ``(nfreq, nfreq, n_ell)``."""
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
