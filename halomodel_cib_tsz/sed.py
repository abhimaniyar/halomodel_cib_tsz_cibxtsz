"""
SED loading and tSZ spectral function.

Handles:
- Planck HFI bandpass-filtered SEDs (FITS)
- Herschel SPIRE bandpass-filtered SEDs (FITS)
- Unfiltered Béthermin+2015 SED tables (TXT)
- tSZ spectral function g(x)
"""

import os
import re
import glob

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from astropy.io import fits

from . import config


def load_planck_seds(z_grid):
    """
    Load bandpass-filtered Planck HFI SEDs and interpolate to z_grid.

    Parameters
    ----------
    z_grid : array_like
        Redshift grid to interpolate onto.

    Returns
    -------
    snu : ndarray, shape (6, n_z)
        Effective SED in Jy/L_sun for 6 Planck channels
        (100, 143, 217, 353, 545, 857 GHz). The 7th channel (3000/IRAS)
        is dropped.
    """
    fpath = os.path.join(config.DATA_DIR, 'filtered_snu_planck.fits')
    with fits.open(fpath) as hdulist:
        snu_eff = hdulist[0].data   # (7, 210) — 7 freqs × 210 redshifts
        redshifts = hdulist[1].data  # (210,)

    # Interpolate each frequency to z_grid
    f_interp = interp1d(redshifts, snu_eff, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    snu = f_interp(z_grid)

    # Drop 7th frequency (IRAS 3000 GHz)
    return snu[:6, :]


def load_spire_seds(z_grid):
    """
    Load bandpass-filtered Herschel SPIRE SEDs and interpolate to z_grid.

    Parameters
    ----------
    z_grid : array_like
        Redshift grid.

    Returns
    -------
    snu : ndarray, shape (3, n_z)
        Effective SED in Jy/L_sun for 3 SPIRE channels (600, 857, 1200 GHz).
    """
    fpath = os.path.join(config.DATA_DIR, 'filtered_snu_spire.fits')
    with fits.open(fpath) as hdulist:
        snu_eff = hdulist[0].data   # (3, 210)
        redshifts = hdulist[1].data  # (210,)

    f_interp = interp1d(redshifts, snu_eff, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    return f_interp(z_grid)


def load_unfiltered_seds(freqs, z_grid):
    """
    Load Béthermin+2015 unfiltered SED tables and evaluate at (freqs, z_grid).

    Each table gives S_nu(lambda) at a specific redshift. We build a 2D
    interpolator in (freq_GHz, z) and evaluate at the requested frequencies.

    Parameters
    ----------
    freqs : array_like
        Observed frequencies in GHz.
    z_grid : array_like
        Redshift grid.

    Returns
    -------
    snu : ndarray, shape (n_freq, n_z)
        Effective SED in Jy/L_sun, normalised to L_IR.
    """
    sed_dir = os.path.join(config.DATA_DIR, 'TXT_TABLES_2015')
    file_list = glob.glob(os.path.join(sed_dir, 'EffectiveSED_B15_z*.txt'))

    # Extract redshifts and sort numerically
    z_file = []
    for f in file_list:
        m = re.search(r'z(\d+\.?\d*)', os.path.basename(f))
        if m:
            z_file.append(float(m.group(1)))
    order = np.argsort(z_file)
    z_file = np.array(z_file)[order]
    file_list = np.array(file_list)[order]

    # Read wavelengths from first file (all files share the same wavelength grid)
    data0 = np.loadtxt(file_list[0])
    wavelengths = data0[:, 0]  # microns
    n_wav = len(wavelengths)

    # Frequency in GHz: c [km/s] / wavelength [µm] × numerical factor
    # c_light = 2.998e5 km/s, wavelength in µm → freq in GHz:
    # freq = c_light [km/s] / wavelength [µm] * (1e3 [m/km] * 1e6 [µm/m]) / 1e9 [Hz/GHz]
    # = c_light / wavelength * 1.0  (the factors cancel to 1)
    freq = config.c_light / wavelengths  # GHz

    # Frequency in Hz for L_IR integration
    freq_hz = freq * config.ghz  # Hz

    # Read all SED tables
    snu_all = np.zeros((n_wav, len(z_file)))
    for i, fname in enumerate(file_list):
        snu_all[:, i] = np.loadtxt(fname)[:, 1]

    # Compute L_IR for normalisation
    freq_rest = freq_hz[:, None] * (1.0 + z_file[None, :])  # (n_wav, n_z)
    L_IR_vals = _L_IR(snu_all, freq_rest, z_file)

    # Normalise: snu * L_sun / L_IR
    for i in range(len(z_file)):
        snu_all[:, i] *= config.L_sun / L_IR_vals[i]

    # Reverse to increasing frequency order
    freq = freq[::-1]
    snu_all = snu_all[::-1, :]

    # 2D interpolation in (freq, z)
    interp2d = RectBivariateSpline(freq, z_file, snu_all)
    return interp2d(np.asarray(freqs), z_grid)


def _L_IR(snu_eff, freq_rest, redshifts):
    """
    Compute infrared luminosity for SED normalisation.

    Integrates from 8µm (3.747e13 Hz) to 1000µm (2.998e11 Hz) in rest frame.

    Parameters
    ----------
    snu_eff : ndarray, shape (n_wav, n_z)
        SED values in Jy/L_sun (unnormalised).
    freq_rest : ndarray, shape (n_wav, n_z)
        Rest-frame frequencies in Hz.
    redshifts : array_like
        Redshifts corresponding to each SED table.

    Returns
    -------
    L_IR : ndarray, shape (n_z,)
        Infrared luminosity for each redshift.
    """
    from colossus.cosmology import cosmology
    cosmo = cosmology.getCurrent()
    h = cosmo.H0 / 100.0

    fmin = 2.99792458e11   # 1000 µm in Hz
    fmax = 3.7474057250e13  # 8 µm in Hz
    n_int = 10000
    fint = np.linspace(np.log10(fmin), np.log10(fmax), n_int)

    L_IR = np.zeros(len(redshifts))
    for i in range(len(redshifts)):
        z = redshifts[i]
        # Luminosity distance in Mpc (physical)
        chi = cosmo.comovingDistance(0.0, z) / h  # Mpc
        d_L = chi * (1.0 + z)  # Mpc
        d_L_m = d_L * config.Mpc_to_m  # metres

        L_feq = snu_eff[:, i] * 4.0 * np.pi * d_L_m**2 / (config.w_jy * (1.0 + z))
        freq_sorted = np.sort(freq_rest[:, i])
        L_sorted = L_feq[::-1]  # reverse to match sorted freq
        Lint = np.interp(fint, np.log10(freq_sorted), L_sorted)
        dfeq = 10**fint
        L_IR[i] = np.trapz(Lint, dfeq)

    return L_IR


def tsz_spectral_fn(nu_ghz, experiment='Planck'):
    """
    tSZ spectral function g(x) = x * coth(x/2) - 4.

    For Planck, returns pre-computed bandpass-convolved values.

    Parameters
    ----------
    nu_ghz : array_like
        Frequencies in GHz.
    experiment : str
        Experiment name. 'Planck' uses tabulated bandpassed values.

    Returns
    -------
    g_nu : ndarray
        tSZ spectral function values.
    """
    if experiment == 'Planck':
        return config.PLANCK['f_nu_bandpassed']

    nu_hz = np.asarray(nu_ghz) * config.ghz
    x = config.h_planck * nu_hz / (config.k_B * config.T_CMB)
    return x * (np.exp(x) + 1.0) / (np.exp(x) - 1.0) - 4.0
