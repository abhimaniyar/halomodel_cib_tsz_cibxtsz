"""
Constants, default parameters, and experiment presets.
"""

import os
import numpy as np
import scipy.constants as con

# ── Physical constants (SI) ──────────────────────────────────────────────────

KC = 1.0e-10          # Kennicutt constant (Chabrier IMF) [M_sun / yr / (L_sun)]
T_CMB = 2.7255        # CMB temperature [K]
c_light = 2.99792458e5  # Speed of light [km/s]

m_e = con.electron_mass       # Electron mass [kg]
h_planck = con.h              # Planck constant [J·s]
k_B = con.k                   # Boltzmann constant [J/K]
sigma_T = con.physical_constants['Thomson cross section'][0]  # [m^2]

M_sun = 1.989e30      # Solar mass [kg]
Mpc_to_m = 3.0857e22  # Mpc to metres
L_sun = 3.828e26      # Solar luminosity [W]

eV_to_J = 1.602e-19   # eV to Joules
cm_to_m = 1e-2         # cm to metres
ghz = 1e9              # GHz to Hz
w_jy = 1e26            # W/m^2/Hz to Jy
nW = 1e9               # W to nW

# ── Data directory ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# ── Default grids ────────────────────────────────────────────────────────────

MASS_RANGE = (1e10, 1e15)
N_MASS = 100
Z_RANGE = (0.005, 7.0)
N_Z = 100
ELL_RANGE = (50, 11000)
N_ELL = 50

# ── Default model choices ────────────────────────────────────────────────────

COSMO_NAME = 'planck18'
HMF_MODEL = 'tinker08'
BIAS_MODEL = 'tinker10'
CONC_MODEL = 'duffy08'

MDEF_CIB = '200c'
MDEF_TSZ = '500c'

# ── CIB best-fit parameters ─────────────────────────────────────────────────

# 200c mass definition (CIB auto-spectrum)
CIB_PARAMS_200C = {
    'Meff': 5.34372e12,
    'eta_max': 0.5126,
    'sigma_Mh': 1.2483,
    'tau': 0.8255,
}

# 500c mass definition (CIB×tSZ cross)
CIB_PARAMS_500C = {
    'Meff': 6.96252e12,
    'eta_max': 0.4967,
    'sigma_Mh': 1.8074,
    'tau': 1.2017,
}

# ── tSZ parameters ──────────────────────────────────────────────────────────

TSZ_B = 1.5  # Hydrostatic mass bias (1-b)

# Arnaud+2010 generalized NFW pressure profile parameters
PRESSURE_PROFILE = {
    'P_0': 6.41,
    'c_500': 1.81,
    'gamma': 0.31,
    'alpha': 1.33,
    'beta': 4.13,
}

# ── CIB model constants ─────────────────────────────────────────────────────

Z_C = 1.5        # Redshift cutoff for sigma evolution
F_SUB = 0.134 * np.log(10)  # Subhalo mass fraction

# ── Experiment presets ───────────────────────────────────────────────────────

PLANCK = {
    'name': 'Planck',
    'freq_cib': [100., 143., 217., 353., 545., 857.],
    'cc': np.array([1.076, 1.017, 1.119, 1.097, 1.068, 0.995]),
    'fc': np.ones(6),
    # Bandpass-convolved tSZ spectral function values
    'f_nu_bandpassed': np.array([-4.031, -2.785, 0.187, 6.205, 14.455, 26.335]),
    # Kcmb to MJy/sr conversion factors
    'Kcmb_MJy': np.array([244.1, 371.74, 483.69, 287.45, 58.04, 2.27]),
}

HERSCHEL_SPIRE = {
    'name': 'Herschel-spire',
    'freq_cib': [600., 857., 1200.],
    'cc': np.array([0.974, 0.989, 0.988]),
    'fc': np.ones(3),
}

CCAT = {
    'name': 'CCAT-p',
    'freq_cib': [220., 280., 350., 410., 850.],
    'cc': np.ones(5),
    'fc': np.ones(5),
}

# ── Shot noise (Jy^2/sr) ────────────────────────────────────────────────────
# Auto-spectrum shot noise from best-fit parameter file.
# Only auto-spectra (diagonal) are stored; cross-spectra TBD.

SHOT_NOISE = {
    (217, 217): 14.5,
    (353, 353): 357.3,
    (545, 545): 2343.8,
    (857, 857): 7384.6,
}
