# halomodel_cib_tsz

Halo model code for computing CIB (Cosmic Infrared Background), tSZ (thermal Sunyaev-Zel'dovich), and CIB x tSZ angular power spectra and mean CIB intensity.

Based on the model from [Maniyar, Bethermin & Lagache (2021)](https://arxiv.org/abs/2006.16329). Original code: [github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz](https://github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz).

## What it computes

- **CIB auto-power spectra** C_l^{nu x nu'} (1-halo + 2-halo + shot noise)
- **tSZ auto-power spectra** C_l^{yy} (Arnaud+2010 pressure profile)
- **CIB x tSZ cross-power spectra** C_l^{CIB x tSZ}
- **Mean CIB specific intensity** \<I_nu\> in nW/m^2/sr

The four free CIB parameters are: M_eff (peak efficiency halo mass), eta_max (maximum SFR efficiency), sigma_Mh (log-normal width), and tau (late-time evolution slope).

## Installation

### Option 1: conda/mamba (recommended)

```bash
git clone <this-repo-url>
cd halomodel_cib_tsz

# Create the environment
conda env create -f environment.yml
conda activate halomodel

# Install astropy (needed for reading SED FITS files)
pip install astropy
```

### Option 2: pip into an existing environment

```bash
pip install numpy scipy matplotlib colossus astropy
```

Requires Python 3.11+.

### Verify the installation

```bash
conda activate halomodel
python -c "from halomodel_cib_tsz import compute_spectra; print('OK')"
```

## Quick start

### Python

```python
from halomodel_cib_tsz import compute_spectra

# Compute all spectra with default Planck settings
result = compute_spectra()

# Access results
ell = result['ell']              # multipole array
cl_cib = result['cl_cib']       # CIB total, shape (6, 6, 50)
cl_tsz = result['cl_tsz']       # tSZ total
cl_cross = result['cl_cibxtsz'] # CIB x tSZ total
I_nu = result['I_nu']           # mean CIB intensity (nW/m^2/sr)
```

### Command line

```bash
conda activate halomodel
python -m halomodel_cib_tsz.examples.driver
```

This computes all spectra for Planck frequencies and saves a plot to `power_spectra.png`.

### Jupyter notebook

See [`halomodel_cib_tsz/examples/quickstart.ipynb`](halomodel_cib_tsz/examples/quickstart.ipynb) for an interactive walkthrough with plots.

## Usage

### Selecting which spectra to compute

By default, `compute_spectra()` calculates CIB, tSZ, and CIB x tSZ. You can select only what you need using the `components` parameter:

```python
# CIB only (fastest -- skips tSZ and cross)
result = compute_spectra(components=('cib',))

# tSZ only
result = compute_spectra(components=('tsz',))

# CIB + tSZ, no cross-correlation
result = compute_spectra(components=('cib', 'tsz'))

# All three (default)
result = compute_spectra(components=('cib', 'tsz', 'cibxtsz'))
```

### Full parameter reference

```python
result = compute_spectra(
    components=('cib', 'tsz', 'cibxtsz'),  # which spectra to compute
    experiment='Planck',          # 'Planck', 'Herschel-spire', or 'CCAT-p'
    cosmo='planck18',             # colossus cosmology name
    cib_params=None,              # dict with {Meff, eta_max, sigma_Mh, tau}, or None for defaults
    tsz_B=1.5,                    # hydrostatic mass bias (M_true = M_500 / B)
    mass_range=(1e10, 1e15),      # halo mass range [M_sun]
    n_mass=100,                   # number of mass grid points
    z_range=(0.005, 7.0),         # redshift range
    n_z=100,                      # number of redshift points
    ell_range=(50, 11000),        # multipole range
    n_ell=50,                     # number of ell points
    shot_noise=True,              # include CIB shot noise
)
```

### Return values

The returned dictionary contains (only for requested components):

| Key | Shape | Description |
|-----|-------|-------------|
| `ell` | (n_ell,) | Multipole array |
| `freqs` | list | Frequency channels [GHz] |
| `cl_cib` | (n_freq, n_freq, n_ell) | CIB total (1h + 2h + shot noise) |
| `cl_cib_1h` | (n_freq, n_freq, n_ell) | CIB 1-halo term |
| `cl_cib_2h` | (n_freq, n_freq, n_ell) | CIB 2-halo term |
| `I_nu` | (n_freq,) | Mean CIB intensity [nW/m^2/sr] |
| `cl_tsz` | (n_freq, n_freq, n_ell) | tSZ total |
| `cl_tsz_1h` | (n_freq, n_freq, n_ell) | tSZ 1-halo |
| `cl_tsz_2h` | (n_freq, n_freq, n_ell) | tSZ 2-halo |
| `cl_cibxtsz` | (n_freq, n_freq, n_ell) | CIB x tSZ total |
| `cl_cibxtsz_1h` | (n_freq, n_freq, n_ell) | CIB x tSZ 1-halo |
| `cl_cibxtsz_2h` | (n_freq, n_freq, n_ell) | CIB x tSZ 2-halo |

For Planck, the frequency indices are: 0=100, 1=143, 2=217, 3=353, 4=545, 5=857 GHz.

### Supported experiments

| Experiment | Frequencies [GHz] | SEDs |
|-----------|-------------------|------|
| `'Planck'` | 100, 143, 217, 353, 545, 857 | Bandpass-convolved (FITS) |
| `'Herschel-spire'` | 600, 857, 1200 | Bandpass-convolved (FITS) |
| `'CCAT-p'` | 220, 280, 350, 410, 850 | Unfiltered Bethermin+2012 |

### Custom CIB parameters

```python
my_params = {
    'Meff': 5e12,        # peak efficiency halo mass [M_sun]
    'eta_max': 0.5,      # maximum SFR efficiency
    'sigma_Mh': 1.25,    # log-normal width
    'tau': 0.8,          # late-time evolution slope
}

result = compute_spectra(components=('cib',), cib_params=my_params)
```

### Using model classes directly

For finer control, use the individual model classes:

```python
from halomodel_cib_tsz.halo import HaloModel
from halomodel_cib_tsz.sed import load_planck_seds
from halomodel_cib_tsz.cib import CIBModel
from halomodel_cib_tsz import config

# Build halo model with custom grids
hm = HaloModel(cosmo_name='planck18', n_mass=100, n_z=100, n_ell=80)

# Load SEDs and create CIB model
snu = load_planck_seds(hm.z)
cib = CIBModel(hm, snu,
               freqs=config.PLANCK['freq_cib'],
               cc=config.PLANCK['cc'],
               fc=config.PLANCK['fc'],
               mdef='200c')

# Compute spectra
cl_1h = cib.cl_1h()       # (6, 6, 80)
cl_2h = cib.cl_2h()       # (6, 6, 80)
I_nu = cib.mean_intensity()  # (6,)
```

### Matching the original paper

To reproduce the results in Maniyar+2021:

```python
result = compute_spectra(
    components=('cib', 'tsz', 'cibxtsz'),
    cosmo='planck15',          # original used Planck 2015
    experiment='Planck',
)
```

For even closer agreement, initialise `HaloModel` with `hmf_model='tinker08_original'`, which uses the original cubic-spline interpolation of Tinker08 parameters. With exact original grids, this matches the paper to ~3% (the residual is from Eisenstein-Hu vs CAMB for the matter power spectrum).

## Physics

The model follows [Maniyar, Bethermin & Lagache (2021)](https://arxiv.org/abs/2006.16329):

- **Cosmology**: Planck 2018 (default) via [colossus](https://bdiemer.bitbucket.io/colossus/)
- **Halo mass function**: Tinker+2008 (switchable)
- **Halo bias**: Tinker+2010 (switchable)
- **Concentration-mass**: Duffy+2008 (switchable)
- **P(k,z)**: Eisenstein-Hu transfer function via colossus
- **SFR model**: Log-normal efficiency x baryonic accretion rate (Fakhouri+2010)
- **Subhalo contribution**: Tinker & Wetzel (2010) subhalo MF with min(SFR_I, SFR_II)
- **tSZ pressure profile**: Arnaud+2010 generalised NFW with pre-tabulated y_ell kernel
- **Mass definitions**: CIB uses M_200c, tSZ uses M_500c, cross uses M_500c for both
- **Units**: Physical units in the public API (M_sun, Mpc, GHz). All h-unit conversions are handled internally.

## Package structure

```
halomodel_cib_tsz/
  __init__.py  -- Public API
  config.py    -- Constants, defaults, experiment presets, shot noise
  utils.py     -- Integration helpers
  halo.py      -- HaloModel: colossus wrapper (HMF, bias, NFW FT, P(k), distances)
  sed.py       -- SED loading (Planck/SPIRE FITS, Bethermin tables), tSZ spectral fn
  cib.py       -- CIBModel: SFR, emissivities, 1h/2h power spectra, mean intensity
  tsz.py       -- tSZModel: pressure profile, y_ell, 1h/2h power spectra
  cross.py     -- CIBxTSZModel: CIB x tSZ cross-correlation
  spectra.py   -- compute_spectra() unified interface
  data/        -- SED files, y_ell kernel, best-fit parameters
  examples/
    driver.py        -- Example script with plotting
    quickstart.ipynb  -- Jupyter notebook tutorial
```

## Citation

If you use this code, please cite:

```bibtex
@article{Maniyar2021,
    author = {Maniyar, Abhishek S. and B{\'e}thermin, Matthieu and Lagache, Guilaine},
    title = {Star formation history, infrared luminosity functions and dust mass functions
             from far-infrared/sub-millimetre galaxy surveys using a halo model framework},
    journal = {Astronomy \& Astrophysics},
    year = {2021},
    volume = {645},
    pages = {A40},
    doi = {10.1051/0004-6361/202038790},
    eprint = {2006.16329},
    archivePrefix = {arXiv},
}
```

## License

See the original repository: https://github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz
