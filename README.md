# halomodel_cib_tsz

Halo model code for computing CIB (Cosmic Infrared Background), tSZ (thermal Sunyaev-Zel'dovich), and CIB x tSZ angular power spectra and mean CIB intensity.

Based on the model from [Maniyar, Bethermin & Lagache (2021)](https://arxiv.org/abs/2006.16329). Original code: [github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz](https://github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz).

## What it computes

- **CIB auto-power spectra** C_l^{nu x nu'} (1-halo + 2-halo + shot noise)
- **tSZ auto-power spectra** C_l^{yy} (Arnaud+2010 pressure profile)
- **CIB x tSZ cross-power spectra** C_l^{CIB x tSZ}
- **Galaxy auto-power spectrum** C_l^{gg} (HOD model, More+2015)
- **CIB x galaxy cross-power spectra** C_l^{CIB x gal}
- **tSZ x galaxy cross-power spectra** C_l^{tSZ x gal}
- **Mean CIB specific intensity** nu*\<I_nu\> in nW/m^2/sr — `CIBModel.mean_intensity()`
- **Cosmic SFR density** rho_SFR(z) in M_sun/yr/Mpc^3 — `CIBModel.sfrd()`
  (integrates SFR(M, z) over the halo mass function for centrals + integrated subhalos)

The four free CIB parameters are: M_eff (peak efficiency halo mass), eta_max (maximum SFR efficiency), sigma_Mh (log-normal width), and tau (late-time evolution slope).

All 2-halo terms include the **non-linear halo bias correction β^NL** of [Mead & Verde (2021)](https://arxiv.org/abs/2011.08858), as adopted for the CIB by [Jego et al. (2026)](https://arxiv.org/abs/2609.22057). It is on by default; see [Non-linear halo bias](#non-linear-halo-bias-βnl).

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

# Galaxy auto-spectrum
result = compute_spectra(components=('gal',), galaxy_survey='CMASS')

# Galaxy cross-correlations
result = compute_spectra(
    components=('cib', 'tsz', 'gal', 'cibxgal', 'tszxgal'),
    galaxy_survey='CMASS',
)
```

Available components: `'cib'`, `'tsz'`, `'cibxtsz'`, `'gal'`, `'cibxgal'`, `'tszxgal'`.

### Full parameter reference

```python
result = compute_spectra(
    components=('cib', 'tsz', 'cibxtsz'),  # which spectra to compute
    experiment='Planck',          # 'Planck', 'Herschel-spire', or 'CCAT-p'
    cosmo='planck18',             # colossus cosmology name
    cib_params=None,              # dict with {Meff, eta_max, sigma_Mh, tau}, or None for defaults
    tsz_B=1.5,                    # hydrostatic mass bias (M_true = M_500 / B)
    galaxy_survey='CMASS',        # 'CMASS', 'DESI_LRG', or 'DESI_ELG'
    hod_params=None,              # HOD parameters, or None for survey defaults
    mass_range=(1e10, 1e15),      # halo mass range [M_sun]
    n_mass=100,                   # number of mass grid points
    z_range=(0.005, 7.0),         # redshift range
    n_z=100,                      # number of redshift points
    ell_range=(50, 11000),        # multipole range
    n_ell=50,                     # number of ell points
    shot_noise=True,              # include CIB and galaxy shot noise
    beta_nl=True,                 # non-linear halo bias in all 2-halo terms
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
| `cl_gal` | (n_ell,) | Galaxy auto total (1h + 2h + shot) |
| `cl_gal_1h` | (n_ell,) | Galaxy auto 1-halo |
| `cl_gal_2h` | (n_ell,) | Galaxy auto 2-halo |
| `cl_cibxgal` | (n_freq, n_ell) | CIB x galaxy total |
| `cl_cibxgal_1h` | (n_freq, n_ell) | CIB x galaxy 1-halo |
| `cl_cibxgal_2h` | (n_freq, n_ell) | CIB x galaxy 2-halo |
| `cl_tszxgal` | (n_freq, n_ell) | tSZ x galaxy total |
| `cl_tszxgal_1h` | (n_freq, n_ell) | tSZ x galaxy 1-halo |
| `cl_tszxgal_2h` | (n_freq, n_ell) | tSZ x galaxy 2-halo |

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
I_nu = cib.mean_intensity()  # (6,) nu*<I_nu> in nW/m^2/sr
sfrd = cib.sfrd()         # (n_z,) cosmic SFRD in M_sun/yr/Mpc^3
```

### SFRD and mean CIB intensity

The CIB model exposes two integral observables that are useful on their own
(e.g. for fitting the four CIB parameters to SFRD measurements or galaxy
number-count CIB levels without computing any C_l):

```python
from halomodel_cib_tsz.halo import HaloModel
from halomodel_cib_tsz.sed import load_unfiltered_seds
from halomodel_cib_tsz.cib import CIBModel
import numpy as np

hm = HaloModel(z_range=(0.005, 5.0), n_z=40, n_mass=80, n_ell=10)
freqs = np.array([353., 600., 857., 1200., 3000.])  # GHz
snu = load_unfiltered_seds(freqs, hm.z)
cib = CIBModel(hm, snu, list(freqs),
               cc=np.ones_like(freqs), fc=np.ones_like(freqs),
               mdef='200c')

# Cosmic SFR density on the halo-model redshift grid
sfrd_z = cib.sfrd()           # M_sun/yr/Mpc^3, shape (n_z,)

# Mean CIB nu*<I_nu> (instrument-convention if cc != 1)
nu_inu = cib.mean_intensity()  # nW/m^2/sr, shape (n_freq,)

# Cheap parameter swap for MCMC / scans (skips the expensive halo-model
# precompute -- only re-runs the SFR-dependent emissivities):
cib.update_params({'Meff': 6e12, 'eta_max': 0.45,
                   'sigma_Mh': 1.4, 'tau': 0.85})
sfrd_z = cib.sfrd()
nu_inu = cib.mean_intensity()
```

`mean_intensity()` returns `cc * nu * integral`, so `cc` controls the
photometric convention of the output: pass `cc=1` for the bandpass-naive
nu*I_nu(=const) convention, or pass a per-frequency colour correction to
emit values in an instrument's convention.

`update_params()` keeps the cached halo-model precompute (HMF, bias, NFW
FT) and only re-runs the SFR-dependent emissivities, making `sfrd()` and
`mean_intensity()` cheap enough (~10 ms per call on the default grid) to
sit in the inner loop of an MCMC fit to SFRD or mean-CIB data.

### Galaxy surveys

Three galaxy survey presets are available:

| Survey | Description | HOD reference |
|--------|-------------|---------------|
| `'CMASS'` | BOSS CMASS LRGs | More+2015 |
| `'DESI_LRG'` | DESI Luminous Red Galaxies | More+2015 parametrization |
| `'DESI_ELG'` | DESI Emission Line Galaxies | More+2015 (ELG extension) |

### Generic tracer framework

For advanced use (custom cross-correlations, new tracers), the code provides a generic tracer-based framework:

```python
from halomodel_cib_tsz import (
    HaloModel, CIBModel, tSZModel, GalaxyHOD,
    CIBTracer, tSZTracer, GalaxyTracer, AngularCrossSpectrum,
)
from halomodel_cib_tsz.sed import load_planck_seds
from halomodel_cib_tsz import config

# Build halo model
hm = HaloModel(cosmo_name='planck18', n_mass=100, n_z=100, n_ell=50)
snu = load_planck_seds(hm.z)
exp = config.PLANCK

# Create physics models
cib = CIBModel(hm, snu, exp['freq_cib'], exp['cc'], exp['fc'], mdef='200c')
tsz = tSZModel(hm, exp['freq_cib'], experiment='Planck')
hod = GalaxyHOD('CMASS')

# Wrap as tracers
cib_tracer = CIBTracer(cib)
tsz_tracer = tSZTracer(tsz)
gal_tracer = GalaxyTracer(hm, hod)

# Compute any cross-correlation
xspec = AngularCrossSpectrum(cib_tracer, gal_tracer)
cl_1h = xspec.cl_1h()    # (n_freq, 1, n_ell)
cl_2h = xspec.cl_2h()    # (n_freq, 1, n_ell)
```

New tracers (e.g. CO line intensity mapping, CMB lensing) can be added by subclassing `Tracer` and implementing `central_weight()`, `satellite_weight()`, `window()`, and `n_fields`. Window functions can be frequency-dependent — return shape `(n_fields, n_z)` instead of `(n_z,)`.

When crossing two different multi-frequency tracers (e.g. CIB x tSZ), the engine automatically symmetrizes: `C[νi, νj] = ⟨A(νi)B(νj)⟩ + ⟨B(νi)A(νj)⟩`.

### Non-linear halo bias (β^NL)

The linear 2-halo term assumes P_hh(k, M, M') = b(M) b(M') P_lin(k). On
intermediate scales the halo-halo power exceeds this. [Mead & Verde (2021)](https://arxiv.org/abs/2011.08858)
measured the correction β^NL(k, z, M, M') in N-body simulations, and
[Jego et al. (2026)](https://arxiv.org/abs/2609.22057) applied it to the CIB halo
model:

P_2h(k, z) = P_lin(k, z) [1 + A β^NL_med(k, z)]

The package uses Jego et al.'s fast prescription, β^NL_med: the median over halo
mass pairs in 10^11.5–10^14.5 M_sun, looked up in peak height. It is extended as a
power law in k above the tabulated range (capped at k = 10 h/Mpc), set to zero below
k = 0.08 h/Mpc, and held constant at fixed peak height above z = 2.89. It applies to
every 2-halo term (CIB, tSZ, CIB × tSZ and the generic tracer pairs). For CIB
spectra it raises the 2-halo term by about ×1.75 at ℓ ≈ 1200 and ×2.2 at ℓ ≈ 2000.

```python
compute_spectra(beta_nl=False)                # linear-bias 2-halo term (as in Maniyar+2021)
HaloModel(beta_nl=True,                       # default
          beta_nl_amplitude=1.0,              # A
          beta_nl_options={'k_extrap': 'freeze'})   # e.g. hold beta at the last tabulated k
```

If you use it, please cite both Mead & Verde (2021) and Jego et al. (2026) (see
[Citation](#citation)). The tables in `data/beta_nl/bnl_tables.npz` are Mead &
Verde's MultiDark measurements, as used by Jego et al.

### Default CIB parameters

`config.CIB_PARAMS_200C` holds the posterior medians of a refit with β^NL on (A = 1),
with 68% intervals:

| parameter | median and 68% interval |
|---|---|
| M_eff [M_sun] | 1.60 (+0.19 −0.15) ×10¹² (log10 M_eff = 12.205 +0.050 −0.044) |
| η_max | 0.338 +0.044 −0.028 |
| σ_Mh | 1.65 +0.17 −0.20 |
| τ | 0.14 +0.17 −0.10 |

- `data/bestfit_200c.txt` lists these values with their errors.
- `data/bestfit_200c_covmat.txt` is the 4×4 posterior covariance in
  (log10 M_eff, η_max, σ_Mh, τ), e.g. for using the fit as a prior. τ peaks near its
  τ = 0 prior edge, so a Gaussian built from it is approximate.
- The Maniyar+2021 fits are kept as `data/bestfit_200c_maniyar2021.txt` and
  `data/bestfit_500c_maniyar2021.txt`.
- **Data:** Planck XXX CIB spectra at 217–857 GHz with a cross-frequency covariance,
  plus SFRD and mean-CIB constraints.
- **Halo model:** Planck18 with `hmf_model='tinker08_original'`. Use that HMF for
  consistency with these values.
- **CIB × tSZ:** the cross-spectrum uses M_500c halos for both tracers, as in
  Maniyar+2021. It no longer needs a separate 500c CIB fit: the same 200c parameters
  are evaluated at each halo's M_200c(M_500c, z), converted with an NFW profile and
  the halo model's concentration relation, via
  `CIBModel(..., mdef='500c', sfr_mdef='200c')`. So the cross uses the same CIB model
  as the auto spectra, including any `cib_params` you pass.
- **Mass-function consistency:** with the default colossus Tinker08 mass function,
  the 500c and 200c mass functions agree on the halo population to about 10%, and
  this conversion reproduces the 200c SFRD and mean CIB to within 6%. With
  `hmf_model='tinker08_original'`, the 500c mass function has 23–32% more halos than
  its own 200c version, so the converted CIB in the cross comes out about 25–35% too
  high. Prefer the default HMF for CIB × tSZ.
- **Paper's approach:** `config.CIB_PARAMS_500C`, the Maniyar+2021 fit made directly
  at 500c without β^NL, is kept for reproducing the paper:
  `CIBModel(..., params=CIB_PARAMS_500C, mdef='500c')`.

### Matching the original paper

To reproduce the results in Maniyar+2021, switch β^NL off and pass the paper's CIB
parameters:

```python
paper_params = {'Meff': 5.34372e12, 'eta_max': 0.5126, 'sigma_Mh': 1.2483, 'tau': 0.8255}
result = compute_spectra(
    components=('cib', 'tsz', 'cibxtsz'),
    cosmo='planck15',          # original used Planck 2015
    experiment='Planck',
    cib_params=paper_params,
    beta_nl=False,
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
- **Non-linear halo bias**: β^NL_med(k, z) of Mead & Verde (2021) in all 2-halo terms, as in Jego et al. (2026) (switchable)
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
  halo.py      -- HaloModel: colossus wrapper (HMF, bias, NFW FT, P(k), distances, beta^NL)
  beta_nl.py   -- Non-linear halo bias tables and their extensions (Mead & Verde 2021; Jego et al. 2026)
  sed.py       -- SED loading (Planck/SPIRE FITS, Bethermin tables), tSZ spectral fn
  cib.py       -- CIBModel: SFR, emissivities, 1h/2h power spectra, mean intensity
  tsz.py       -- tSZModel: pressure profile, y_ell, 1h/2h power spectra
  cross.py     -- CIBxTSZModel: CIB x tSZ cross-correlation
  galaxy.py    -- GalaxyHOD: More+2015 HOD (CMASS/DESI_LRG/DESI_ELG)
  tracers.py   -- Tracer ABC + CIBTracer, tSZTracer, GalaxyTracer
  angular_power.py -- AngularCrossSpectrum: generic Limber cross-Cl engine
  spectra.py   -- compute_spectra() unified interface
  data/        -- SED files, y_ell kernel, best-fit parameters, dN/dz files
  examples/
    driver.py        -- Example script with plotting
    quickstart.ipynb  -- Jupyter notebook tutorial
```

## Citation

If you use this code, please cite:

```bibtex
@article{Maniyar2021,
    author = {Maniyar, Abhishek S. and B{\'e}thermin, Matthieu and Lagache, Guilaine},
    title = {Simple halo model formalism for the cosmic infrared background and its
             correlation with the thermal Sunyaev-Zel'dovich effect},
    journal = {Astronomy \& Astrophysics},
    year = {2021},
    volume = {645},
    pages = {A40},
    doi = {10.1051/0004-6361/202038790},
    eprint = {2006.16329},
    archivePrefix = {arXiv},
}
```

If you use the non-linear halo bias correction (on by default), please also cite:

```bibtex
@article{MeadVerde2021,
    author = {Mead, Alexander J. and Verde, Licia},
    title = {Including beyond-linear halo bias in halo models},
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2021},
    volume = {503},
    pages = {3095--3111},
    eprint = {2011.08858},
    archivePrefix = {arXiv},
}

@article{Jego2026,
    author = {Jego, Baptiste and B{\'e}thermin, Matthieu and Kraljic, Katarina and
              Gkogkou, Athanasia and Lagache, Guilaine and Maniyar, Abhishek S.},
    title = {Non-linear halo bias for accurate modelling of cosmic infrared
             background anisotropies},
    year = {2026},
    eprint = {2609.22057},
    archivePrefix = {arXiv},
}
```

## License

See the original repository: https://github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz
