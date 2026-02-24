"""
Reproduce Fig. 6 of Maniyar+2021: dimensionless tSZ power spectrum.

Compares:
  - Default (colossus tinker08 + Planck18): previous ~40% low result
  - tinker08_original + Planck15: matches the paper to within ~2%

Plots D_ell = 10^12 * ell*(ell+1)/(2*pi) * C_ell^{yy}
with 1-halo, 2-halo, and total for both configurations.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from halomodel_cib_tsz.halo import HaloModel
from halomodel_cib_tsz.tsz import tSZModel
from halomodel_cib_tsz import config

# Grid settings — n=60 for speed (~2 min per config)
n_mass = 60
n_z = 60
n_ell = 40
ell_range = (10, 1e4)

freqs = config.PLANCK['freq_cib']
Kcmb_MJy = config.PLANCK['Kcmb_MJy']

# Use 143 GHz (index 1) for extracting Cl_yy
fidx = 1

def run_tsz(cosmo_name, hmf_model, label):
    """Run tSZ and return (ell, dl_1h, dl_2h, dl_tot) in dimensionless yy units."""
    t0 = time.time()
    print(f"  Building HaloModel ({cosmo_name}, {hmf_model})...", flush=True)
    hm = HaloModel(
        cosmo_name=cosmo_name,
        hmf_model=hmf_model,
        n_mass=n_mass, n_z=n_z, n_ell=n_ell,
        ell_range=ell_range,
    )
    print(f"    HaloModel: {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"  Building tSZModel...", flush=True)
    tsz = tSZModel(hm, freqs, experiment='Planck', B=config.TSZ_B)
    print(f"    tSZModel: {time.time()-t1:.1f}s", flush=True)

    t2 = time.time()
    cl_1h = tsz.cl_1h()
    cl_2h = tsz.cl_2h()
    print(f"    Cl computation: {time.time()-t2:.1f}s", flush=True)

    # Divide out f(nu) to get dimensionless Cl_yy
    fnu_factor = (tsz._f_nu[fidx] * 1e6 * Kcmb_MJy[fidx])**2
    cl_yy_1h = cl_1h[fidx, fidx, :] / fnu_factor
    cl_yy_2h = cl_2h[fidx, fidx, :] / fnu_factor

    ell = hm.ell
    prefactor = 1e12 * ell * (ell + 1) / (2 * np.pi)
    dl_1h = prefactor * cl_yy_1h
    dl_2h = prefactor * cl_yy_2h
    dl_tot = dl_1h + dl_2h

    idx_peak = np.argmax(dl_tot)
    print(f"  {label}: peak D_ell^yy x 10^12 = {dl_tot[idx_peak]:.4f} "
          f"at ell={ell[idx_peak]:.0f}  (total: {time.time()-t0:.1f}s)", flush=True)

    return ell, dl_1h, dl_2h, dl_tot


# ── Run both configurations ──────────────────────────────────────────

print("Configuration 1: colossus tinker08 + Planck18 (default)", flush=True)
ell1, dl1_1h, dl1_2h, dl1_tot = run_tsz('planck18', 'tinker08', 'Default')

print("\nConfiguration 2: tinker08_original + Planck15 (matching paper)", flush=True)
ell2, dl2_1h, dl2_2h, dl2_tot = run_tsz('planck15', 'tinker08_original', 'Paper-matched')


# ── Plot ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

# Configuration 1: default (gray, thinner)
ax.loglog(ell1, dl1_tot, '-', color='gray', lw=1.5, alpha=0.7,
          label='total (colossus tinker08 + Planck18)')
ax.loglog(ell1, dl1_1h, '--', color='gray', lw=1, alpha=0.5,
          label='1-halo (colossus tinker08 + Planck18)')
ax.loglog(ell1, dl1_2h, '-.', color='gray', lw=1, alpha=0.5,
          label='2-halo (colossus tinker08 + Planck18)')

# Configuration 2: paper-matched (black, bold)
ax.loglog(ell2, dl2_tot, 'k-', lw=2.5, label='total (tinker08_original + Planck15)')
ax.loglog(ell2, dl2_1h, 'k--', lw=1.5, label='1-halo (tinker08_original + Planck15)')
ax.loglog(ell2, dl2_2h, 'k-.', lw=1.5, label='2-halo (tinker08_original + Planck15)')

# Reference line at D_ell = 1.0 (approximate paper peak)
ax.axhline(1.0, color='red', ls=':', lw=1, alpha=0.5, label='Paper peak ~1.0')

ax.set_xlabel(r'$\ell$', fontsize=14)
ax.set_ylabel(r'$10^{12}\,\ell(\ell+1)\,C_\ell^{yy}/2\pi$', fontsize=14)
ax.set_xlim(10, 1e4)
ax.set_ylim(1e-3, 5)
ax.legend(fontsize=9, frameon=False, loc='lower right')
ax.tick_params(which='both', direction='in', top=True, right=True)
ax.grid(True, which='both', alpha=0.2)
ax.set_title('tSZ power spectrum — Fig. 6 of Maniyar+2021', fontsize=13)

# Add text annotation
peak2 = dl2_tot.max()
ax.text(0.03, 0.97,
        f'Paper-matched peak: {peak2:.3f}\n'
        f'Default peak: {dl1_tot.max():.3f}\n'
        f'Paper target: ~1.0',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plot_tsz_comparison.png', dpi=150)
print(f"\nSaved plot_tsz_comparison.png", flush=True)
