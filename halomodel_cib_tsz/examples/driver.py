"""
Example driver: compute and plot CIB, tSZ, and CIB×tSZ power spectra.

Usage:
    python -m halomodel_cib_tsz.examples.driver
"""

import numpy as np
import matplotlib.pyplot as plt

from halomodel_cib_tsz.spectra import compute_spectra


def main():
    print("Computing all spectra (Planck, default params)...")
    result = compute_spectra(
        components=('cib', 'tsz', 'cibxtsz'),
        experiment='Planck',
        n_mass=80,
        n_z=80,
        n_ell=40,
    )

    ell = result['ell']
    freqs = result['freqs']
    freq_names = ['100', '143', '217', '353', '545', '857']

    # ── Print mean CIB intensity ─────────────────────────────────────
    print("\nMean CIB intensity [nW/m^2/sr]:")
    for i, f in enumerate(freq_names):
        print(f"  {f} GHz: {result['I_nu'][i]:.4f}")

    # ── Plot CIB 143×143 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    nu1, nu2 = 1, 1  # 143×143
    ax = axes[0]
    ax.loglog(ell, np.abs(result['cl_cib_1h'][nu1, nu2, :]), 'b-.', label='1-halo')
    ax.loglog(ell, np.abs(result['cl_cib_2h'][nu1, nu2, :]), 'b--', label='2-halo')
    ax.loglog(ell, np.abs(result['cl_cib'][nu1, nu2, :]), 'b-', label='total')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell$ [Jy$^2$/sr]')
    ax.set_title(f'CIB {freq_names[nu1]}×{freq_names[nu2]} GHz')
    ax.legend(frameon=False)
    ax.grid(True, which='both', alpha=0.3)

    # ── Plot tSZ 143×143 ─────────────────────────────────────────────
    nu1, nu2 = 1, 1
    ax = axes[1]
    ax.loglog(ell, np.abs(result['cl_tsz_1h'][nu1, nu2, :]), 'r-.', label='1-halo')
    ax.loglog(ell, np.abs(result['cl_tsz_2h'][nu1, nu2, :]), 'r--', label='2-halo')
    ax.loglog(ell, np.abs(result['cl_tsz'][nu1, nu2, :]), 'r-', label='total')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell$ [Jy$^2$/sr]')
    ax.set_title(f'tSZ {freq_names[nu1]}×{freq_names[nu2]} GHz')
    ax.legend(frameon=False)
    ax.grid(True, which='both', alpha=0.3)

    # ── Plot CIB×tSZ 143×857 ─────────────────────────────────────────
    nu1, nu2 = 1, 5  # 143×857
    ax = axes[2]
    ax.loglog(ell, np.abs(result['cl_cibxtsz_1h'][nu1, nu2, :]), 'g-.', label='1-halo')
    ax.loglog(ell, np.abs(result['cl_cibxtsz_2h'][nu1, nu2, :]), 'g--', label='2-halo')
    ax.loglog(ell, np.abs(result['cl_cibxtsz'][nu1, nu2, :]), 'g-', label='total')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$|C_\ell|$ [Jy$^2$/sr]')
    ax.set_title(f'CIB×tSZ {freq_names[nu1]}×{freq_names[nu2]} GHz')
    ax.legend(frameon=False)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('power_spectra.png', dpi=150)
    print("\nPlot saved to power_spectra.png")
    plt.show()


if __name__ == '__main__':
    main()
