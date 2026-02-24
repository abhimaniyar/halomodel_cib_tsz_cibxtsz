"""
Validation script: run the halo model and check outputs against expected
physical behaviour from Maniyar, Béthermin & Lagache (2021).

Checks:
1. Mean CIB intensity monotonically increases with frequency
2. CIB auto-spectra are non-negative and symmetric
3. 1-halo dominates at high ell, 2-halo at low ell (CIB)
4. tSZ auto-spectra are non-negative and symmetric
5. CIB×tSZ sign: negative at low freq (tSZ spectral fn < 0 below ~217 GHz)
6. Mean CIB intensity order of magnitude (Table 5 of paper)

Usage:
    python -m halomodel_cib_tsz.examples.validate
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from halomodel_cib_tsz.spectra import compute_spectra


def main():
    print("=" * 60)
    print("Validation: CIB/tSZ/CIBxtSZ halo model")
    print("=" * 60)

    print("\nComputing all spectra (Planck, n_mass=80, n_z=80, n_ell=40)...")
    result = compute_spectra(
        components=('cib', 'tsz', 'cibxtsz'),
        experiment='Planck',
        n_mass=80,
        n_z=80,
        n_ell=40,
    )

    ell = result['ell']
    freqs = result['freqs']
    nfreq = len(freqs)
    freq_names = [str(int(f)) for f in freqs]

    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            failed += 1

    # ── 1. Mean CIB intensity ─────────────────────────────────────────
    print("\n--- Mean CIB Intensity ---")
    I_nu = result['I_nu']
    for i, f in enumerate(freq_names):
        print(f"  {f:>4s} GHz: {I_nu[i]:.4f} nW/m^2/sr")

    check("All I_nu > 0", np.all(I_nu > 0))
    check("I_nu monotonically increases with frequency",
          np.all(np.diff(I_nu) > 0))

    # Order-of-magnitude checks from Table 5 of Maniyar+2021
    # Published values (model, nW/m^2/sr):
    #   217 GHz ~ 0.06, 353 ~ 0.3-0.6, 545 ~ 2, 857 ~ 6
    # Our implementation uses different P(k) (Eisenstein-Hu vs pre-computed)
    # and different cosmology details, so exact match is not expected.
    # Check that values are within a factor of ~5 of published values.
    check("I_nu(857) > 0.5 nW/m^2/sr (order of magnitude)", I_nu[5] > 0.5)
    check("I_nu(857) < 30 nW/m^2/sr (order of magnitude)", I_nu[5] < 30)
    check("I_nu(100) < I_nu(857)", I_nu[0] < I_nu[5])

    # ── 2. CIB auto-spectra ──────────────────────────────────────────
    print("\n--- CIB Auto-Spectra ---")
    cl_cib = result['cl_cib']
    cl_cib_1h = result['cl_cib_1h']
    cl_cib_2h = result['cl_cib_2h']

    check("CIB auto-spectra non-negative",
          np.all(cl_cib >= 0))

    # Symmetry: Cl[i,j] == Cl[j,i]
    sym_err = 0.0
    for i in range(nfreq):
        for j in range(i + 1, nfreq):
            diff = np.abs(cl_cib[i, j, :] - cl_cib[j, i, :])
            rel = diff / (np.abs(cl_cib[i, j, :]) + 1e-30)
            sym_err = max(sym_err, np.max(rel))
    check(f"CIB symmetric across freq pairs (max rel err = {sym_err:.2e})",
          sym_err < 1e-10)

    # 1-halo vs 2-halo scaling
    # At high ell, 1-halo should dominate; at low ell, 2-halo
    f_idx = 1  # 143 GHz
    ratio_low = cl_cib_2h[f_idx, f_idx, 0] / (cl_cib_1h[f_idx, f_idx, 0] + 1e-30)
    ratio_high = cl_cib_1h[f_idx, f_idx, -1] / (cl_cib_2h[f_idx, f_idx, -1] + 1e-30)
    check(f"CIB 2h/1h ratio at low ell = {ratio_low:.2f} (expect > 1)",
          ratio_low > 1)
    check(f"CIB 1h/2h ratio at high ell = {ratio_high:.2f} (expect > 1)",
          ratio_high > 1)

    # ── 3. tSZ auto-spectra ──────────────────────────────────────────
    print("\n--- tSZ Auto-Spectra ---")
    cl_tsz = result['cl_tsz']
    cl_tsz_1h = result['cl_tsz_1h']
    cl_tsz_2h = result['cl_tsz_2h']

    # Only diagonal (same-frequency) auto-spectra must be non-negative.
    # Cross-frequency tSZ spectra can be negative when f_nu has opposite
    # signs (below and above the ~217 GHz tSZ null).
    tsz_diag_nonneg = all(
        np.all(cl_tsz[i, i, :] >= 0) for i in range(nfreq)
    )
    check("tSZ diagonal auto-spectra non-negative", tsz_diag_nonneg)

    # tSZ symmetry
    sym_err_tsz = 0.0
    for i in range(nfreq):
        for j in range(i + 1, nfreq):
            diff = np.abs(cl_tsz[i, j, :] - cl_tsz[j, i, :])
            rel = diff / (np.abs(cl_tsz[i, j, :]) + 1e-30)
            sym_err_tsz = max(sym_err_tsz, np.max(rel))
    check(f"tSZ symmetric across freq pairs (max rel err = {sym_err_tsz:.2e})",
          sym_err_tsz < 1e-10)

    # tSZ 1h should dominate at high ell
    ratio_tsz = cl_tsz_1h[f_idx, f_idx, -1] / (cl_tsz_2h[f_idx, f_idx, -1] + 1e-30)
    check(f"tSZ 1h/2h ratio at high ell = {ratio_tsz:.2f} (expect > 1)",
          ratio_tsz > 1)

    # ── 4. CIB×tSZ cross-spectra ────────────────────────────────────
    print("\n--- CIB x tSZ Cross-Spectra ---")
    cl_cross = result['cl_cibxtsz']

    # At 143 GHz, tSZ spectral fn is negative → cross should be negative
    # (since CIB emissivity is positive)
    check("CIBxtSZ at 143x143 GHz is negative (f_nu < 0 below 217 GHz)",
          np.all(cl_cross[1, 1, :] <= 0))

    # At 857 GHz, tSZ spectral fn is positive → check 857x857 sign
    check("CIBxtSZ at 857x857 GHz is positive (f_nu > 0 above 217 GHz)",
          np.all(cl_cross[5, 5, :] >= 0))

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    # ── Plots ────────────────────────────────────────────────────────
    _make_plots(result, ell, freq_names)

    return failed == 0


def _make_plots(result, ell, freq_names):
    """Generate validation plots."""
    nfreq = len(freq_names)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── Row 1: CIB auto-spectra for a few freq pairs ────────────────
    pairs = [(1, 1), (3, 3), (5, 5)]  # 143x143, 353x353, 857x857
    for idx, (i, j) in enumerate(pairs):
        ax = axes[0, idx]
        ax.loglog(ell, result['cl_cib_1h'][i, j, :], 'b-.', label='1h', alpha=0.7)
        ax.loglog(ell, result['cl_cib_2h'][i, j, :], 'b--', label='2h', alpha=0.7)
        ax.loglog(ell, result['cl_cib'][i, j, :], 'b-', label='total', lw=2)
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$C_\ell$ [Jy$^2$/sr]')
        ax.set_title(f'CIB {freq_names[i]}x{freq_names[j]} GHz')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

    # ── Row 2, col 0: tSZ 143x143 ───────────────────────────────────
    ax = axes[1, 0]
    ax.loglog(ell, result['cl_tsz_1h'][1, 1, :], 'r-.', label='1h', alpha=0.7)
    ax.loglog(ell, result['cl_tsz_2h'][1, 1, :], 'r--', label='2h', alpha=0.7)
    ax.loglog(ell, result['cl_tsz'][1, 1, :], 'r-', label='total', lw=2)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell$ [Jy$^2$/sr]')
    ax.set_title(f'tSZ {freq_names[1]}x{freq_names[1]} GHz')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # ── Row 2, col 1: CIBxtSZ 143x857 ───────────────────────────────
    ax = axes[1, 1]
    cl_x = result['cl_cibxtsz']
    ax.loglog(ell, np.abs(cl_x[1, 5, :]), 'g-', lw=2, label='|total|')
    ax.loglog(ell, np.abs(result['cl_cibxtsz_1h'][1, 5, :]),
              'g-.', label='|1h|', alpha=0.7)
    ax.loglog(ell, np.abs(result['cl_cibxtsz_2h'][1, 5, :]),
              'g--', label='|2h|', alpha=0.7)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$|C_\ell|$ [Jy$^2$/sr]')
    ax.set_title(f'CIBxtSZ {freq_names[1]}x{freq_names[5]} GHz')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # ── Row 2, col 2: Mean CIB intensity ────────────────────────────
    ax = axes[1, 2]
    freqs_float = [float(f) for f in freq_names]
    ax.semilogy(freqs_float, result['I_nu'], 'ko-', lw=2)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel(r'$\langle I_\nu \rangle$ [nW/m$^2$/sr]')
    ax.set_title('Mean CIB Intensity')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('validation_plots.png', dpi=150)
    print("\nPlots saved to validation_plots.png")
    plt.close()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
