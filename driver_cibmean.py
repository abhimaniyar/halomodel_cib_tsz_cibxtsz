from headers_constants import *
from input_var_cibmean import *
from Inu_cib import *

Planck = {'name': 'Planck_only',
          'do_cibmean': 1,
          'cc_cibmean': np.array([1.076, 1.017, 1.119, 1.097, 1.068, 0.995, 0.960]),
          'freq_Iv': np.array([100., 143., 217., 353., 545., 857., 3000.]),
          'snuaddr': 'data_files/filtered_snu_planck.fits',

          'cibpar_resfile': 'data_files/one_halo_bestfit_allcomponents_' +
          'lognormal_sigevol_1p5zcutoff_nospire_fcpl_onlyautoshotpar_' +
          'no3000_gaussian600n857n1200_planck_spire_hmflog10.txt'}

exp = Planck
# cc_cibmean = np.array([0.97125, 0.999638, 1.00573, 0.959529, 0.973914, 0.988669, 0.987954, 1., 1.])
# freq_iv = np.array([1875, 3000, 667, 353, 600, 857, 1200, 250, 242])
# freq_iv = np.array([100., 143., 217., 353., 545., 857., 3000.])
# snuaddr: 'data_files/filtered_snu_cib_15_new.fits'

ell = np.linspace(150., 2000., 20)
redshifts = np.loadtxt('data_files/redshifts.txt')
z1 = np.linspace(min(redshifts), max(redshifts), 200)
# z2 = np.linspace(min(redshifts), 1.5, 80)
# z3 = np.linspace(1.51, max(redshifts), 30)
# z11 = np.concatenate((z2, z3))
# zn = np.linspace(min(redshifts), 3., 130)
z = z1  # z1  # redshifts # zn # z11

logmass = np.arange(6, 15.005, 0.1)
mass = 10**logmass

driver = data_var_iv(exp, mass, z, ell)

if exp['do_cibmean'] == 1:
    cibmean = I_nu_cib(driver)
    I_nu = cibmean.Iv()
    freq = ['100', '143', '217', '353', '545', '857', '3000']
    # Iv_cen = np.array([13.63, 12.61, 1.64, 0.46, 2.8, 6.6, 10.1, 0.08, 0.05])
    for i in range(len(I_nu)):
        print "Intensity is %f nW/m^2/sr at %s GHz" % (I_nu[i], freq[i])
