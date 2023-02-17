from headers_constants import *
from input_var_cibmean import *
from Inu_cib import *

"""
Calculating the observed CIB intensity for halos with a given mass at given
redshifts for different Planck frequencies. The SEDs used for the Planck
channels are bandpassed, so the observed intensities are calculated as
they would be observed with Planck frequency channels at 100, 143, 353, 545,
847 GHz as well as 3000 GHz for IRAS. Intensity is calculated in nW/m^2/sr.
"""

Planck = {'name': 'Planck',
          'do_cib': 1, 'do_tsz': 1, 'do_cibxtsz': 1,
          'freq_cib': [100., 143., 217., 353., 545., 857.],
          'cc': np.array([1.076, 1.017, 1.119, 1.097, 1.068, 0.995, 0.960]),
          'cc_cibmean': np.array([1.076, 1.017, 1.119, 1.097, 1.068, 0.995, 0.960]),
          'freq_cibmean': np.array([100., 143., 217., 353., 545., 857.]),
          'fc': np.ones(7),
          }

Herschel = {'name': 'Herschel-spire',
            'do_cib': 1, 'do_tsz': 0, 'do_cibxtsz': 0,
            'freq_cib': [600., 857., 1200.],
            'cc': np.array([0.974, 0.989, 0.988]),
            'cc_cibmean': np.array([0.974, 0.989, 0.988]),
            'freq_cibmean': np.array([600., 857., 1200.]),
            'fc': np.ones(3),
            }

CCAT = {'name': 'CCAT-p',
        'do_cib': 1, 'do_tsz': 0, 'do_cibxtsz': 0,
        'freq_cib': [220., 280., 350., 410., 850.],
        'cc': np.ones(5),
        'cc_cibmean': np.ones(5),
        'freq_cibmean': np.array([220., 280., 350., 410., 850.]),
        'fc': np.ones(5),
         }

exp = Planck
# cc_cibmean = np.array([0.97125, 0.999638, 1.00573, 0.959529, 0.973914, 0.988669, 0.987954, 1., 1.])
# freq_cibmean = np.array([1875, 3000, 667, 353, 600, 857, 1200, 250, 242])
# freq_cibmean = np.array([100., 143., 217., 353., 545., 857., 3000.])

# ell = np.linspace(150., 2000., 20)
redshifts = np.loadtxt('data_files/redshifts.txt')

zsource = 9.
z1 = np.linspace(min(redshifts), zsource, 200)
z = z1  # z1  # redshifts # z1

logmass = np.arange(6, 15.005, 0.1)
mass = 10**logmass

driver = data_var_iv(exp, mass, z)  # , ell)

if exp['do_cibmean'] == 1:
    cibmean = I_nu_cib(driver)
    I_nu = cibmean.Iv()
    freq = exp['freq_cibmean']  # ['100', '143', '217', '353', '545', '857', '3000']
    # Iv_cen = np.array([13.63, 12.61, 1.64, 0.46, 2.8, 6.6, 10.1, 0.08, 0.05])
    for i in range(len(I_nu)):
        print ("Intensity is %f nW/m^2/sr at %s GHz" % (I_nu[i], str(freq[i])))
