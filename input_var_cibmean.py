from headers_constants import *


class data_var_iv(object):

    def __init__(self, exp, mass, z):  # , ell):
        # ############### cib data #########################
        self.exp = exp
        name = self.exp['name']
        self.z_c = 1.5
        self.cc_cibmean = self.exp['cc_cibmean']
        self.freq_cibmean = self.exp['freq_cibmean']

        self.mass = mass
        self.z = z
        # self.ell = ell

        nm = len(self.mass)
        nz = len(self.z)

        deltah_cib = 200.
        # ########## reading in the matter power spectra #############
        redshifts = np.loadtxt('data_files/redshifts.txt')

        if min(self.z) < min(redshifts) or max(self.z) > max(redshifts):
            print ("If the redshift range is outside of [%s to %s], then " +
                   "values of the matter power spectrum and effective " +
                   "CIB SEDs are extrapolated and might be incorrect.") % (min(redshifts), max(redshifts))
        ll = [str(x) for x in range(1, 211)]
        """
        please note that the matter power spectrum files here are arranged in
        a reversed order i.e. redshift decreases as you go from _1 file to _210.
        Perhaps better to generate your own power spectra for the redshift
        array you want to consider and read that here. Also, note that I am
        getting rid of the reduced Hubble constant here from units of k and Pk.
        """
        addr = 'data_files/matter_power_spectra'
        pkarray = np.loadtxt('%s/test_highk_lin_matterpower_210.dat' % (addr))
        k = pkarray[:, 0]*cosmo.h
        Pk = np.zeros((len(k), len(redshifts)))
        for i in range(len(redshifts)):
            pkarray = np.loadtxt("%s/test_highk_lin_matterpower_%s.dat" % (addr, ll[209-i]))
            Pk[:, i] = pkarray[:, 1]/cosmo.h**3

        pkinterpz = interp1d(redshifts, Pk, kind='linear', bounds_error=False, fill_value="extrapolate")

        if self.exp['do_cibmean'] == 1:
            # ######### reading and interpolating the SEDs
            # snuaddr = self.exp['snuaddr']
            """
            The effective SEDs for the CIB for Planck (100, 143, 217, 353, 545,
            857) and
            IRAS (3000) GHz frequencies.
            Here we are shwoing the CIB power spectra corressponding to the
            Planck
            frequency channels. If you want to calculate the Hershel/Spire
            power spectra, use corresponding files in the data folder.
            """
            if name == 'Planck':
                snuaddr = 'data_files/filtered_snu_planck.fits'
                hdulist = fits.open(snuaddr)
                redshifts = hdulist[1].data
                snu_eff = hdulist[0].data  # in Jy/Lsun
                hdulist.close()
                # snuinterp = interp1d(redshifts, snu_eff, kind='linear',
                #                      bounds_error=False, fill_value=0.)
                snuinterp = interp1d(redshifts, snu_eff, kind='linear',
                                     bounds_error=False, fill_value="extrapolate")
                self.snu = snuinterp(z)
            elif name == 'Herschel-spire':
                snuaddr = 'data_files/filtered_snu_spire.fits'
                hdulist = fits.open(snuaddr)
                redshifts = hdulist[1].data
                snu_eff = hdulist[0].data  # in Jy/Lsun
                hdulist.close()
                # snuinterp = interp1d(redshifts, snu_eff, kind='linear',
                #                      bounds_error=False, fill_value=0.)
                snuinterp = interp1d(redshifts, snu_eff, kind='linear',
                                     bounds_error=False, fill_value="extrapolate")
                self.snu = snuinterp(z)
            else:
                # ######### unfiltered SEDs ###########################

                list_of_files = sorted(glob.glob('data_files/TXT_TABLES_2015/./*.txt'))
                a = list_of_files[95]
                b = list_of_files[96]
                for i in range(95, 208):
                    list_of_files[i] = list_of_files[i+2]
                list_of_files[208] = a
                list_of_files[209] = b

                wavelengths = np.loadtxt('data_files/TXT_TABLES_2015/EffectiveSED_B15_z0.012.txt')[:, [0]]
                # wavelengths are in microns
                freq = c_light/wavelengths
                # c_light is in Km/s, wavelength is in microns and we would like to
                # have frequency in GHz. So gotta multiply by the following
                # numerical factor which comes out to be 1
                # numerical_fac = 1e3*1e6/1e9
                numerical_fac = 1.
                freqhz = freq*1e3*1e6
                freq *= numerical_fac
                freq_rest = freqhz*(1+redshifts)

                n = np.size(wavelengths)

                snu_unfiltered = np.zeros([n, len(redshifts)])
                for i in range(len(list_of_files)):
                    snu_unfiltered[:, i] = np.loadtxt(list_of_files[i])[:, 1]
                L_IR15 = self.L_IR(snu_unfiltered, freq_rest, redshifts)
                # print (L_IR15)

                for i in range(len(list_of_files)):
                    snu_unfiltered[:, i] = snu_unfiltered[:, i]*L_sun/L_IR15[i]

                # Currently unfiltered snus are ordered in increasing wavelengths,
                # we re-arrange them in increasing frequencies i.e. invert it

                freq = freq[::-1]
                snu_unfiltered = snu_unfiltered[::-1]
                fsnu_unfiltered = RectBivariateSpline(freq, redshifts,
                                                      snu_unfiltered)
                
                nuinp = self.exp['freq_cibmean']
                self.snu = fsnu_unfiltered(nuinp, self.z)

            # ######### CIB halo model parameters ###################
            cibparresaddr = 'data_files/one_halo_bestfit_allcomponents_lognormal_sigevol_1p5zcutoff_nospire_fcpl_onlyautoshotpar_no3000_gaussian600n857n1200_planck_spire_hmflog10.txt'
            self.Meffmax, self.etamax, self.sigmaMh, self.tau = np.loadtxt(cibparresaddr)[:4, 0]
            # self.Meffmax, self.etamax, self.sigmaMh, self.tau = 8753289339381.791, 0.4028353504978569, 1.807080723258688, 1.2040244128818796

            # if name == 'Planck_only':
                # self.fc[-4:] = np.loadtxt(cibparresaddr)[-4:, 0]

            # ######## hmf, bias, nfw ###########
            print ("Calculating the halo mass function " +
                   "for given mass and redshift for CIB mean calculations.")
    
            self.hmf = np.zeros((nm, nz))
            delta_h = deltah_cib
    
            for r in range(nz):
                pkz = pkinterpz(self.z[r])
                instance = hmf_unfw_bias.h_u_b(k, pkz, self.z[r],
                                               cosmo, delta_h, self.mass)
                self.hmf[:, r] = instance.dn_dlogm()
