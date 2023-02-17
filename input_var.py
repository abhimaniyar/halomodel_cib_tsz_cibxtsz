from headers_constants import *


class data_var(object):

    def __init__(self, exp, mass, z, ell):
        # ############### cib data #########################
        self.exp = exp
        name = self.exp['name']
        deltah_cib = 200
        self.z_c = 1.5
        self.freqcib = self.exp['freq_cib']
        self.cc = self.exp['cc']
        self.fc = self.exp['fc']

        self.cc_cibmean = self.exp['cc_cibmean']
        self.freq_cibmean = self.exp['freq_cibmean']

        self.mass = mass
        self.z = z
        self.ell = ell

        nm = len(self.mass)
        nz = len(self.z)

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

        self.k_array = np.zeros((len(self.ell), len(self.z)))
        self.Pk_int = np.zeros(self.k_array.shape)
        """
        Pk_int 2-d array for corresponding redshifts and
        given ell range such that k = ell/chi i.e. for every
        redshift
        """
        chiz = cosmo.comoving_distance(self.z).value
        for i in range(len(self.ell)):
            self.k_array[i, :] = self.ell[i]/chiz
            for j in range(len(self.z)):
                pkz = pkinterpz(self.z[j])
                self.Pk_int[i, j] = np.interp(self.k_array[i, j], k, pkz)


        if self.exp['do_cib'] == 1 or self.exp['do_cibxtsz'] == 1:
            # ######### reading and interpolating the SEDs
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
                fsnu_eff = interp1d(redshifts, snu_eff, kind='linear',
                                    bounds_error=False, fill_value="extrapolate")
                self.snu = fsnu_eff(self.z)
            elif name == 'Herschel-spire':
                snuaddr = 'data_files/filtered_snu_spire.fits'
                hdulist = fits.open(snuaddr)
                redshifts = hdulist[1].data
                snu_eff = hdulist[0].data  # in Jy/Lsun
                hdulist.close()
                fsnu_eff = interp1d(redshifts, snu_eff, kind='linear',
                                    bounds_error=False, fill_value="extrapolate")
                self.snu = fsnu_eff(self.z)
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
                
                nuinp = self.freqcib
                self.snu = fsnu_unfiltered(nuinp, self.z)


        if self.exp['do_cib'] == 1:
            # ######### CIB halo model parameters ###################
            cibparresaddr = 'data_files/one_halo_bestfit_allcomponents_lognormal_sigevol_1p5zcutoff_nospire_fcpl_onlyautoshotpar_no3000_gaussian600n857n1200_planck_spire_hmflog10.txt'
            self.Meffmax, self.etamax, self.sigmaMh, self.tau = np.loadtxt(cibparresaddr)[:4, 0]
            # self.Meffmax, self.etamax, self.sigmaMh, self.tau = 8753289339381.791, 0.4028353504978569, 1.807080723258688, 1.2040244128818796


            # ######## hmf, bias, nfw ###########
            print ("Calculating the halo mass function, halo bias, nfw " +
                   "profile " +
                   "for given mass and redshift for CIB calculations.")
    
            self.hmf = np.zeros((nm, nz))
            self.u_nfw = np.zeros((nm, len(self.k_array[:, 0]), nz))
            self.bias_m_z = np.zeros((nm, nz))
            delta_h = deltah_cib
    
            for r in range(nz):
                pkz = pkinterpz(self.z[r])
                instance = hmf_unfw_bias.h_u_b(k, pkz, self.z[r],
                                               cosmo, delta_h, self.mass)
                self.hmf[:, r] = instance.dn_dlogm()
                # nfw_u[:, :, r] = instance.nfwfourier_u()
                self.bias_m_z[:, r] = instance.b_nu()
                instance2 = hmf_unfw_bias.h_u_b(self.k_array[:, r],
                                                self.Pk_int[:, r], self.z[r],
                                                cosmo, delta_h, self.mass)
                self.u_nfw[:, :, r] = instance2.nfwfourier_u()

        if self.exp['do_tsz'] == 1 or self.exp['do_cibxtsz'] == 1:
            # ############################### tSZ params #####################
            xstep = 50
            lnx = np.linspace(-6, 1, xstep)
            self.x = 10**lnx
            # self.nutsz = np.array([100., 143., 217., 353., 545., 857.])*ghz
            self.nutsz = np.array(self.freqcib)*ghz
            #nus = ['100', '143', '217', '353', '545', '857']
            self.delta_h_tsz = 500  # 500 # 200
            self.B = 1.5  # 1.41
            self.m500 = np.repeat(self.mass[..., np.newaxis], len(self.z),
                                  axis=1)

            print ("Calculating the halo mass function, halo bias, nfw " +
                   "profile " +
                   "for given mass and redshift for tSZ calculations.")
            self.hmf = np.zeros((len(self.m500), nz))
            self.bias_m_z = np.zeros((len(self.m500[:, 0]), nz))
            self.u_nfw = np.zeros((nm, len(self.k_array[:, 0]), nz))
            delta_h = self.delta_h_tsz

            for r in range(nz):
                pkz = pkinterpz(self.z[r])
                instance = hmf_unfw_bias.h_u_b(k, pkz, self.z[r],
                                               cosmo, delta_h, self.m500[:, 0])
                self.hmf[:, r] = instance.dn_dlogm()
                # nfw_u[:, :, r] = instance.nfwfourier_u()
                self.bias_m_z[:, r] = instance.b_nu()
                instance2 = hmf_unfw_bias.h_u_b(self.k_array[:, r],
                                                self.Pk_int[:, r], self.z[r],
                                                cosmo, delta_h, self.mass)
                self.u_nfw[:, :, r] = instance2.nfwfourier_u()

        if self.exp['do_cibxtsz'] == 1:
            # mass definition here is m500 for both the CIB and tSZ. The best fit values
            # for CIB parameters change due to this.

            self.snu = self.snu[:-1, :]
            self.cc = self.exp['cc'][:-1]
            self.fc = self.exp['fc'][:-1]
            # ################################ cib x tSZ #####################
            cibxtszparresaddr = 'data_files/one_halo_bestfit_allcomponents_lognormal_sigevol_highk_deltah500_onlyautoshotpar_no3000_gaussian600n857n1200_planck_spire_hmflog10.txt'
            self.Meffmax, self.etamax, self.sigmaMh, self.tau = np.loadtxt(cibxtszparresaddr)[:4, 0]
            # self.Meffmax_cross, self.etamax_cross, self.sigmaMh_cross = 6962523672799.227, 0.4967291547804018, 1.8074450009861387
            # self.tau_cross = 1.2016980179374213
