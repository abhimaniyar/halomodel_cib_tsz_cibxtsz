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
        self.cc_cibmean = self.exp['cc_cibmean']
        self.freq_Iv = self.exp['freq_Iv']
        self.fc = self.exp['fc']

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
        addr = 'data_files/matter_power_spectra'
        pkarray = np.loadtxt('%s/test_highk_lin_matterpower_210.dat' % (addr))
        k = pkarray[:, 0]*cosmo.h
        Pk = np.zeros((len(k), len(redshifts)))
        for i in range(len(redshifts)):
            pkarray = np.loadtxt("%s/test_highk_lin_matterpower_%s.dat" % (addr, ll[209-i]))
            Pk[:, i] = pkarray[:, 1]/cosmo.h**3

        # pkinterp2d = RectBivariateSpline(k, redshifts, Pk)
        # pkinterpk = interp1d(k, Pk.T, kind='linear', bounds_error=False, fill_value=0.)
        # pkinterpz = interp1d(redshifts, Pk, kind='linear', bounds_error=False, fill_value=0.)
        pkinterpz = interp1d(redshifts, Pk, kind='linear', bounds_error=False, fill_value="extrapolate")

        self.k_array = np.zeros((len(self.ell), len(self.z)))
        self.Pk_int = np.zeros(self.k_array.shape)
        chiz = cosmo.comoving_distance(self.z).value
        for i in range(len(self.ell)):
            self.k_array[i, :] = self.ell[i]/chiz
            for j in range(len(self.z)):
                pkz = pkinterpz(self.z[j])
                self.Pk_int[i, j] = np.interp(self.k_array[i, j], k, pkz)

        """
        for i in range(len(ell)):
            self.k_array[i, :] = self.ell[i]/chiz
            self.Pk_int[i, :] = pkinterpk(self.k_array[i, :])
            # self.Pk_int[i, :] = pkinterp2d(self.k_array[i, :], self.z)
        """

        if self.exp['do_cib'] == 1 or self.exp['do_cibxtsz'] == 1:
            self.z_c = 1.5
            # ######### reading and interpolating the SEDs
            snuaddr = self.exp['snuaddr']
            hdulist = fits.open(snuaddr)
            """
            The effective SEDs for the CIB for Planck (100, 143, 217, 353, 545,
            857) and
            IRAS (3000) GHz frequencies.
            Here we are shwoing the CIB power spectra corressponding to the
            Planck
            frequency channels. If you want to calculate the Hershel/Spire
            power spectra, use corresponding files in the data folder.
            """
            redshifts = hdulist[1].data
            snu_eff = hdulist[0].data  # in Jy/Lsun
            hdulist.close()
            # snuinterp = interp1d(redshifts, snu_eff, kind='linear',
            #                      bounds_error=False, fill_value=0.)
            snuinterp = interp1d(redshifts, snu_eff, kind='linear',
                                 bounds_error=False, fill_value="extrapolate")
            self.snu = snuinterp(z)

        if self.exp['do_cib'] == 1:
            # ######### CIB halo model parameters ###################
            cibparresaddr = self.exp['cibpar_resfile']
            # self.Meffmax, self.etamax, self.sigmaMh, self.tau = np.loadtxt(cibparresaddr)[:4, 0]
            self.Meffmax, self.etamax, self.sigmaMh, self.tau = 8753289339381.791, 0.4028353504978569, 1.807080723258688, 1.2040244128818796

            # if name == 'Planck_only':
                # self.fc[-4:] = np.loadtxt(cibparresaddr)[-4:, 0]

            # ######## hmf, bias, nfw ###########
            print ("Calculating the halo mass function, halo bias, nfw " +
                   "profile " +
                   "for given mass and redshiftcfor CIB calculations.")
    
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
            self.nu = np.array([100., 143., 217., 353., 545., 857.])*ghz
            nus = ['100', '143', '217', '353', '545', '857']
            self.delta_h_tsz = 500
            self.B = 1.41
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
            self.snu = self.snu[:-1, :]
            self.cc = self.exp['cc'][:-1]
            self.fc = self.exp['fc'][:-1]
            # ################################ cib x tSZ #####################
            cibxtszparresaddr = self.exp['cibxtszpar_resfile']
            self.Meffmax, self.etamax, self.sigmaMh, self.tau = np.loadtxt(cibxtszparresaddr)[:4, 0]
            # self.Meffmax_cross, self.etamax_cross, self.sigmaMh_cross = 6962523672799.227, 0.4967291547804018, 1.8074450009861387
            # self.tau_cross = 1.2016980179374213
