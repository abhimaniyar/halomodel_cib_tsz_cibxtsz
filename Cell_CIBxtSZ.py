from headers_constants import *

# mass definition here is m500 for both the CIB and tSZ. The best fit values
# for CIB parameters change due to this


class cl_cibxtsz(object):

    def __init__(self, cell_cib, cell_tsz):
        self.cib = cell_cib
        self.tsz = cell_tsz

        self.nfreq = self.cib.nfreq
        self.cosmo = self.cib.cosmo
        self.z = self.cib.z
        self.mh = self.cib.mh
        self.ell = self.cib.ell

    def E_z(self):
        return np.sqrt(self.cosmo.Om0*(1+self.z)**3 + self.cosmo.Ode0)  # dim z

    def dVc_dz(self):  # dim z
        return c_light*self.cosmo.comoving_distance(self.z).value**2/(self.cosmo.H0.value*self.E_z())

    def dj2cibprime(self):
        djcen, djsub = self.cib.djc_dlnMh(), self.cib.djsub_dlnMh()
        cosm = (1+self.z)*self.cosmo.comoving_distance(self.z).value**2
        djcen_prime = djcen/(self.tsz.hmf*cosm)
        djsub_prime = djsub/(self.tsz.hmf*cosm)
        return djcen_prime, djsub_prime

    def onehalo(self):
        Kcmb_MJy = [244.1, 371.74, 483.69, 287.45, 58.04, 2.27]
        cl_1h = np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        u_nfw = self.cib.unfw  # dim m,ell,z
        geo = self.dVc_dz()
        dj_c, dj_sub = self.dj2cibprime()
        g_v = self.tsz.g_nu()*1e6*Kcmb_MJy
        y_ell = self.tsz.y_ell_tab()
        dlog10m = np.log10(self.mh[1] / self.mh[0])
        for i in range(len(self.ell)):
            for f in range(self.nfreq):
                a = y_ell[i, :]*((dj_c+dj_sub*u_nfw[:, i, :])*self.cib.cc[:, None, None]*g_v[f] +
                                 (dj_c[f, :]+dj_sub[f, :] *
                                 u_nfw[:, i, :])*self.cib.cc[f]*g_v[:, None, None]) * \
                        self.tsz.hmf
                intgn_mh = intg.simps(a, dx=dlog10m, axis=1, even='avg')
                b = geo*intgn_mh
                intgn_z = intg.simps(b, x=self.z, axis=-1, even='avg')
                cl_1h[f, :, i] = intgn_z  # *T_cmb
        return cl_1h  # final units: Jy^2/sr if Kcmb_jy factor is multiplied
    # otherwise the units are Kcmb*Jy/sr

    def twohalo(self):
        Kcmb_MJy = [244.1, 371.74, 483.69, 287.45, 58.04, 2.27]
        cl_2h = np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        u_nfw = self.cib.unfw  # dim m,ell,z
        geo = self.dVc_dz()*self.cib.Pk_int
        dj_c, dj_sub = self.dj2cibprime()
        g_v = self.tsz.g_nu()*1e6*Kcmb_MJy
        y_ell = self.tsz.y_ell_tab()
        bhmf = self.tsz.biasmz*self.tsz.hmf
        dlog10m = np.log10(self.mh[1] / self.mh[0])
        for i in range(len(self.ell)):
            a1 = y_ell[i, :]*bhmf
            intgn_mh1 = intg.simps(a1, dx=dlog10m, axis=0, even='avg')
            for f in range(self.nfreq):
                a2 = ((dj_c+dj_sub*u_nfw[:, i, :])*g_v[f] +
                      (dj_c[f, :]+dj_sub[f, :]*u_nfw[:, i, :]) *
                      g_v[:, None, None])*bhmf
                intgn_mh2 = intg.simps(a2, dx=dlog10m, axis=1, even='avg')
                b = geo[i, :]*intgn_mh1*intgn_mh2
                intgn_z = intg.simps(b, x=self.z, axis=-1, even='avg')
                cl_2h[f, :, i] = intgn_z  # *T_cmb
        return cl_2h  # final units: Jy^2/sr if Kcmb_jy factor is multiplied
    # otherwise the units are Kcmb*Jy/sr
