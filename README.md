#  halomodel_cib_tsz_cibxtsz

Computes the CIB, tSZ, and CIB-tSZ correlation power spectrum using a newly developed halo model

* This code is based on a newly developed halo model for the CIB and CIB-tSZ correlation with just four physical model parameters.
* These models can be used in the CMB data analysis to account for the CIB, tSZ, and CIBxtSZ foregrounds instead of just fitting them with a power law or with some templates.
* hmf_unfw_bias.py code is used to calculate the halo mass function, Fourier transform of the NFW profile, and the halo bias

Clone the repository, then compute the power spectra with:
```
python driver_cell.py
```
or look at ```driver_cell.ipynb```.

Hope you find this code useful! Do not hesitate to contact me with any questions: abhishek.maniyar@nyu.edu

