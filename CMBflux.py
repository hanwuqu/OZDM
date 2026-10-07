import numpy as np
from constants import c

def J_CMB(rs, Egamma):
# The flux of the CMB photon at redshift rs, the unit of Egamma is eV, output unit is cm^-2s^-1
    ccm = c * 1e2 # speed of light in cm/s
    hbareV = 6.582e-16 #eV * s
    kB = 8.617330e-5 #eV/K
    z = rs - 1
    T0 = 2.72548
    factor1 = np.power(Egamma, 2) / np.power(np.pi, 2) / np.power(hbareV * ccm, 3)
    x = Egamma / T0 / rs / kB
    factor2 = 1 / np.power(np.exp(x) - 1, 1)
    result = factor1 * factor2 * ccm / 4
    return result
