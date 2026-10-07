# Photons from the background radiation, expression from 1912.09486

# photon spectrum from DM annihilation
import numpy as np
from constants import *

def dNdE(mchi, Egamma, z):
# mchi, Egamma unit in eV
    factor = mchi / Egamma - 1
    if z - factor != 0:
        return 0
    else:
        return 2 / Egamma * np.inf

def Hubble(z):
    rs = 1 + z
    hubble = np.power(rs, 3) * omega_m + np.power(rs, 4) * omega_rad + omega_lambda
    return H0 * np.sqrt(hubble)

def z_int(mchi, Egamma, zup):
# Integrate term for z, result term in unit eV^-1 s
    integrand = lambda z: np.power(1 + z, 3) / Hubble(z) * dNdE(mchi, Egamma, z)
    from scipy import integrate
    result, err = integrate.quad(integrand, 0, zup)
    return result

def dPhidE(mchi, xsection, Egamma, zup):
# mchi, Egamma unit in eV, xsection in cm^3s^-1
# Output unit: eV^-1 cm^-2 s^-1
    h = 0.6736
    rho_crit_eV = 1.05371e4*(h**2) # unit in eV*cm^-3
#    print(rho_crit_eV)
    factor = 1 / 4 / np.pi * np.power(omega_DM, 2) * np.power(rho_crit_eV, 2) * xsection / np.power(mchi, 2)
#    z = mchi / Egamma - 1 correct at 6.Feb.2025
    z = mchi * (1 + zup) / Egamma
#    unit_convert = 3e10 * np.power(1.78266192e-33, -2)
    unit_convert = 3e10
    if z >= zup and z <= 1100:
        return factor * np.power(1 + z, 3) / Hubble(z) * 2 / Egamma * unit_convert
    else:
        return 0

#def J_background_ann(mchi, xsection, Egammamax, zup):
# output unit in cm^-2 s^-1    
#    integrand = lambda Egamma: dPhidE(mchi, xsection, Egamma, zup)
#    from scipy import integrate
#    result, err = integrate.quad(integrand, 0, Egammamax)
#    result = dPhidE(mchi, xsection, Egammamax, zup) * Egammamax
#    return result * np.power(1 + zup, 3)

def J_background_ann(mchi, xsection, Egamma, rsstep, type='Majorana', thermal='Non', fgamma=1):
# mchi, Egamma unit in eV, xsection in cm^3s^-1
# Output unit: eV^-1 cm^-2 s^-1
    h = 0.6736
    rho_crit_eV = 1.05371e4*(h**2) # unit in eV*cm^-3
#    print(rho_crit_eV)
    if type == 'Majorana':
        p = 1
    else:
        p = 2
    if thermal == 'Non':
        fdm = 1
    else:
        fdm = 3*1e26 / xsection
    factor = 1 / 4 / np.pi * np.power(omega_DM, 2) * np.power(rho_crit_eV, 2) * xsection / np.power(mchi, 2)
    factor = factor * fgamma * fdm**2 / np.power(2, p)
    def integrand0(z):
        expression = np.power(rsstep, 2) * 6 * Egamma / mchi * np.power(1 + z, 4) / Hubble(z)
        return expression
    def integrand1(z):
        expression = rsstep * 6 * np.power(Egamma, 2) / np.power(mchi, 2) * np.power(1 + z, 5) / Hubble(z)
        return expression
    def integrand(z):
        return integrand0(z) + integrand1(z)
    zup = mchi * rsstep / Egamma - 1
    zstep = rsstep - 1
    if zup <= zstep:
        return 0
    else:
        if zup >= 1100:
 # If the photon from the background before the recombination, it will just heat the plasma 
            zup = 1100
        else:
            zup = zup
        unit_convert = 3e10
        from scipy import integrate
        result, err = integrate.quad(integrand, zstep, zup)
        return result * factor * unit_convert



#tt = list(J_background_ann(185, 1e-33, i, 100) for i in np.linspace(1e-6, 1000, 100))
#print(rho_crit)

# Background DM Decay Spectrum
def decay_factor(mchi, tau):
# The input unit: mdm - eV, tau - s
    h = 0.6736
    rho_crit = 1.05371e4*(h**2) # unit in eV*cm^-3
    factor = 1 / 4 / np.pi * omega_DM * (rho_crit) / tau / mchi
    return factor

def integrand_box(rs, rs_step, Egamma):
    z = rs - 1
    z_step = rs_step - 1
    integrand = np.power(rs_step, 3) / Hubble(z) / Egamma
    return integrand

def integrand_parabola(rs, rs_step, mchi, Egamma):
    z = rs - 1
    z_step = rs_step - 1
    integrand1 = 6 * Egamma * rs * np.power(rs_step, 2)/ Hubble(z) / mchi
    integrand2 = -6 * Egamma * rs_step * np.power(rs, 2) / Hubble(z) / rs_step * Egamma / np.power(mchi, 2)
    integrand = integrand1 + integrand2
    return integrand

def zupper_decay(rs_step, mchi, Egamma):
    zupper = rs_step * mchi / Egamma - 1
    return zupper

def dPhidE_decay(rs_step, mchi, tau, Egamma, type='box'):
# Input Unit: mchi-eV, Egamma-eV, tau-s
# Output Unit: eV^-1*cm_2*s^-1
#    unit_convert = 5.3479749e-23
    unit_convert = 3e10
    z_step = rs_step - 1
    if type == 'box':
        def integrand(z):
            return integrand_box(z + 1, rs_step, Egamma) * Egamma* decay_factor(mchi, tau) * unit_convert
    else:
        def integrand(z):
            return integrand_parabola(z + 1, rs_step, mchi, Egamma) * decay_factor(mchi, tau) * unit_convert
    from scipy import integrate
    if zupper_decay(rs_step, mchi, Egamma) > z_step:
        if zupper_decay(rs_step, mchi, Egamma) <= 1100:
            result, err = integrate.quad(integrand, z_step, zupper_decay(rs_step, mchi, Egamma))
        else:
            result, err = integrate.quad(integrand, z_step, 1100)
        return result
    else:
        return 0

def J_background_decay(rs_step, mchi, tau, Egamma, type='box'):
    result = dPhidE_decay(rs_step, mchi, tau, Egamma, type)
    return result 
