# This file contains functions for self-shielding, for photoionization
# photodetachment, self-shielding comes from the 3D Gaussian method;
# for photondissociation, it comes from the expression in Avi's 
# paper. There are two part of self-shielding, from the background and from 
# the halo, each of them includes dark matter decay and annihilation.

import numpy as np
from background_radiation import Hubble
from constants import omega_DM
#----------------------Self-Shielding of Flux from Halo-----------------------#

def L(r, rp, cth, Rcloud):
# Distance between point \vec{r} to \vec{r'}, the unit of r, rp and Rcloud should be same, let's say pc, cth\in[-1, 1]
# Output unit in cm
    pctocm = 3.086e18
    r = r * pctocm
    rp = rp * pctocm
    Rcloud = Rcloud * pctocm
    if np.power(r, 2) + np.power(rp, 2) - 2 * r * rp * cth > 0:
        rp_r = np.sqrt(np.power(r, 2) + np.power(rp, 2) - 2 * r * rp * cth)
    else:
        rp_r = 0
    if rp <= Rcloud:
        distance = rp_r
        return max(0, distance)
    else:
        s = -r * cth + np.sqrt(np.power(r * cth, 2) + (np.power(Rcloud, 2) - np.power(r, 2))) # path inside the cloud
        l = rp_r - s # path outside the cloud
        return max(s, 0), max(l, 0)

def optical_depth(r, rp, cth, Rcloud, ncloud, nbg, sigma):
# Optical depth calculation, number density in unit cm^-3, cross section in unit cm^2, so L in unit cm
# the unit of r, rp and Rcloud are pc, output quantity is dimensionless
    if rp <= Rcloud:
        distance = L(r, rp, cth, Rcloud)
        tau = ncloud * sigma * distance
    else:
        s, l = L(r, rp, cth, Rcloud)
        tau = s * ncloud + l * nbg
        tau = tau * sigma 
    return tau

def optical_depth_total(r, rp, cth, Rcloud, ncloud, nbg, Egamma):
    nHIc, nHeIc, nHeIIc, nHmc = ncloud
    nHIb, nHeIb, nHeIIb, nHmb = nbg
    from DM_rate import xsection_detach, sigma_ion_element
    sigma_detachment = xsection_detach(Egamma)
    sigma_HI_ion = sigma_ion_element('H', Egamma) * 1e-18 # Mb to cm^2
    sigma_HeI_ion = sigma_ion_element('He', Egamma) * 1e-18 # Mb to cm^2
    sigma_HeII_ion = sigma_ion_element('Hep', Egamma) * 1e-18 # Mb to cm^2
    optical_depth_detachment = optical_depth(r, rp, cth, Rcloud, nHmc, nHmb, sigma_detachment)
    optical_depth_HI_ion = optical_depth(r, rp, cth, Rcloud, nHIc, nHIb, sigma_HI_ion)
    optical_depth_HeI_ion = optical_depth(r, rp, cth, Rcloud, nHeIc, nHeIb, sigma_HeI_ion)
    optical_depth_HeII_ion = optical_depth(r, rp, cth, Rcloud, nHeIIc, nHeIIb, sigma_HeII_ion)
#    print(optical_depth_detachment, optical_depth_HI_ion, optical_depth_HeI_ion, optical_depth_HeII_ion)
    total = optical_depth_detachment + optical_depth_HI_ion + optical_depth_HeI_ion + optical_depth_HeII_ion
    return total

# Dark Matter Decay

def Jeff_decay_halo(mdm, taudm, Egamma, density, Rcloud, Rhalo, ncloud, nbg, shape='box', fdm=1, fgamma=1):
# Unit of Rcloud and Rhalo should be pc, number density- cm^-3, density eV cm^-3, sigma-cm^2, taudm-s, mdm-eV, Egamma-eV
# Output unit is cm^{-2} * s^{-1}
    import vegas
    pctocm = 3.086e18
    from photon_flux import ps_box, ps_parabola
    from scipy.integrate import nquad
    if shape == 'box':
        EdNdE = ps_box(Egamma, mdm)
    else:
        EdNdE = ps_parabola(Egamma, mdm)
    factor = 3 / 4 / np.power(Rcloud, 3)
    factor = factor * fgamma * fdm * density / taudm / mdm
    # Define the integrand in scaled coordinates
    def integrand(x):
        cth = 2 * x[0] - 1            # Map from [0,1] to [-1,1]
        rp = x[1] * Rhalo             # Map from [0,1] to [0,Rhalo]
        r = x[2] * Rcloud             # Map from [0,1] to [0,Rcloud]

        rp_r_sq = r**2 + rp**2 - 2 * r * rp * cth
        if rp_r_sq < 1e-5:
            rp_r_sq = 1e-5

        tau = optical_depth_total(r, rp, cth, Rcloud, ncloud, nbg, Egamma)
        if tau > 100:
            return 0.0

        return (r**2 * rp**2 / rp_r_sq) * np.exp(-tau) * 2 * Rhalo * Rcloud  # Include Jacobian

    integ = vegas.Integrator([[0, 1], [0, 1], [0, 1]])
    integ(integrand, nitn=5, neval=1_000)  # Warm-up iterations
    result = integ(integrand, nitn=10, neval=10_000)  # Actual integration

    flux = factor * result.mean * EdNdE * pctocm
    return flux if flux != 0 else 1e-60

# Dark Matter Annihilation

def Jeff_annihilation_halo(mdm, xsection, Egamma, density, Rcloud, Rhalo, ncloud, nbg, type='Majorana', thermal='Non', fgamma=1):
# Unit of Rcloud and Rhalo should be pc, number density- cm^-3, density-eV cm^-3, sigma-cm^2, 
# xsection-cm^3*s^-1, mdm-eV, Egamma-eV
# Output unit is cm^{-2} * s^{-1}
    import vegas
    pctocm = 3.086e18
    from photon_flux import ps_parabola
    from scipy.integrate import quad
    EdNdE = ps_parabola(Egamma, mdm)
    if type == 'Majorana':
        p = 1
    else:
        p = 2
    if thermal == 'Non':
        fdm = 1
    else:
        fdm = 3 * 1e26 / xsection
    factor = 3 / 4 / np.power(Rcloud, 3)
    factor = factor * fgamma * fdm**2 * xsection / np.power(mdm, 2) / np.power(2, p) * np.power(density, 2)
    # Define the integrand in scaled coordinates
    def integrand(x):
        cth = 2 * x[0] - 1            # Map from [0,1] to [-1,1]
        rp = x[1] * Rhalo             # Map from [0,1] to [0,Rhalo]
        r = x[2] * Rcloud             # Map from [0,1] to [0,Rcloud]

        rp_r_sq = r**2 + rp**2 - 2 * r * rp * cth
        if rp_r_sq < 1e-5:
            rp_r_sq = 1e-5

        tau = optical_depth_total(r, rp, cth, Rcloud, ncloud, nbg, Egamma)
        if tau > 100:
            return 0.0

        return (r**2 * rp**2 / rp_r_sq) * np.exp(-tau) * 2 * Rhalo * Rcloud  # Include Jacobian

    integ = vegas.Integrator([[0, 1], [0, 1], [0, 1]])
    integ(integrand, nitn=5, neval=1_000)  # Warm-up iterations
    result = integ(integrand, nitn=10, neval=10_000)  # Actual integration

    flux = factor * result.mean * EdNdE * pctocm
    return flux if flux != 0 else 1e-60

#----------------Self-Shielding of the Flux from the Background-------------# 
def optical_depth_background_total(rs, rsobs, nbg, Egamma):
# This function returns the dimensionless optical depth
# nbg-cm^-3, sigma-cm^2
# Remember the redshift of cross section
    z = rs - 1
    zobs = rsobs - 1
    c = 3e10 # speed of light in cm/s
    nHIb, nHeIb, nHeIIb, nHmb = nbg
    from DM_rate import xsection_detach, sigma_ion_element
    def cross_sections(Eg):
    # Already done the unit convert, don't worry, output unit in cm^2
        return xsection_detach(Eg), sigma_ion_element('H', Eg) * 1e-18, sigma_ion_element('He', Eg) * 1e-18, sigma_ion_element('Hep', Eg) * 1e-18
    def integrand(zp):
        Egamma_rs = Egamma * (1 + zp) / rsobs
        sdetach, sHI, sHeI, sHeII = cross_sections(Egamma_rs)
        term_Hm = nHmb * sdetach
        term_HI = nHIb * sHI
        term_HeI = nHeIb * sHeI
        term_HeII = nHeIIb * sHeII
        integrand_expression = (term_Hm + term_HI + term_HeI + term_HeII) / (1 + zp) / Hubble(zp)
        return integrand_expression
    from scipy.integrate import quad
    result, err = quad(integrand, zobs, z)
    return result * c

def zupper(mdm, Egamma, rsobs):
    result = mdm * rsobs / Egamma - 1
    if result > 1100:
        return 1100
    else:
        return result

# Dark Matter Decay

def Jeff_decay_background(mdm, taudm, Egamma, nbg, rsobs, shape='box', fdm=1, fgamma=1):
    unit_convert = 3e10
    zobs = rsobs - 1
    if type == 'box':
        def integrand(z):
            if optical_depth_background_total((1 + z), rsobs, nbg, Egamma) > 400:
                return 0
            else:
                return integrand_box(z + 1, rsobs, Egamma) * decay_factor(mdm, taudm) * unit_convert * fdm * fgamma * np.exp(-optical_depth_background_total((1 + z), rsobs, nbg, Egamma))
    else:
        def integrand(z):
            if optical_depth_background_total((1 + z), rsobs, nbg, Egamma) > 400:
                return 0
            else:
                return integrand_parabola(z + 1, rsobs, mdm, Egamma) * decay_factor(mdm, taudm) * unit_convert * fdm * fgamma * np.exp(-optical_depth_background_total((1 + z), rsobs, nbg, Egamma))
    from scipy.integrate import quad
    if zupper(mdm, Egamma, rsobs) <= zobs:
        flux = 0
    else:
        flux, err = quad(integrand, zobs, zupper(mdm, Egamma, rsobs))
    return flux if flux != 0 else 1e-60

# Dark Matter Annihilation

def Jeff_annihilation_background(mdm, xsection, Egamma, nbg, rsobs, type='Majorana', thermal='Non', fgamma=1):
# Unit of Rcloud and Rhalo should be pc, number density- cm^-3, density-eV cm^-3, sigma-cm^2, 
# xsection-cm^3*s^-1, mdm-eV, Egamma-eV
# Output unit is cm^{-2} * s^{-1}
    unit_convert = 3e10
    h = 0.6736
    rho_crit_eV = 1.05371e4*(h**2) # unit in eV*cm^-3
    zobs = rsobs - 1
    factor = 1 / 4 / np.pi * np.power(omega_DM, 2) * np.power(rho_crit_eV, 2) * xsection / np.power(mdm, 2)
    def integrand0(z):
        expression = np.power(rsobs, 2) * 6 * Egamma / mdm * np.power(1 + z, 4) / Hubble(z)
        return expression
    def integrand1(z):
        expression = rsobs * 6 * np.power(Egamma, 2) / np.power(mdm, 2) * np.power(1 + z, 5) / Hubble(z)
        return expression
    def integrand(z):
        if optical_depth_background_total((1 + z), rsobs, nbg, Egamma) > 400:
            return 0
        else:
            return (integrand0(z) + integrand1(z)) * np.exp(-optical_depth_background_total((1 + z), rsobs, nbg, Egamma))
    from scipy.integrate import quad
    if zupper(mdm, Egamma, rsobs) <= zobs:
        flux = 0
    else:
        flux, err = quad(integrand, zobs, zupper(mdm, Egamma, rsobs))
    return flux * factor * unit_convert if flux != 0 else 1e-60

#--------------------Shielding Fraction of Photodissociation-----------------
def jeans_length(temperature, rho, mu):
# Input unit: temperature-K, rho-g\cdotcm^-3, mu is dimensionless mean molecular weight
# Output unit in cm
    G_cgs = 6.673015e-8 # Gravitational constant in cm^3g^-1s^-2
    kB_cgs = 1.3806488e-16 # Boltzman constant in g cm^2s^-2K^-1
    mp = 1.67262192e-24 # proton mass in g
    expression = 5 * np.pi * kB_cgs / 3 / G_cgs / mp / mu / rho
    return np.sqrt(expression)

def Doppler_broadening(temperature):
# Unit of temperature is K
# Output unit in cm\cdot s^-1
    kB_cgs = 1.3806488e-16 # Boltzman constant in g cm^2s^-2K^-1
    mp = 1.67262192e-24 # proton mass in g
    return np.sqrt(2 * kB_cgs * temperature / 2 / mp)

def shielding_fraction(temperature, rho, mu, nH2):
# Input unit: temperature-K, rho-g\cdot cm^-3, mu is dimensionless, nH2-cm^-3
# This is a fitting function, output is dimensionless
    NH2 = nH2 * jeans_length(temperature, rho, mu)
    x = NH2 / 5e14
    b5 = Doppler_broadening(600) / 1e5
    term1 = 0.965 / np.power(1 + x / b5, 1.1)
    term2 = 0.035 / np.power(1 + x, 0.5)
    term3 = np.exp(-8.5e-4 * np.power(1 + x, 0.5))
    return term1 + term2 * term3

#-------------------Photoionization Self Shielding--------------------
#Some of expressions here comes from Joop's paper
def ssh_factor(nHI, n0, alpha1, alpha2, beta, f):
# This function returns the fraction of shielded photons for the reaction rate
# Input unit: nHI, n0-cm^-3, a1, a2, b and f are dimensionless
    n_term = nHI / n0
    term0 = (1 - f) * np.power(1 + np.power(n_term , beta), alpha1)
    term1 = f * np.power(1 + n_term, alpha2)
    return term0 + term1

def nssh_deduce(sigmabar, temperature, ratio):
# I deduced this expression, here ratio is nHI/nH
    factor1 = np.power(2.5e-4, 2)
    factor2 = 2.49e-18 # cm^2
    T4 = temperature / 1e4
    h = 0.6736
    omega_m = 0.3153
    omega_baryon = 0.02237/(h**2)
    fg = omega_baryon / omega_m
    term0 = np.power(sigmabar / factor2, -2)
    term1 = np.power(T4, -1)
    term2 = np.power(ratio, -2)
    term3 = np.power(fg / 0.17, -1)
    return factor1 * term0 * term1 * term2 * term3

def mean_sigma_HI(J_flux, Egammamax, kionHI):
    from DM_rate import kion
    EgammaHI = 13.6
    EgammaHeII = 54.42
    if Egammamax <= EgammaHeII:
        Emax = Egammamax
    else:
        Emax = EgammaHeII
    numerator = kionHI
    from scipy import integrate
    def integrand(Egamma):
        return J_flux(Egamma) / Egamma
    denominator, err = integrate.quad(integrand, EgammaHI, Emax)
    if denominator is np.nan:
        denominator = 1e-60 # Some tiny number
    return numerator / denominator

#----------------------Total Self Shielding Function------------------
def flux_dict(halo_path):
    path = halo_path + '/J_flux.h5'
    import h5py
    flux_data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():  # iterate over all group names
            grp = f[key]
            Egamma = grp["Egamma"][:]
            Jflux  = grp["Jflux"][:]
            flux_data[key] = (Egamma, Jflux)
    return flux_data

def ssh_total(folder_path, Egammamax, rs, parameter):
# This function takes files of raw data as well as photon flux as input
# returns two fractions of self shielding for photodissociation and
# photoionization, Egammamax in unit eV, but maybe not very important...
    flux_data = flux_dict(folder_path)
    def J_gamma(Egamma):
        from scipy.interpolate import interp1d
        Egammalst, Jfluxlst = flux_data[str(round(rs, 1))]
        result = interp1d(Egammalst, Jfluxlst)
        return result(Egamma)
    from DM_rate import kion
    kionHI = kion(J_gamma, 100)[0]
    data_path = folder_path + '/raw_data.txt'
    from files import last_line
    data = last_line(data_path)
    density = data[1]
    temperature = data[2]
    HI = data[3]
    H2I = data[9]
    mu = data[-1]
    nHI = HI / 1.67262192e-24
    nH2I = H2I / 1.67262192e-24 / 2
    nH = density / 1.67262192e-24 / mu * 0.76
    alpha = nHI / nH
    fH2 = shielding_fraction(temperature, density, mu, nH2I)
    if Egammamax <= 13.6:
        fion = 1e-60
        return fion, fH2
    else:
        sigma_bar = mean_sigma_HI(J_gamma, Egammamax, kionHI)
        n0 = nssh_deduce(sigma_bar, temperature, alpha)
# For the fitting parameters, I will take those from 1707.06993 at z=10
        alpha1, alpha2, beta, f = parameter
        fion = ssh_factor(nH, n0, alpha1, alpha2, beta, f)
#    fH2 = shielding_fraction(temperature, density, mu, nH2I)
        return fion, fH2
