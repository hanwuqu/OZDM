# This file only contains rate functions, such as the calculation of rates,
# cross section, the flux of photon is the input of function

# The input photon flux should be a function as Egamma

import numpy as np
from constants import *

'''------------------Photodissociation and Photodetachment-------------'''
def xsection_detach(Egamma):
# Photodetachment cross section
    Etilde = np.sqrt((Egamma - 0.754) / 13.6)
    term = np.power(Etilde, 3) / np.power((0.0555 + np.power(Etilde, 2)), 3)
    return 4.31 * 1e-18 * term

def kHm(J_flux):
    Emin, Emax = 0.76, 13.6 # eV
    def integrand(Egamma):
        return J_flux(Egamma) * xsection_detach(Egamma) / Egamma
    from scipy import integrate
    integral, error = integrate.quad(integrand, Emin, Emax)
    if integral == 0 or np.isnan(integral):
        return 1e-60
    else:
        return integral

def kH2(J_flux):
    Emin, Emax = 11.2, 13.6 # eV
    from scipy import integrate
    integral, error = integrate.quad(J_flux, Emin, Emax)
    J21 = integral / (13.6 - 11.2) * cm_2s_1toJ21
    if J21 == 0 or np.isnan(J21):
        return 1e-60
    else:
        return 1.39* 1e-12 * J21 

'''-----------------Photoionization and Photoheating----------------'''
ionization_table = {'H':[13.6, 0.4298, 5.475e4, 32.88, 2.963, 0, 0, 0], 'He':[24.59, -13.61, 9.492e2, 1.469, 3.188, 2.039, 44.34, 0.136], 'Hep':[54.42, 1.72, 1.369e4, 32.88, 2.963, 0, 0, 0]}

def sigma_ion_element(element, Egamma):
# Unit in Mb
    ion_table = ionization_table[element]
    Eth, E0, sigma0, ya, P, yw, y0, y1 = ion_table
    if Egamma > Eth:
        x = Egamma / E0 -y0
        y = np.sqrt(np.power(x, 2) + np.power(y1, 2))
        Fterm1 = np.power(x - 1, 2) + np.power(yw, 2)
        Fterm2 = np.power(y, 0.5 * P - 5.5)
        Fterm3 = np.power(1 + np.sqrt(y / ya), -P)
        F = Fterm1 * Fterm2 * Fterm3
        return sigma0 * F
    else:
        return 0

def kion(J_flux, Egammamax):
# Return photoionization rate for elements, unit in s^-1
    element_list = ['H', 'He', 'Hep']
    result_list = []
    for i in element_list:
        def integrand(Egamma):
            if Egamma > Egammamax:
                return 0
            else:
            	return J_flux(Egamma) * sigma_ion_element(i, Egamma) / Egamma
        Egammamin = ionization_table[i][0]
        from scipy import integrate
        result, err = integrate.quad(integrand, Egammamin, Egammamax)
        result_list.append(result * 1e-18)
    result_list = [1e-60 if np.isnan(x) else x for x in result_list]
    return result_list

def kheat(J_flux, Egammamax):
# Return photoheating rate for elements, unit in eV*s^-1
    element_list = ['H', 'He', 'Hep']
    result_list = []
    for i in element_list:
        Egammamin = ionization_table[i][0]
        def integrand(Egamma):
            if Egamma > Egammamax:
                return 0
            else:
                return J_flux(Egamma) * sigma_ion_element(i, Egamma) * (Egamma - Egammamin)/ Egamma
        from scipy import integrate
        result, err = integrate.quad(integrand, Egammamin, Egammamax)
        result_list.append(result * 1e-18)
    result_list = [1e-60 if np.isnan(x) else x for x in result_list]
    return result_list

'''------------------Photoionization of Molecular Hydrogen-----------------'''
# Photoionization Cross Section of Molecular Hydrogen
def sigma_H2_low(Egamma):
# Unit of Egamma is eV, output unit is in barn
    x = Egamma / 15.4
    term0 = -37.895
    term1 = 99.723 * x
    term2 = -87.227 * np.power(x, 2)
    term3 = 25.400 * np.power(x, 3)
    sum = term0 + term1 + term2 + term3
    return 1e8 * sum

def sigma_H2_mid(Egamma):
# Unit of Egamma is eV, output unit is barn
    x = Egamma / 15.4
    s = 0.252
    term0 = 0.071 * np.power(x, -s)
    term1 = -0.673 * np.power(x, -(s + 1))
    term2 = 1.977 * np.power(x, -(s + 2))
    term3 = -0.692 * np.power(x, -(s + 3))
    sum = term0 + term1 + term2 + term3
    return 2e7 * sum

def sigma_H2_high(Egamma):
# Unit of Egamma is eV, output unit is barn
    Egamma_keV = Egamma / 1e3
    x = Egamma / 15.4
    term0 = 1
    term1 = -2.003 / np.power(x, 1 / 2)
    term2 = -4.806 / x
    term3 = 50.577 / np.power(x, 3 / 2)
    term4 = -171.044 / np.power(x, 2)
    term5 = 231.608 / np.power(x, 5 / 2)
    term6 = -81.885 / np.power(x, 3)
    sum = term0 + term1 + term2 + term3 + term4 + term5 + term6
    return 45.57 * sum / np.power(Egamma_keV, 7 / 5)

def sigma_H2(Egamma):
# Output unit in cm^2
    if Egamma <= 15.4:
        return 0
    elif Egamma > 15.4 and Egamma <= 18:
        return sigma_H2_low(Egamma) * 1e-24
    elif Egamma > 18 and Egamma <=85:
        return sigma_H2_mid(Egamma) * 1e-24
    else:
        return sigma_H2_high(Egamma) * 1e-24

def k29(Jflux, Egammamax):
# Unit of flux is cm^-2 s^-1, Egammamax is eV, Output unit is s^-1
    Egammamin = 15.4
    def integrand(Egamma):
        if Egamma > Egammamax:
            return 0
        else:
            return Jflux(Egamma) * sigma_H2(Egamma) / Egamma
    from scipy import integrate
    result, err = integrate.quad(integrand, Egammamin, Egammamax)
    return result if result != 0 else 1e-60

'''-----------------Photodissociation of Molecular Hydrogen-----------------'''
# k28, H2II + gamma \rightwrrow HI + HII
def sigma_H2_28_low(Egamma):
# Input unit in eV, check for the output unit
    term0 = -40.97
    term1 = 6.03 * Egamma
    term2 = -0.504 * np.power(Egamma, 2)
    term3 = 1.387e-2 * np.power(Egamma, 3)
    sum = term0 + term1 + term2 + term3
    return np.power(10, sum)

def sigma_H2_28_high(Egamma):
    term0 = -30.26
    term1 = 2.79 * Egamma
    term2 = -0.184 * np.power(Egamma, 2)
    term3 = 3.535e-3 * np.power(Egamma, 3)
    sum = term0 + term1 + term2 + term3
    return np.power(10, sum)

def sigma_H2_28(Egamma):
# H2II + gamma \rightwrrow HI + HII, Input unit is eV, output unit is cm^2
    if Egamma > 2.65 and Egamma <=11.27:
        return sigma_H2_28_low(Egamma)
    elif Egamma > 11.27 and Egamma <= 21.0:
        return sigma_H2_28_high(Egamma)
    else:
        return 0

# k30, HII + gamma \rightarrow 2HII + e
def sigma_H2_30(Egamma):
# HII + gamma \rightarrow 2HII + e, input unit is eV, output unit is cm^2
    if Egamma < 30 or Egamma > 100:
        return 0
    else:
        term0 = -16.926
        term1 = -4.528e-2 * Egamma
        term2 = 2.238e-4 * np.power(Egamma, 2)
        term3 = 4.245e-7 * np.power(Egamma, 3)
        sum = term0 + term1 + term2 + term3
        return np.power(10, sum)

def k28(Jflux, Egammamax):
# Unit of flux is cm^-2 s^-1, Egammamax is eV, Output unit is s^-1
    Egammamin = 2.65
    def integrand(Egamma):
        if Egamma > Egammamax:
            return 0
        else:
            return Jflux(Egamma) * sigma_H2_28(Egamma) / Egamma
    from scipy import integrate
    result, err = integrate.quad(integrand, Egammamin, Egammamax)
    return result if result != 0 else 1e-60

def k30(Jflux, Egammamax):
# Unit of flux is cm^-2 s^-1, Egammamax is eV, Output unit is s^-1
    Egammamin = 30
    def integrand(Egamma):
        if Egamma > Egammamax:
            return 0
        else:
            return Jflux(Egamma) * sigma_H2_30(Egamma) / Egamma
    from scipy import integrate
    result, err = integrate.quad(integrand, Egammamin, Egammamax)
    return result if result != 0 else 1e-60
