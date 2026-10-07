''' In this file, the flux of photon from DM halo will be calculated'''
''' Input is the density of halo in unit M_\odot Mpc^-3'''
''' Output unit in cm^-2*s^-1 '''
import numpy as np
from constants import *

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

def ps_parabola(Egamma, mdm):
    x = Egamma / mdm
    return 6 * (x - np.power(x, 2)) * step(1 - x)

def ps_box(Egamma, mdm):
    x = Egamma / mdm
    return step(1 - x)

def J_decay(mdm, taudm, Egamma, radius, density, shape='box', fdm=1, fgamma=1):
# unit: mdm-eV, taudm-s, Egamma-eV, radius-pc, density-M_\odot Mpc^-3
# output unit: J-> cm^-2*s^-1
    factor = fdm * fgamma / mdm / taudm
    radius = radius * 3.086e18# From pc to cm
#    density = density * 1e-18 # From M_odot Mpc^-3 to M_odot pc^-3
    density = density * 1.12e66 * np.power(3.086e24, -3)
    density_term = density * radius
    if shape == 'box':
        EdNdE = ps_box(Egamma, mdm)
    else:
        EdNdE = ps_parabola(Egamma, mdm)
#    return factor * density_term * EdNdE * Jdecayunit
    return factor * density_term * EdNdE
def J_annihilation(mdm, xsection, Egamma, radius, density,  type='Majorana', thermal='Non', fgamma=1):
# unit: mdm-eV, xsection-cm^3*s^-1, Egamma-eV, radius-pc, density-M_\odot Mpc^-#3
# output unit: J-> cm^-2*s^-1
    if type == 'Majorana':
        p = 1
    else:
        p = 2
    if thermal == 'Non':
        fdm = 1
    else:
        fdm = 3*1e26 / xsection
    factor = fgamma * fdm**2 * xsection / mdm**2 / np.power(2, p)
    EdNdE = ps_parabola(Egamma, mdm)
    radius = radius * 3.086e18# From pc to cm
#    density = density * 1e-18 # From M_odot Mpc^-3 to M_odot pc^-3
    density = density * 1.12e66 * np.power(3.086e24, -3)
    density_term = np.power(density, 2) * radius
#    density_term = np.power(density, 2) 
#    return factor * EdNdE * density_term * Jannihilationunit
    return factor * EdNdE * density_term
