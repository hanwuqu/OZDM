# Dark Matter halo from Gizmo simulation
# photon spectrum

import numpy as np
from files import read_data
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

# Mean density data
def density_decay_mean(time):
    from scipy import interpolate
    path = '/Users/hanwu/Desktop/SMBH/code/DM_One_Zone/input/density_mean.txt'
    time_list, rho_bar = read_data(path)
    radius = 100 # unit: pc
    integral = list(i * radius for i in rho_bar)
    term = interpolate.interp1d(time_list, integral)
    return term(time).item()

def density_annihilation_mean(time):
    from scipy import interpolate
    path = '/Users/hanwu/Desktop/SMBH/code/DM_One_Zone/input/density_mean.txt'
    time_list, rho_bar = read_data(path)
    if time > max(time_list) or time < min(time_list):
        return 0
    else:
        radius = 100 # unit: pc
        integral = list(i**2 * radius for i in rho_bar)
        term = interpolate.interp1d(time_list, integral)
        return term(time).item()

def J_decay(mdm, taudm, Egamma, time, shape='box', fdm=1, fgamma=1):
# unit: mdm-eV, taudm-s, Egamma-eV, time-Myr
# output unit: J-> cm^-2*s^-1
    factor = fdm * fgamma / mdm / taudm
    density_term = density_decay_mean(time)
    if shape == 'box':
        EdNdE = ps_box(Egamma, mdm)
    else:
        EdNdE = ps_parabola(Egamma, mdm)
    return factor * density_term * EdNdE * Jdecayunit

def J_annihilation(mdm, xsection, Egamma, time, type='Majorana', thermal='Non', fgamma=1):
# unit: mdm-eV, xsection-cm^3*s^-1, Egamma-eV, time-Myr
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
    density_term = density_annihilation_mean(time)
    return factor * EdNdE * density_term * Jannihilationunit
