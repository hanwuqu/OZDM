# One Zone Model Halo Functions
from constants import *
import numpy as np

def halo_density(rs, rs_vir, rho_current=rho_crit):
    A = rs_vir / rs
    rho = rho_current * rs**3 * np.exp(1.9*A/(1-0.75*A**2))
    return rho

def vir_density(rs, rho_current=rho_crit):
    return rho_current*18*np.pi**2*rs**3

# Find the 'real' virial redshift, \rho_halo = \rho_vir
def equal(rs, rs_vir, rho_current=rho_crit):
    return halo_density(rs, rs_vir, rho_current) - vir_density(rs, rho_current)

def real_vir(rs_vir, rho_current=rho_crit):
    from scipy.optimize import fsolve
    ini = 2 * rs_vir
    root = fsolve(equal, ini, args=(rs_vir, rho_current))
    return root[0]

def one_zone_density(rs, rs_vir, rho_current=rho_crit):
    vir_real = real_vir(rs_vir, rho_current)
    if rs >= vir_real:
        return halo_density(rs, rs_vir, rho_current)
    else:
        return halo_density(vir_real, rs_vir, rho_current)

# It is kind of dangerous to use this function directly, if the temperature
# of halo arrives at T_vir early than redshift arrives at z_vir, the density
# of halo keeps constant. Neec to be updated.

def nH(rs, rs_vir, nH=nH0):
    A = rs_vir / rs
    if rs > real_vir(rs_vir):
        nHA = nH * rs**3 * np.exp(1.9*A/(1-0.75*A**2))
    else:
        A_fix = rs_vir / real_vir(rs_vir)
        nHA = nH * real_vir(rs_vir)**3 * np.exp(1.9*A_fix/(1-0.75*A_fix**2))
    return nHA

# Virial temperature, unit in K
def Tvir(Mhalo, rs_vir):
# Mhalo is the mass of halo in solar mass
    factor1 = np.power(omega_m * h**2 / 0.14, 1/3)
    mass = np.power(Mhalo / 1e4, 2/3)
    rsterm = rs_vir / 100
    return 224 * factor1 * mass * rsterm

# Adiabatic internal energy
def internal_energy(rho_old, rho_new, gamma):
    return np.power(rho_new / rho_old, (gamma - 1))
