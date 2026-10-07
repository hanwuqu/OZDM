import numpy as np
from constants import *
from DMhalo import ini_radius_deduce, DMhalo_profile, rho_cz

# In this file, we will get the top-hat halo evoluation firstly, then store
# the redshift and radius, density in a .txt file. Then use this file as
# input of artifical NFW profile functions.

def top_hat_halo(Mcloud, datapath):
# In the code, call this function at first to generate the halo data file
# Unit of input is solar mass
    fb = omega_baryon / omega_m
    Mhalo = Mcloud / fb
    rhalo_ini = ini_radius_deduce(1101, Mhalo, 0.02)
    h_rs, h_radius, h_profile, h_rsvir, h_rhovir, h_overdensity = DMhalo_profile(1101, 1, rhalo_ini, Mhalo, 0.02)
    txt_path = datapath + '/top_hat_halo.txt'
    with open(txt_path, 'w') as txt:
        for i in range(len(h_rs)):
            rs = h_rs[i]
            radius = h_radius[i]
            density = h_profile[i]
            txt.write(str(rs) + '  ' + str(radius) + '  ' + str(density) + '\n')
    return Mhalo, h_rsvir

def halo_data(datapath):
    txt_path = datapath + '/top_hat_halo.txt'
    data = np.loadtxt(txt_path)
    rsdata = data[:, 0]
    rdata = data[:, 1]
    rhodata = data[:, 2]
    return rsdata, rdata, rhodata

def halo_radius(rs, datapath):
# Output unit in pc
    h_rs = halo_data(datapath)[0]
    h_radius = halo_data(datapath)[1]
    from scipy import interpolate
    result = interpolate.interp1d(h_rs, h_radius)
    return result(rs)

def halo_density(rs, datapath):
# Output unit in M_\odot\cdot Mpc^-3
    h_rs = halo_data(datapath)[0]
    h_profile = halo_data(datapath)[2]
    from scipy import interpolate
    result = interpolate.interp1d(h_rs, h_profile)
    return result(rs)

def c_expression(rs, rsvir):
# This is a guessing function of c, it keeps almost 1 before virialization, then gradually closes to 4, when halo
# virialized, it becomes 4
    if rs > rsvir:
        return 1 + 3 * np.exp(-(rs - rsvir))
    else:
        return 4

def rho_s(Mh, rh, concentration):
# Be careful, the unit of rhalo and r_s here should be in Mpc
    r_s = rh / concentration
    term1 = Mh / 16 / np.pi / np.power(r_s, 3)
    term2 = (1 + concentration) / (-concentration + (1 + concentration) * np.log(1 + concentration))
    return term1 * term2

def rho_NFW(r, Mh, rh, concentration):
# Again, the unit of rh should be in Mpc, the output unit is M_\odot\cdot Mpc^{-3}
    term1 = 4 * rho_s(Mh, rh, concentration)
    r_s = rh / concentration
    r_reduce = r / r_s
    term2 = r_reduce * np.power(1 + r_reduce, 2)
    return term1 / term2

def J_decay_NFW(mdm, taudm, Egamma, Mh, radius, concentration, shape='box', fdm=1, fgamma=1):
# unit: mdm-eV, taudm-s, Egamma-eV, radius-pc, Mh-M\odot
# output unit: J-> cm^-2*s^-1
    factor = fdm * fgamma / mdm / taudm
    if shape == 'box':
        from photon_flux import ps_box
        EdNdE = ps_box(Egamma, mdm)
    else:
        from photon_flux import ps_parabola
        EdNdE = ps_parabola(Egamma, mdm)
    Rhalo = radius * 1e-6 
    def density(r):
    # From M_odot\cdot \Mpc^-3 to eV\cdot cm^-3
        return rho_NFW(r, Mh, Rhalo, concentration) 
    Rhalomin = Rhalo * 1e-5
    from scipy.integrate import quad
    density_term, err = quad(density, Rhalomin, Rhalo)
    return factor * density_term * EdNdE * 1.12e66 * np.power(3.086e24, -3) * 3.086e18 / 1e-6

def J_annihilation_NFW(mdm, xsection, Egamma, Mh, radius, concentration, type='Majorana', thermal='Non', fgamma=1):
# unit: mdm-eV, xsection-cm^3*s^-1, Egamma-eV, radius-pc, Mh-M_\odot
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
    from photon_flux import ps_parabola
    EdNdE = ps_parabola(Egamma, mdm)
    Rhalo = radius * 1e-6 # * 3.086e18# From pc to cm
    def density_func(r):
    # From M_odot\cdot \Mpc^-3 to eV\cdot cm^-3
        return np.power(rho_NFW(r, Mh, Rhalo, concentration), 2)
    Rhalomin = Rhalo * 1e-5
    from scipy.integrate import quad
    density_term, err = quad(density_func, Rhalomin, Rhalo)
    return factor * EdNdE * density_term * np.power(1.12e66 * np.power(3.086e24, -3), 2) * 3.086e18 / 1e-6    
