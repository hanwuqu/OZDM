# This file include the solution of ODE to obtain the density of DM halo,
# however, due to the abnormal high initail overdensity, I'm not 
# convinced completely. May update someday.

import numpy as np

G_origin = 6.673015e-11 # Gravitational Constant, m^3 kg^-1 s^-2
m_to_pc = 3.24078e-17 # meter in pc
kg_to_solar = 5.02785e-31 # kg to solar mass
G = G_origin * m_to_pc**3 / kg_to_solar # G in pc^3 M_\odot^-1 s^-2
h = 0.6736 # Hubble Constant
H0 = 100*h*3.241e-20 # unit in s^-1
omega_m      = 0.3153
omega_rad    = 8e-5
omega_lambda = 0.6847
omega_baryon = 0.02237/(h**2)
omega_DM = 0.1200/(h**2)
#f_b = omega_baryon / omega_m
gyr_to_s = 3.15e16 # Gyr to second
rho_crit = 1.05371e4*(h**2) # unit in eV*cm^-3
f_b = 0

'''-------------------Time Functions----------------------'''
def time(rs):
    from astropy.cosmology import Planck15
    # Unit of output is s
    z = rs - 1
#    return (Planck15.age(0).value - Planck15.age(z).value) * gyr_to_s
    return Planck15.age(z).value * gyr_to_s

def time_step(rs0, rs1):
    # Unit of output is s
    return time(rs0) - time(rs1)

def time_to_redshift(t_seconds):
    from astropy.cosmology import Planck15
    import astropy.units as u
    from scipy.optimize import root_scalar
    t_age = t_seconds * u.s
    t_gyr = t_age.to(u.Gyr)

    def func(z):
        return (Planck15.age(z) - t_gyr).value

    result = root_scalar(func, bracket=[1e-10, 1e4], method='brentq')
    return result.root + 1

'''--------------------Initial Overdensity----------------------'''
def overdensity_ini(rs_ini, rs_vir):
    t_ini = time(rs_ini)
    t_vir = time(rs_vir)
    t_ta = t_vir / 2
    overdensity_ta = 1.06
#    t_ta = t_vir
#    overdensity_v = 1.69
    return overdensity_ta * np.power(t_ini / t_ta, 2 / 3)

'''-----------------A lot of Deduction Functions----------------'''
def rho_cz(rs):
    Om = 0.3153
    OL = 0.6847
    h = 0.6736
    rho_c0 = 2.775e11*h**2 #solar masses per Mpc^3
#    rho_c0 = rho_crit / (1.989e33) / np.power(3.24e-25, 3) # in M_\odot\cdot Mpc^{-3}
    rho_cz = rho_c0 * np.power(rs, 3)
    return rho_c0*(omega_rad * rs**4 + omega_m * rs**3 + omega_lambda)

def Mhalo_deduce(rs_ini, r_ini, delta=0.01):
#    overdensity = overdensity_ini(rs_ini, rs_vir) * (1 + delta) + 1
    overdensity = delta + 1
    rho_h = rho_cz(rs_ini) * overdensity
    M = 4 * np.pi * np.power(r_ini / 1e6, 3) * rho_h / 3
    return M

def ini_radius_deduce(rs_ini, Mhalo, delta=0.01):
    overdensity = delta + 1
    rho_h = rho_cz(rs_ini) * overdensity
    radius_cubic = Mhalo * 3 / 4 / np.pi / rho_h
    radius = np.power(radius_cubic, 1 / 3)
    return radius * 1e6

def rho_cz_unit(rs):
# Still the critical density of the universe, unit g*cm^-3
    Om = 0.3153
    OL = 0.6847
    h = 0.6736
    rho_c0 = 2.775e11*h**2 #solar masses per Mpc^3
#    rho_c0 = rho_crit / (1.989e33) / np.power(3.24e-25, 3) # in M_\odot\cdot Mpc^{-3}
    rho_cz = rho_c0 * np.power(rs, 3)
    return rho_c0*(omega_rad * rs**4 + omega_m * rs**3 + omega_lambda) * 6.76964e-41

def rhovir_overdensity(rs):
    one_plus_delta_vir = 18 * np.pi**2
    return rho_cz_unit(rs) * one_plus_delta_vir

def Hubble_parameter(rs):
    Hz = H0 * np.sqrt(omega_rad * rs**4 + omega_m * rs**3 + omega_lambda)
    return Hz

def Hubble_flow(rs_ini, r_ini):
# Output in unit pc*s^{-1}    
    H_ini = Hubble_parameter(rs_ini)
    return H_ini * r_ini

'''------------------------ODE Solver--------------------------'''
def solve_dm_only(rs_ini, rs_fin, r_ini, M_h, delta=0.01):
    from scipy.integrate import solve_ivp
    t_ini = time(rs_ini)
    t_fin = time(rs_fin)
#    delta_ini = overdensity_ini(rs_ini, rs_vir) * (1 + delta)
    v_ini = Hubble_flow(rs_ini, r_ini) * (1 - delta / 3)
    M_h = Mhalo_deduce(rs_ini, r_ini, delta)
    Lambda = 1.4657e-52 # Cosmological constant in m^{-2}
    Lambda_s = Lambda * (3e8)**2 # To s^{-2}
    Lambda_pc = Lambda * (3.086e16) # in unit pc^{-2}
    def phi(x):
        return -G * M_h / x -  Lambda_pc / 6 * np.power(x, 2)
    def rhs(t, y):
        r, v = y
        acc = -G * M_h / r**2 + Lambda_s / 3 * r 
        return [v, acc]
#    print(v_ini**2 / 2 + phi(r_ini), v_ini, Hubble_flow(rs_ini, r_ini), phi(r_ini))
    y0 = [r_ini, v_ini]
    t_eval = np.linspace(t_ini, t_fin, 10000)
    sol = solve_ivp(rhs, (t_ini, t_fin), y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    r_array = sol.y[0]
    v_array = sol.y[1]
    E_array = np.power(v_array, 2) / 2 + phi(r_array)
    rs_array = np.array([time_to_redshift(t) for t in sol.t])
    return rs_array, r_array, E_array

def find_virialization(rs_ini, rs_fin, r_ini, M_h, delta=0.01):
# Output density unit: M_\odot pc^-3
    rs_list, r_list, E_list = solve_dm_only(rs_ini, rs_fin, r_ini, M_h, delta)
    def is_decreasing(lst):
        return all(lst[i] > lst[i+1] for i in range(len(lst) - 1))
    if is_decreasing(r_list):
        print('NO SOLUTION')
        return null

    else:
        r_list[r_list < 0] = 0
        M_h = Mhalo_deduce(rs_ini, r_ini, delta)
        rho_list = list(M_h / (4 * np.pi * (i / 1e6)**3 / 3) for i in r_list)
        rs_real = []
        rho_real = []
        r_real = []
        for i in range(len(rho_list)):
            if rho_list[i] != np.inf:
                rs_real.append(rs_list[i])
                rho_real.append(rho_list[i])
                r_real.append(r_list[i])
        from scipy.optimize import root_scalar
        from scipy.interpolate import interp1d
        halo_density = interp1d(rs_real, rho_real)
        def target_function(rs):
            return halo_density(rs) / rho_cz(rs) - 200
        sol = root_scalar(target_function, bracket=[rs_real[0], rs_real[-1]], method='brentq')
        if sol.converged:
            overdensity = list(halo_density(i) / rho_cz(i) for i in rs_real)
            return rs_real, r_real, rho_real, sol.root, halo_density(sol.root), overdensity 
        else:
            print("Fail to Find Virialization")
            return null

'''-------------------NFW and Einasto Profile-------------------'''
# NFW Profile----------------
def r200(M, rs):
    delta = 200
    Om = 0.3153
    OL = 0.6847
    h = 0.6736
    rho_c0 = 2.775e11*h**2 #solar masses per Mpc^3
#    rho_c0 = rho_crit / (1.989e33) / np.power(3.24e-25, 3) # in M_\odot\cdot Mpc^{-3}
#    rho_cz = rho_c0 * np.power(rs, 3)
    rho_cz = rho_c0*(omega_rad * rs**4 + omega_m * rs**3 + omega_lambda)
    return (3*M / (4*np.pi*delta*rho_cz))**(1./3.)

def r_s(M, rs, conc):
    return r200(M, rs)/conc

def rho_s(M, rs, conc):
    numerator = np.power(conc, 3) * (1 + conc) * M
#    log_numerator = 3 * np.log(conc) + np.log(1 + conc) + np.log(M)
#    numerator = np.exp(log_numerator) 
    denominator = 16 * np.pi * np.power(r200(M, rs), 3) * (-conc + (1 + conc) * np.log(1 + conc))
    return numerator / denominator
#    log_numerator = 3 * np.log(conc) + np.log(1 + conc) + np.log(M)
#    log_denominator = np.log(16 * np.pi) + 3 * np.log(r200(M, rs)) + np.log(-conc + (1 + conc) * np.log(1 + conc))
    
#    return np.exp(log_numerator - log_denominator)  # Avoids overflow

def rhoNFW(M, rs, r, conc):
    numerator = 4 * rho_s(M, rs, conc)
    radius_term  = r / r_s(M, rs, conc)
    denominator = radius_term * np.power(1 + radius_term, 2)
    return numerator / denominator

# Einasto Profile-----------------
# Einasto \alpha parameter
def peakHeightDutton(M,z):
    #peak heigh nu from Dutton and Maccio 2014. arxiv:1402.7073
    #M is halo mass in solar masses, z is redshift
    h = 0.6736
    m = np.log10(M*h / 1e12)
    v0 = np.power(10, -0.11 + 0.146*m + 0.0138*np.square(m) + 0.00123*np.power(m,3)) #eq A1
    v = v0 * (0.033 + 0.79*(1+z) + 0.176*np.exp(-1.356*z)) #eq A2
    return v

def einastoAlphaDutton(M,rs):
    #Einasto slope parameter from Dutton and Maccio 2014. arxiv:1402.7073
    #M is halo mass in solar masses, z is redshift
    z = rs - 1
    v = peakHeightDutton(M,z)
    alpha = 0.0095 * np.square(v) + 0.155 #eq 17
    return alpha

# Einasto Profile, from Avi's code
def densityEinasto(r, r0, rho0, alpha):
    return rho0 * np.exp(-2/alpha * (np.power(r/r0,alpha) - 1))

def calcEinastoParameters(M200, rs, conc):
    conc_value = conc
    if not np.isscalar(conc_value):
        conc_value = conc_value[0]
    alpha = einastoAlphaDutton(M200,rs)
    rsNFW = r_s(M200, rs, conc_value)
#    print(rsNFW)
    rho0NFW = rho_s(M200, rs, conc_value)
    Nrfit = 1000
    rvals = np.linspace(rsNFW, 3*rsNFW, Nrfit)
#    rvals = np.linspace(1e-4, 200, Nrfit)
    profileNFW = rhoNFW(M200, rs, rvals, conc_value)
#    print(profileNFW)
    def fitProfile(r, r0, rho0):
        return densityEinasto(r, r0, rho0, alpha)
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(fitProfile, rvals, profileNFW, p0=(rsNFW, rho0NFW))
    r0Einasto = popt[0]
    rho0Einasto = popt[1]

    return r0Einasto, rho0Einasto

def rhoEinasto(M200, rs, r, conc):
    r0, rho0 = calcEinastoParameters(M200, rs, conc)
    alpha = einastoAlphaDutton(M200,rs)
    return densityEinasto(r, r0, rho0, alpha)

'''-------------------Mean Density of NFW/ Einasto DM halo------------------'''
def r_cz(M, rs):
# where the density of the halo equals to the critical density of the Universe
    Om = 0.3153
    OL = 0.6847
    h = 0.6736
    rho_c0 = 2.775e11*h**2 #solar masses per Mpc^3
    rho_cz = rho_c0*(OL + Om*rs**3)
#    rho_c0 = rho_crit / (1.989e33) / np.power(3.24e-25, 3) # in M_\odot\cdot Mpc^{-3}
#    rho_cz = rho_c0 * np.power(rs, 3)
    return (3*M / (4*np.pi*rho_cz))**(1./3.)

def mean_density(profile, conc, M, rs, rmin):
    conc_value = conc
    def integrand(r):
        return profile(M, rs, r, conc_value) * np.power(r, 2)
    rmax = r200(M, rs)
    rmin = rmin # divergency at r=0 for NFW profile
#    rmin = 1e-8 * r200(M, rs) / 4
    from scipy.integrate import quad
    integral, _ = quad(lambda r: 4 * np.pi * integrand(r), rmin, rmax)
    return integral * 3 / 4 / np.pi / np.power(rmax, 3)

'''-----------------Main Function of DM Halo----------------'''
# Output unit in M_\odot Mpc^-3, scared me...
def DMhalo_profile(rs_ini, rs_fin, r_ini, M_h, delta=0.01):
    rs_list, r_list, rho_list, rsvir, rhovir, overdensity_list = find_virialization(rs_ini, rs_fin, r_ini, M_h, delta)
    from scipy.interpolate import interp1d
    halo_density = interp1d(rs_list, rho_list)
    halo_radius = interp1d(rs_list, r_list)
    rs_output = list(i for i in np.logspace(np.log10(10), np.log10(1101), num=1000))
    profile = []
    radius = []
    for i in rs_output:
        if i <= rsvir:
#            rho = mean_density(rhoNFW, 4, M_h, i, 1e-3 * r_s(M_h, i, 4))
            rho = mean_density(rhoEinasto, 4, M_h, i, 0)
            profile.append(rho)
            radius.append(r200(M_h, i) * 1e6)
        else:
            profile.append(halo_density(i))
            radius.append(halo_radius(i))
    return rs_output, radius, profile, rsvir, rhovir, overdensity_list
