import numpy as np
from files import new_initial


def Radius(Mcloud, rho):
# Input unit: Mcloud-M\odot, \rho-g/cm^3; Output unit: radius-pc
    rho = rho / 6.76964e-41
    R = np.power(3 * Mcloud / 4 / np.pi / rho, 1 / 3)
    return R * 1e6 # to pc

def Jeans_parameter(Mcloud, radius, temperature, mu):
# Input quantities Mcloud is the mass of gas cloud in unit M_\odot, radius is in unit pc, temeprature in unit K, mu is the mean molecular weight
    G_cgs = 6.67259e-8 # Gravitational constant in cgs
    mp = 1.67262192e-24
    kB_cgs = 1.3807e-16 # Boltzman constant in cgs
    Mcloud_g = Mcloud * 1.989e33
    Rg = radius * 3.086e18
    vesc_s = 2 * G_cgs * Mcloud_g / Rg # squared escape velocity in cm^2/s^2
    cs_s = kB_cgs * temperature / mu / mp #squared sound velocity in cm^2/s^2
    lambda_J = vesc_s / cs_s / 2
    return lambda_J

def Jeans_escape(rho, radius, temperature, mu, lambda_J):
# input unit: rho-g/cm^3, radius-pc, tempeature-K, mu and lambda_J are dimensionless
# Current output unit in g/s
    G_cgs = 6.67259e-8 # Gravitational constant in cgs
    kB_cgs = 1.3807e-16 # Boltzman constant in cgs
    mp = 1.67262192e-24
    cs = np.sqrt(kB_cgs * temperature / mu / mp)
    radius = radius * 3.086e18 # Convert to cm
    rate = rho * cs * (1 + lambda_J) * np.exp(-lambda_J) / 2 / np.sqrt(np.pi)
    rate = rate * 4 * np.pi * np.power(radius, 2)
    return rate

def hydrodynamic_escape(rho, radius, temperature, mu):
# input unit: rho-g/cm^3, radius-pc, tempeature-K
# Current output unit in g/s
    G_cgs = 6.67259e-8 # Gravitational constant in cgs
    kB_cgs = 1.3807e-16 # Boltzman constant in cgs
    mp = 1.67262192e-24
    cs = np.sqrt(kB_cgs * temperature / mu / mp)
    radius = radius * 3.086e18 # Convert to cm
    rate = rho * np.pi * np.power(radius, 2) * cs
    return rate

def mass_loss(rho, radius, temperature, mu, lambda_J):
# input unit: rho-g/cm^3, radius-pc, tempeature-K, mu and lambda_J are dimensionless
# Output unit in M_\odot/Myr
    if lambda_J > 3 and lambda_J <= 10:
        rate = Jeans_escape(rho, radius, temperature, mu, lambda_J)
    elif lambda_J <= 3:
        rate = hydrodynamic_escape(rho, radius, temperature, mu)
    else:
        rate = 0
    return rate * 5.02785e-34 / 3.17e-14

def density_variance(radius, time, Loss):
# Input unit: rho-g/cm^3, radius-pc, time-Myr, Loss-M_\odot/Myr
# Output unit-g/cm^3
    radius = radius / 1e6 # Convert to Mpc just to use the same convert parameter because I'm lazy
    Delta = 3 * Loss / np.pi / 4 / np.power(radius, 3) * time
    return Delta * 6.76964e-41

def density_loss(data_path, Mcloud, time):
    rs, density, temperature, HI, HII, HeI, HeII, HeIII, de, H2I, H2II, HM, energy, pressure, mu = new_initial(data_path)
    radius = Radius(Mcloud, density)
    lambda_J = Jeans_parameter(Mcloud, radius, temperature, mu)
    Loss = mass_loss(density, radius, temperature, mu, lambda_J)
    rho_loss = density_variance(radius, time, Loss)
    return rho_loss

def vir_T(Mcloud, rsvir):
# Output in K
    Omega_m = 0.3153
    h = 0.6736
    term1 = Omega_m * np.power(h, 2) / 0.14
    term2 = Mcloud / 1e4
    term3 = rsvir / 100
    return 224 * np.power(term1, 1 / 3) * np.power(term2, 2 / 3) * term3
