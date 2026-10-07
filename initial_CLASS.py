import numpy as np
G_CS = 6.6743e-11 #m^3\cdot kg^{-1}\cdot s^{-1}
c = 2.99792458e8 # m\cdot s^{-1}
Mpc_to_m = 3.0856776e22
mp = 1.67262192e-27 # mass of proton in kg
mH = 1.6735575e-27 # mass of hydrogen atom in kg
me = 9.109e-31 # electron mass in kg
mHe = 6.6464731e-27 # heilum atom mass in kg

# CLASS Functions
def get_density(cosmo, z):
    background = cosmo.get_background()
    idx = (np.abs(background['z'] - z)).argmin()
    rho_b = background['(.)rho_b'][idx]  # Baryon density
    return rho_b

def get_ionization_fractions(cosmo, z):
    thermodynamics = cosmo.get_thermodynamics()
    z_arr = thermodynamics['z']
    idx = (np.abs(z_arr - z)).argmin()
    xe = thermodynamics['x_e'][idx]
    xH = thermodynamics['x_H'][idx]    # <-- New: hydrogen ionization fraction
    xHe = thermodynamics['x_He'][idx]  # <-- New: helium ionization fraction
#    print(thermodynamics.keys())
    return xe, xH, xHe

def get_temperature(cosmo, z):
    thermodynamics = cosmo.get_thermodynamics()
    z_arr = thermodynamics['z']
    T_m_arr = thermodynamics['Tb [K]']
    # Find the closest redshift in the array
    idx = (np.abs(z_arr - z)).argmin()
    T_m = T_m_arr[idx]
    return T_m

def get_critical_density(cosmo, z):
    background = cosmo.get_background()
    idx = (np.abs(background['z'] - z)).argmin()
    rho_c = background['(.)rho_crit'][idx]  # Critical density
    return rho_c

def CLASS_initial(rs):
# The unit of density in output is kg/m^3, temperature in K
    z = rs - 1
    from classy import Class
    params = {
    'output': 'tCl,lCl,mPk',
    'h':0.6736,
    'Omega_b': 0.049301692328524445,
    'Omega_cdm': 0.26447041034523616,
    'A_s': 2.1e-9,
    'n_s': 0.96,
#    'tau_reio': 0.06,
    'YHe': 0.241,
    'z_reio': 8.5,  # Reionization redshift
    'reio_parametrization': 'reio_camb',
    'reionization_width': 0.5,
    'z_pk': '200,1000',  # Redshifts of interest
    'Omega_k': 0.0  # Flat universe
    }

# Initialize CLASS
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    background = cosmo.get_background()
    a = 1 / (1 + z)
    H = cosmo.Hubble(a)
    xe, xH, xHe = get_ionization_fractions(cosmo, z)
    rho_crit = get_critical_density(cosmo, z)
    YHe = params['YHe'] 
# Get approximate temperature at the specified redshift
    temperature = get_temperature(cosmo, z)
    rhob = get_density(cosmo, z)
#    print(background.keys()) 
#    print(cosmo.get_background()['(.)rho_tot'])
    return xe, xH, xHe, temperature, rhob / Mpc_to_m**2 / (G_CS / c**2), rho_crit / Mpc_to_m**2 / (G_CS / c**2), YHe
    

# Initial Functions
def n_e(rho_b, xe):
    return rho_b / mp * xe

def n_H(rho_b, YHe):
    return rho_b / mp * (1 - YHe)

def n_He(rho_b, YHe):
    nH = n_H(rho_b, YHe)
    ratio = YHe / (4 * (1 - YHe))
    return nH * YHe / 4

def hydrogen_species(rho_b, YHe, xH):
    fHm = 1e-20
    fH2I = 1e-20
    fH2II = 1e-20
    nH = n_H(rho_b, YHe)
    nHm = nH * fHm
    nH2I = nH * fH2I / 2
    nH2II = nH * fH2II / 2
    nHII = nH * xH
    nHI = nH * (1 - xH - fHm - fH2I - fH2II)
    return nHI, nHII, nHm, nH2I, nH2II

def helium_species(rho_b, YHe, xHe):
    nHe = n_He(rho_b, YHe)
    nH = n_H(rho_b, YHe)
    fHeIII = 1e-40
    nHeIII = nHe * fHeIII
    nHeII = xHe * nH - 2 * nHeIII
    nHeI = nHe - nHeII - nHeIII
    return nHeI, nHeII, nHeIII

def check_charge_conservation(rho_b, YHe, xe, xH, xHe):
    ne = n_e(rho_b, xe)
    nHI, nHII, nHm, nH2I, nH2II = hydrogen_species(rho_b, YHe, xH)
    nHeI, nHeII, nHeIII = helium_species(rho_b, YHe, xHe)
    ne_cal = nHII + nHeII + 2 * nHeIII + nH2II
    if ne - ne_cal > 0:
        nHeIII = nHeIII + (ne - ne_cal) / 2
        return ne, nHI, nHII, nHm, nH2I, nH2II, nHeI, nHeII, nHeIII 
    elif ne - ne_cal < 0:
        nHm = nHm + (ne_cal - ne)
        return ne, nHI, nHII, nHm, nH2I, nH2II, nHeI, nHeII, nHeIII
    else:
        return ne, nHI, nHII, nHm, nH2I, nH2II, nHeI, nHeII, nHeIII

def initial_conditions_CLASS(rs):
# Output unit of density is g/cm^3, temperature in K, energy...I don't know
    xe, xH, xHe, temperature, rho_b, rho_crit, YHe = CLASS_initial(rs)
    ne, nHI, nHII, nHm, nH2I, nH2II, nHeI, nHeII, nHeIII = check_charge_conservation(rho_b, YHe, xe, xH, xHe)
    rho_e = ne * me * 1e-3
    rho_HI = nHI * mH * 1e-3
    rho_HII = nHII * mH * 1e-3
    rho_Hm = nHm * mH * 1e-3
    rho_H2I = nH2I * 2 * mH * 1e-3
    rho_H2II = nH2II * 2 * mH * 1e-3
    rho_HeI = nHeI * mHe * 1e-3
    rho_HeII = nHeII * mHe * 1e-3
    rho_HeIII = nHeIII * mHe * 1e-3
    mu = ne + nHI + nHII + nHm + (nHeI + nHeII + nHeIII) / 4 + (nH2I + nH2II) / 2 
    mu = rho_b / mp / mu
#    energy_some_value = 2.8207686789203255e-12
#    ratio = temperature / 265.61701587670353
#    energy = energy_some_value * ratio 
    energy = temperature / mu * 1.261899493077487e-14
#    temperature = 2.725 * rs
#    rho_e = 0.004 * me * 1e-3
    return rs, rho_b * 1e-3, temperature, rho_HI, rho_HII, rho_HeI, rho_HeII, rho_HeIII, rho_e, rho_H2I, rho_H2II, rho_Hm, energy, 0

#print(initial_conditions_CLASS(800))
