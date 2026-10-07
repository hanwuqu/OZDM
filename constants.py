'''-----------------------Cosmological Constants------------------------'''
import numpy as np
h = 0.6736 # Hubble Constant
H0 = 100*h*3.241e-20 # unit in s^-1
mp = 0.938272081e9 # proton mass in eV
omega_m      = 0.3153
omega_rad    = 8e-5
omega_lambda = 0.6847
omega_baryon = 0.02237/(h**2)
omega_DM = 0.1200/(h**2)
rho_crit = 1.05371e4*(h**2) # unit in eV*cm^-3
rho_crit = rho_crit*1.78266192e-33 # unit in g*cm^-3
rho_baryon = rho_crit / 1.78266192e-33 * omega_baryon # Baryon density in eV*cm^-3
nB = rho_baryon / mp # Baryon number density in cm^-3
YHe = 0.245 # Helium abundance by mass
nH0 = (1-YHe)*nB # Atomic hydrogen number density in cm^-3
T0 = 2.725 # Current temperature of CMB, in K

#Physical Constants and Unit Conversions
sigmaT_eV = 1.707e-15 #Thompson cross-section in eV^-2
G_eV = 6.708e-57 #Gravitational constnat in eV^-2
Mstar0 = 1.12e66 #eV/solarMass
eVperg = 5.6096e32 #eV/g
c = 3e8 #m/s
hbar = 6.582e-16 #eV * s
kmperm = 1e-3 #km/m
mpercm = 1e-2 #m/cm
kmpercm = kmperm*mpercm #km/cm
kmperMpc = 3.086e19 #km per Megaparsec
eVMpc =  kmperm * c * hbar/kmperMpc  #eV*Mpc (converts natural units to Mpc)
GeVpereV = 1e-9 #GeV/eV
GeVperMeV = 1e-3 #GeV/MeV
MpcperGpc = 1e3
speryear = 3.154e7 #seconds/year
kB = 8.617330e-5 #eV/K
eVperErg = 6.242e+11 #erg/eV
ergsPereV = 1/eVperErg
eV3perJ21 = 1e-21 * eVperErg * (hbar*c / mpercm)**2 #(eV^3/sr / J21) where J21 is  1e-21 erg / s / cm^2 / Hz / sr

eV2perJ21wrong = 1e-21 * ergsPereV / hbar * (hbar * c / mpercm)**4#1e10−21 ergs−1 Hz−1 cm−2 cm−2
#PDG
tUniverse_years = 13.797e9
tUniverse_s = tUniverse_years * speryear

MsolartoeV = 1.12e66
pctocm = 3.086e18
h = 4.136e-15 # Planck constant in unit eV*Hz^-1
hbar = h / 2 / np.pi
eVtoerg = 1.602e-12
Jdecayunit = MsolartoeV / pctocm**2
Jannihilationunit = MsolartoeV**2 / pctocm**5
cm_2s_1toJ21 = hbar * eVtoerg * 1e21

kB_cgs = 1.3806488e-16 # erg K^-1
mass_hydrogen_cgs = 1.007947 * 1.660538921e-24 # g
G_cgs = 6.67384e-8 # 
