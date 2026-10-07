import numpy as np
import os
import argparse
from pygrackle import chemistry_data, FluidContainer
from pygrackle.utilities.physical_constants import \
    mass_hydrogen_cgs, \
    mass_electron_cgs, \
    sec_per_Myr, \
    cm_per_mpc, \
    gravitational_constant_cgs
from constants import *
from files import initial_file, save_data, moniter, modify, read_datafile, self_shielding_data, last_line, ssh_prepare, ssh_record, ssh_fitting
from grackle_func import grackle_initial, evolve_step
from time_func import time, time_step, data_time, rs_time, new_drs
from UVB_file import setupUVBFile
from data_file_initial import data_initial
from CMBflux import J_CMB
from background_radiation import J_background_ann, J_background_decay
from DMhalo import ini_radius_deduce, DMhalo_profile, rho_cz
from NFW_halo_flux import J_decay_NFW, J_annihilation_NFW, top_hat_halo, halo_radius, halo_density, c_expression
from DM_rate import kHm, kH2, kion, kheat, k28, k29, k30
from one_zone import one_zone_density, Tvir, real_vir
from mass_loss import density_loss, vir_T
from total_flux import flux_total_decay, flux_total_annihilation, load_flux, flux_dict
from self_shielding import ssh_total

# This is the runner of self-shielding case, and the fitting formulas are used
# This is similar with the nss-runner
# Check the data_path and noss_pata carefully


# Get the value of mchi and tau/xsection from the outside
parser = argparse.ArgumentParser()
#parser.add_argument("--mchi", type=float, required=True)
#parser.add_argument("--tau", type=float, required=True)
#parser.add_argument("--xsection", type=float, required=True)
#parser.add_argument("--Mcloud", type=float, required=True)
parser.add_argument("--method", type=str, required=True)
args = parser.parse_args()

#mchi = args.mchi
#tau = args.tau
#xsection = args.xsection
#Mcloud = args.Mcloud

mchi = 33.81
xsection = 1.12e-37
Mcloud = 1.29e6
method = args.method
mchi_str = f"{mchi:.2f}".replace('.', '_')
#tau_str = f"{tau:.2e}".replace('.', '_').replace('+', '')
xsection_str = f"{xsection:.2e}".replace('.', '_').replace('-', '_')
Mcloud_str = f"{Mcloud:.2e}".replace('.', '_').replace('+', '')
print(method, type(method))
# Build the path
#data_path = f"output/Ssh/Decay/Box/Mc{Mcloud_str}/mchi{mchi_str}/tau{tau_str}"
data_path = f"output/Ssh/Annihilation_new/Non_thermal/Mc{Mcloud_str}/mchi{mchi_str}/xsection{xsection_str}"
data_path = data_path + '/' + method
print(data_path)

#noss_path = f"output/Noss/Decay/Box/Mc{Mcloud_str}/mchi{mchi_str}/tau{tau_str}"
noss_path = f"output/Noss/Annihilation_new/Non_thermal/Mc{Mcloud_str}/mchi{mchi_str}/xsection{xsection_str}"

tiny_number = 1e-60
gyr_to_myr = 1e3
step_path = data_path + '/time_step.txt'
initial_rs = 900
final_rs = 10
rs = initial_rs
flag = 0
UVB = 1
DM = True
rho_current  = rho_crit
new = 0
drs_ini = 0.4
vir_rs = 21
real_virv = real_vir(vir_rs) # 'Real' vir redshift
vir = round(real_virv, 1)
halo_path = data_path
data_initial(data_path)
if os.path.exists(noss_path):
    ssh_prepare(noss_path, data_path)
else:
    Mhalo, rsvirhalo = top_hat_halo(Mcloud, halo_path)
if not os.path.exists(halo_path + '/J_flux.h5'):
#    flux_total_decay(Mcloud, halo_path, mchi, tau, initial_rs, final_rs, drs_ini, halo=True, shape='box')
    flux_total_annihilation(Mcloud, halo_path, mchi, xsection, initial_rs, final_rs, drs_ini, halo=True, type='Majorana', thermal='Non', fgamma=1)
flux_data = flux_dict(halo_path)
parameter = ssh_fitting(method)
while rs > vir:
    def J_gamma(Egamma):
        from scipy.interpolate import interp1d
        Egammalst, Jfluxlst = flux_data[str(round(rs, 1))]
        result = interp1d(Egammalst, Jfluxlst)
        return result(Egamma)
    if rs == initial_rs:
        initial_file(data_path, rs, one_zone_density(rs, vir_rs, rho_current), method='STD')
        drs = drs_ini
    drs = drs_ini
    k_Hm = kHm(J_gamma)
    k_H2 = kH2(J_gamma)
    pion = kion(J_gamma, 100)
    pheat = kheat(J_gamma, 100)
    pheat_unit = [x * 1.60218e-12 for x in pheat]
    k28v = k28(J_gamma, 100)
    k29v = k29(J_gamma, 100)
    k30v = k30(J_gamma, 100)
    fion, fH2 = ssh_total(halo_path, mchi, rs, parameter)
    ssh_record(halo_path, rs, fion, fH2, rs_ini = initial_rs)
    k_H2ss = k_H2 * fH2
    pionss = [x * fion for x in pion]
    setupUVBFile(rs, k_H2ss, k_Hm, pionss, k28v, k29v, k30v, pheat_unit, data_path)
    if rs == initial_rs:
        drs = drs_ini
    fc = grackle_initial(data_path, rs, UVB)
    current_time = time(rs) * gyr_to_myr
    time_stepv = time_step(rs, rs - drs) * gyr_to_myr
    new_density = one_zone_density(rs - drs, vir_rs, rho_current) / (1.67e-24)
    new_dict, cooling_time, temperature, mu = evolve_step(fc, rs - drs, current_time, time_stepv, new_density)
#    print(rs, temperature)
    save_data(new_dict, data_path)
    if new == 0:
        with open(step_path, 'w') as step_txt:
            step_txt.write(str(rs) + '  ' + str(drs) + '\n')
        new = 1
    else:
        with open(step_path, 'a') as step_txt:
            step_txt.write(str(rs) + '  ' + str(drs) + '\n')
    if abs(temperature - 1.0) < 1e-6:
        import sys
        sys.exit()
    rs = rs - drs
while rs <= vir and rs >= final_rs:
    def J_gamma(Egamma):
        from scipy.interpolate import interp1d
        Egammalst, Jfluxlst = load_flux(halo_path, round(rs, 1))
        result = interp1d(Egammalst, Jfluxlst)
        return result(Egamma)
    
    drs = 0.4
    k_Hm = kHm(J_gamma)
    k_H2 = kH2(J_gamma)
    pion = kion(J_gamma, 100)
    pheat = kheat(J_gamma, 100)
    pheat_unit = [x * 1.60218e-12 for x in pheat]
    k28v = k28(J_gamma, 100)
    k29v = k29(J_gamma, 100)
    k30v = k30(J_gamma, 100)
    fion, fH2 = ssh_total(halo_path, mchi, rs, parameter)
    ssh_record(halo_path, rs, fion, fH2, rs_ini = initial_rs)
    k_H2ss = k_H2 * fH2
    pionss = [x * fion for x in pion]
    setupUVBFile(rs, k_H2ss, k_Hm, pionss, k28v, k29v, k30v, pheat_unit, data_path)
#    setupUVBFile(rs, k_H2, k_Hm, pion, k28v, k29v, k30v, pheat_unit, data_path)
    gamma = 5 / 3
    last_step = last_line(data_path + '/raw_data.txt')
    old_density = last_step[1]
    old_temperature = last_step[2]
    Tvir = vir_T(Mcloud, vir)
    mean_molecular_weight = last_step[-1]
    if old_temperature < Tvir and flag == 0:
        energy_vir = Tvir / mean_molecular_weight * 1.261899493077487e-14 # Need to be updated carefully
        modify(data_path, Tvir, energy_vir)
        flag = 1
        fc = grackle_initial(data_path, rs, UVB)
    else:
        fc = grackle_initial(data_path, initial_rs, UVB)
    current_time = time(rs) * gyr_to_myr
    time_stepv = time_step(rs, rs - drs) * gyr_to_myr
#    rho_loss = density_loss(data_path, Mcloud, time_stepv) / (1.67e-24)
#    new_density = new_density - rho_loss
#    Mcloud = Mcloud - mass_loss(rho, radius, temperature, mu, lambda_J)
    new_density = one_zone_density(rs - drs, vir_rs, rho_current) / (1.67e-24)
    new_dict, cooling_time, temperature, mu = evolve_step(fc, rs - drs, current_time, time_stepv, new_density)
#    Mcloud = Mcloud - mass_loss(new_density * 1.67e24, Radius(Mcloud, new_density * 1.67e24), temperature, mu, Jeans_parameter(Mcloud, Radius(Mcloud, new_density * 1.67e24), temperature, mu))
    save_data(new_dict, data_path)
    if new == 0:
        with open(step_path, 'w') as step_txt:
            step_txt.write(str(rs) + '  ' + str(drs) + '\n')
        new = 1
    else:
        with open(step_path, 'a') as step_txt:
            step_txt.write(str(rs) + '  ' + str(drs) + '\n')
    if abs(temperature - 1.0) < 1e-6:
        import sys
        sys.exit()
    rs = rs - drs
