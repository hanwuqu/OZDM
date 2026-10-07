from tqdm import tqdm
from scipy.interpolate import interp1d
import h5py
from NFW_halo_flux import J_decay_NFW, J_annihilation_NFW, top_hat_halo, halo_radius, halo_density, c_expression
import numpy as np
from CMBflux import J_CMB
from background_radiation import J_background_ann, J_background_decay



# This file gives the total flux from CMB, background and DM halo

def flux_total_decay(Mcloud, halo_path, mchi, tau, initial_rs, final_rs, drs, halo=True, shape='box'):
    Mhalo, rsvirhalo = top_hat_halo(Mcloud, halo_path)
    rs_list = []
    rs = initial_rs
    while rs > final_rs:
        rs_list.append(rs)
        rs = rs - drs
    Egamma_list = list(i for i in np.logspace(np.log10(0.76), np.log10(100), 100))
    data_dict = {}
    for i in tqdm(rs_list):
        def Jflux_decay_halo(Egamma):
            return J_decay_NFW(mchi, tau, Egamma, Mhalo, halo_radius(i, halo_path), c_expression(i, rsvirhalo), shape)
        def J_gamma(Egamma):
            if halo:
                return J_CMB(i, Egamma) + J_background_decay(i, mchi, tau, Egamma, shape) + Jflux_decay_halo(Egamma)
            else:
                return J_CMB(i, Egamma) + J_background_decay(i, mchi, tau, Egamma, shape)
        J_list = list(J_gamma(i) for i in Egamma_list)
        rate_list = [Egamma_list, J_list]
        data_dict[round(i, 1)] = rate_list
    file_path = halo_path + '/J_flux.h5'
    with h5py.File(file_path, 'w') as f:
        for rsv, lists in data_dict.items():
            grp = f.create_group(str(rsv))  # use rs as group name
            grp.create_dataset("Egamma", data=np.array(lists[0]))
            grp.create_dataset("Jflux", data=np.array(lists[1]))

def flux_total_annihilation(Mcloud, halo_path, mchi, xsection, initial_rs, final_rs, drs, halo=True, type='Majorana', thermal='Non', fgamma=1):
    Mhalo, rsvirhalo = top_hat_halo(Mcloud, halo_path)
    rs_list = []
    rs = initial_rs
    while rs > final_rs:
        rs_list.append(rs)
        rs = rs - drs
    Egamma_list = list(i for i in np.logspace(np.log10(0.76), np.log10(100), 100))
    data_dict = {}
    for i in tqdm(rs_list):
        def Jflux_annihilation_halo(Egamma):
            return J_annihilation_NFW(mchi, xsection, Egamma, Mhalo, halo_radius(i, halo_path), c_expression(i, rsvirhalo), type, thermal, fgamma)
        def J_gamma(Egamma):
            if halo:
                return J_CMB(i, Egamma) + J_background_ann(mchi, xsection, Egamma, i, type='Majorana', thermal='Non', fgamma=1) + Jflux_annihilation_halo(Egamma)
            else:
                return J_CMB(i, Egamma) + J_background_ann(mchi, xsection, Egamma, i, type='Majorana', thermal='Non', fgamma=1)
        J_list = list(J_gamma(i) for i in Egamma_list)
        rate_list = [Egamma_list, J_list]
        data_dict[round(i, 1)] = rate_list
    file_path = halo_path + '/J_flux.h5'
    with h5py.File(file_path, 'w') as f:
        for rsv, lists in data_dict.items():
            grp = f.create_group(str(rsv))  # use rs as group name
            grp.create_dataset("Egamma", data=np.array(lists[0]))
            grp.create_dataset("Jflux", data=np.array(lists[1]))

def load_flux(halo_path, rs):
    file_path = halo_path + '/J_flux.h5'
    with h5py.File(file_path, "r") as f:
        grp = f[str(rs)]   # access group "900"
        Egamma = grp["Egamma"][:]   # load dataset into numpy array
        Jflux  = grp["Jflux"][:]
    return Egamma, Jflux

def flux_dict(halo_path):
    path = halo_path + '/J_flux.h5'
    flux_data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():  # iterate over all group names
            grp = f[key]
            Egamma = grp["Egamma"][:]
            Jflux  = grp["Jflux"][:]
            flux_data[key] = (Egamma, Jflux)
    return flux_data
