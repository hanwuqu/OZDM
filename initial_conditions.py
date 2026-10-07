import numpy as np
import matplotlib.pyplot as plt
import pickle

# This file store the initial condition functions, there are two methods:
# 'CLASS' and 'STD', where 'CLASS' uses data from CLASS, STD uses date
# from the file I stole from darkhistory, most of them are same, but
# the ionization fraction of helium is different.

def nb(density):
    mp = 1.67262192e-24 # mass of proton in g
    return density / mp

def nH(density):
    YHe = 0.24
    return nb(density) * (1 - YHe)

def ne(density, xe):
    return nH(density) * xe

def nHe(density):
    YHe = 0.24
    return nb(density) * YHe / 4

def mu(nb, nHI, nHII, nHeI, nHeII, nHeIII, nHm, nH2I, nH2II):
    mu = nHI + nHII + nHeI + nHeII + nHeIII + nHm + nH2I + nH2II
    mu = nb / mu
    return mu

# ------------------------Data from std file-------------------------#
def std_data(rs):
    from scipy.interpolate import interp1d
    with open("input/std_soln_He.p", "rb") as f:
        data = pickle.load(f)
    glob_hist_data = {
                'rs'    : data[0,:],
                'xHII'  : data[2,:],
                'xHeII' : data[3,:],
                'Tm'    : data[1,:]
            }
    def xHII_std(rs):
# Baseline nHII/nH value.
        rs_vec   = glob_hist_data['rs']
        xHII_vec = glob_hist_data['xHII']

        _xHII_std = interp1d(rs_vec, xHII_vec)

        return _xHII_std(rs)

    def xHeII_std(rs):
# Baseline nHeII/nH value
        rs_vec    = glob_hist_data['rs']
        xHeII_vec = glob_hist_data['xHeII']

        _xHeII_std = interp1d(rs_vec, xHeII_vec)

        return _xHeII_std(rs)

    def Tm_std(rs):
        rs_vec  = glob_hist_data['rs']
        Tm_vec  = glob_hist_data['Tm']
        _Tm_std = interp1d(rs_vec, Tm_vec)

        return _Tm_std(rs)
    xe = xHII_std(rs) + 0.0789 * (xHeII_std(rs) + 1e-12)
    return rs, Tm_std(rs) / 8.617333262e-5, xe, xHII_std(rs), xHeII_std(rs)

def CLASS_data(rs):
    from scipy.interpolate import interp1d
    data = np.loadtxt('input/CLASS_initial.txt')
    rs_data = data[:, 0]
    Tm_data = data[:, 1]
    xe_data = data[:, 2]
    xH_data = data[:, 3]
    xHe_data = data[:, 4]
    def Tm_CLASS(rs):
        Tm = interp1d(rs_data, Tm_data)
        return Tm(rs)
    def xe_CLASS(rs):
        xe = interp1d(rs_data, xe_data)
        return xe(rs)
    def xH_CLASS(rs):
        xH = interp1d(rs_data, xH_data)
        return xH(rs)
    def xHe_CLASS(rs):
        xHe = interp1d(rs_data, xHe_data)
        return xHe(rs)
    return rs, Tm_CLASS(rs), xe_CLASS(rs), xH_CLASS(rs), xHe_CLASS(rs)
#-------------------------Get the Density-----------------------#
def initial_data(rs, density, method='CLASS'):
    if method == 'CLASS':
        rs_data, temperature, xe, xH, xHe = CLASS_data(rs)
    else:
        rs_data, temperature, xe, xH, xHe = std_data(rs)
    
    mp = 1.67262192e-24
    nbaryon = nb(density)
    n_H = nH(density)
    n_He = nHe(density)
    nHm = nb(density) * 1e-20
    nH2I = nb(density) * 1e-20
    nH2II = nb(density) * 1e-20
    nHII = n_H * xH
    nHI = n_H - nHII - nHm - 2 * nH2I - 2 * nH2II

    nHeII = n_He * xHe
    nHeIII = n_He * 1e-12
    nHeI = n_He - nHeII - nHeIII
    mean_molecular_weight = mu(nbaryon, nHI, nHII, nHeI, nHeII, nHeIII, nHm, nH2I, nH2II)
    energy = temperature / mean_molecular_weight * 1.261899493077487e-14
    rhoHI = nHI * mp
    rhoHII = nHII * mp
    rhoHm = nHm * mp
    rhoH2I = nH2I * mp * 2
    rhoH2II = nH2II * mp * 2
    rhoHeI = nHeI * mp * 4
    rhoHeII = nHeII * mp * 4
    rhoHeIII = nHeIII * mp * 4
    rhoe = ne(density, xe) * 9.10938e-28
    return rs, density, temperature, rhoHI, rhoHII, rhoHeI, rhoHeII, rhoHeIII, rhoe, rhoH2I, rhoH2II, rhoHm, energy, 0, mean_molecular_weight, 5. / 3. 
