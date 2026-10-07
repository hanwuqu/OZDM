# Actually, I can include functions in the self_shielding.py file
# but I realized it contains too much, which made it difficult to read
# so here are the functions that will be called in main function
import numpy as np
from files import self_shielding_data, last_line
from DM_rate import kHm, kH2, kion, kheat
from self_shielding import Jeff_annihilation_background
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def get_data_non(rs, method='CLASS'):
# This function gets the background data from CLASS or the one I stole
    from initial_conditions import initial_data
    from DMhalo import rhocz
    def baryon_density(rs):
# Output unit in g\cdot cm^{3}
        solarmasstog = 1.989e33
        Mpctocm = 3.086e24
        unit_convert = solarmasstog / np.power(Mpctocm, 3)
        return rho_cz(rs) * unit_convert
    density = baryon_density(rs)
    return initial_data(rs, density, method)

def get_data_grackle(rs, datapath):
# This function returns the background data from grackle simulation
    return rs, self_shielding_data(rs, datapath)

def _worker_chunk(chunk, mdm, xsection, nbg, rsob):
#    from your_module import Jeff_annihilation_background  # or wherever it’s defined
    return [Jeff_annihilation_background(mdm, xsection, E, nbg, rsob) for E in chunk]

def initial_JbgAnn(mdm, xsection, filepath):
    Egamma_lst = list(i for i in np.logspace(np.log10(0.76), np.log10(mdm), 100))
    last_data = last_line(filepath + '/raw_data.txt')
    rsob = last_data[0]
    rHI = last_data[2]
    rHeI = last_data[4]
    rHeII = last_data[5]
    rHm = last_data[10]
    nHI = rHI / 1.67262192e-24
    nHeI = rHeI / 1.67262192e-24 / 4
    nHeII = rHeII / 1.67262192e-24 / 4
    nHm = rHm / 1.67262192e-24
    nbg = nHI, nHeI, nHeII, nHm
    chunk_size = 5
    n_workers = 8
    chunks = [Egamma_lst[i:i+chunk_size] for i in range(0, len(Egamma_lst), chunk_size)]

    J_lst = []

    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker_chunk, c, mdm, xsection, nbg, rsob): c for c in chunks}
        for fut in as_completed(futures):
            J_lst.extend(fut.result())

    return Egamma_lst, J_lst

# Egamma_current, J_current = initial_JbgAnn(mdm, xsection, filepath)
	
