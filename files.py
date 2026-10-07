import os
import numpy as np
from initial_CLASS import initial_conditions_CLASS
from initial_conditions import initial_data
# Modify at 22th July 2025, include CLASS initial conditions
# Functions for data dealing and grackle, need to be updated

def read_data(path):
    data = np.loadtxt(path)
    data0 = data[:, 0]
    data1 = data[:, 1]
    return data0, data1

def read_datafile(path, name):
    with open(path, 'r') as file:
        keys = file.readline().strip().split('  ')
    if name in keys:
        index = keys.index(name)
        data = np.loadtxt(path, skiprows=1)
        out = data[:, index]
    else:
        out = 'NaN'
    return out

key_list = ['redshift', 'density', 'temperature', 'HI', 'HII', 'HeI', 'HeII', 'HeIII', 'de', 'H2I', 'H2II', 'HM', 'energy', 'pressure', 'mu', 'gamma']

def initial_file(path, rs, density, method='CLASS'):
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path + '/raw_data.txt'
    # The names of variables
    line = '  '.join(key_list)
    with open(file_path, 'w') as file:
        file.write(line + '\n')
    # The values
#    initial_list = ['200', '8.337740505147267e-23', '265.61701587670353', '5.201213510567307e-23', '1.761390160568419e-27', '1.6425440998679904e-23', '4.403747014055844e-36', '2.0548944821714217e-42', '1.761390161669357e-27', '8.337740505147267e-32', '8.337740505147267e-32', '8.337740505147267e-32', '2.8207686789203255e-12', '0']
    outputs = initial_data(rs, density, method)
    initial_list = [str(item) for item in outputs]
    linev = '  '.join(initial_list)
    with open(file_path, 'a') as file:
        file.write(linev + '\n')
    return 0

def save_dict(fc, rs, mu):
    my_chemistry = fc.chemistry_data
    data_dict = {}
# Modify 13.Jan.2025
    
    for i in key_list:
        if i == 'temperature' or i == 'pressure' or i == 'energy':
            data_dict[i] = str(fc[i][0])
        elif i == 'redshift':
            data_dict[i] = str(rs)
        elif i == 'de':
            data_dict[i] = str(fc[i][0] * 9.109e-28)
        elif i == 'mu':
            data_dict[i] = str(mu)
        elif i == 'gamma':
            data_dict[i] = str(fc['gamma'][0])
        else:
            data_dict[i] = str(fc[i][0] * my_chemistry.density_units)
    return data_dict

def save_data(dicts, path):
    value_list = []
    for i in key_list:
        value_list.append(dicts[i])
    linev = '  '.join(value_list)
    file_path = path + '/raw_data.txt'
    with open(file_path, 'a') as file:
        file.write(linev + '\n')

def new_initial(path):
    file_path = path + '/raw_data.txt'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file at '{path}' does not exist.")
    data = np.loadtxt(file_path, dtype=str)
    last_line = data[-1]
#    print(last_line)
    value = list(eval(i) for i in last_line)
    return value

def moniter(path):
    file_path = path + '/raw_data.txt'
    with open(file_path, 'r') as file:
        keys = file.readline().strip().split('  ')
    index = keys.index('temperature')
    data = np.loadtxt(file_path, skiprows=1)[-1]
    temp = data[index]
    return temp

def modify(path, vir_Temp, vir_energy):
#    energy = vir_Temp / 115826294461905.86 * 1.2 # Just use values of some unit convert
    file_path = path + '/raw_data.txt'
    with open(file_path, 'r') as file:
        keys = file.readline().strip().split('  ')
    index0 = keys.index('temperature')
    index1 = keys.index('energy')
#    index2 = keys.index('redshift')
    data = np.loadtxt(file_path, skiprows=1)
    temp = data[-1]
    temp[index0] = vir_Temp
    temp[index1] = vir_energy
#    temp[index2] = vir_rs
    temps = list(str(i) for i in temp)
    linev = '  '.join(temps)
#    print(linev)
    with open(file_path, 'a') as file:
        file.write(linev + '\n')

def self_shielding_prepare(path):
    file_path = path + '/raw_data.txt'
    with open(file_path, 'r') as file:
        keys = file.readline().strip().split('  ')
    temp_index = keys.index('temperature')
    den_index = keys.index('density')
    H2_index =keys.index('H2I')
    HI_index = keys.index('HI')
    HeI_index = keys.index('HeI')
    HeII_index = keys.index('HeII')
    element_list = ['HI', 'HII', 'HeI', 'HeII', 'HeIII', 'de', 'H2I', 'H2II', 'HM']
    index_list = list(keys.index(i) for i in element_list)
    data = np.loadtxt(file_path, skiprows=1)
    last = data[-1]
    temperature = last[temp_index]
    density = last[den_index]
    H2 = last[H2_index]
    HI = last[HI_index]
    HeI = last[HeI_index]
    HeII = last[HeII_index]
    value_list = list(last[i] for i in index_list)
    sum = value_list[0] + value_list[1] + value_list[5] + value_list[8] + 4 * (value_list[2] + value_list[3] + value_list[4]) + 2 * (value_list[6] + value_list[7])
    mean_molecular_weight = density / sum
    return temperature, density, H2, HI, HeI, HeII, mean_molecular_weight

def cooling_information_save(lst, path):
    file_path = path + '/cooling_raw_data.txt'
    with open(file_path, 'a') as file:
        file.write(str(lst[0]) + '  ' + str(lst[1]) + '  ' + str(lst[2]) + '  ' + str(lst[3]) + '\n')   

def self_shielding_data(rs, filepath):
    from scipy.interpolate import interp1d
    data = np.loadtxt(filepath, skiprows=1)  # skip first line with names
    A0 = data[:, 0]
    results = []
    for col in range(1, data.shape[1]):
        f = interp1d(A0, data[:, col])
        results.append(f(rs))
    return tuple(results)

def last_line(filepath):
    with open(filepath, "r") as f:
        last = f.readlines()[-1].strip()
    numbers = [float(x) for x in last.split()]
    return numbers

def ssh_prepare(noss_path, ssh_path):
    import shutil
    halo_file_origin = noss_path + '/top_hat_halo.txt'
    flux_file_origin = noss_path + '/J_flux.h5'
    halo_file_new = ssh_path
    flux_file_new = ssh_path
    shutil.copy(halo_file_origin, halo_file_new)
    shutil.copy(flux_file_origin, flux_file_new)
    return 0

def ssh_record(ssh_path, rs, fion, fH2, rs_ini=900):
    txt_path = ssh_path + '/ssh_record.txt'
    if rs == rs_ini:
        with open(txt_path, 'w') as txt:
            txt.write(str(rs) + '  ' + str(fion) + '  ' + str(fH2) + '\n')
    else:
        with open(txt_path, 'a') as txt:
            txt.write(str(rs) + '  ' + str(fion) + '  ' + str(fH2) + '\n')
    return 0

def ssh_fitting(key):
    txt_path = 'input/fitting_parameter.txt'
    data = np.genfromtxt(
    txt_path,
    dtype=[("name", "U10"), ("v1", float), ("v2", float), ("v3", float), ("v4", float)]
)
#    key = str(key)
    mask = data["name"] == key
    if not mask.any():
        raise ValueError(f"{key} not found")
    row = data[mask][0]
    return row["v1"], row["v2"], row["v3"], row["v4"]
