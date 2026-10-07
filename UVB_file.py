import h5py
import numpy as np
def dataset_setup(data_value, set_name, folder_path):
    file_path = folder_path + '/input/CloudyData_UVB=HM2012.h5' # Change this path for new background
    with h5py.File(file_path, 'r+') as h5file:
        dataset = h5file['/UVBRates/' + set_name]
        new_data = np.full(dataset.shape, data_value)
        dataset[:] = new_data

def construct_rs_database(rs, folder_path):
    file_path = folder_path + '/input/CloudyData_UVB=HM2012.h5'
    value = rs - 1  # center value
    length = 59
    center_idx = length // 2  # index of center
    start_val = max(0, value - center_idx)
    before = np.linspace(start_val, value - 1, center_idx)
    after = np.linspace(value + 1, value + (length - center_idx - 1), length - center_idx - 1)
    arr = np.concatenate([before, [value], after])
    new_data = arr
    with h5py.File(file_path, 'r+') as h5file:
        z_dataset = h5file['UVBRates/z']
        if len(new_data) != len(z_dataset):
            raise ValueError("New data length must match existing dataset length.")
        z_dataset[:] = new_data  # overwrite in place



def setupUVBFile(rs, k27, k31, kion, k28, k29, k30, heatRate, folder_path):
# For photoionization and photoheating, I will set a kind of switch to turn on
# or turn off that. 7.1.2025
# Modification of photoionization and photoheating, 2.9.2025
    kH, kHe, kHep = kion
    heatH, heatHe, heatHep = heatRate 
    update_dict = {}
    fa = 1e-60
    fb = 1e-60
    #Ionization rates
    k24 = fa * 1.94e-16  + kH # H + gam -> H+ + e-
    k25 = fa * 2.02e-23  + kHep # He+ + gam -> He++ + e-
    k26 = fa * 2.68e-16 + kHe # He + gam -> He+ + e-
    k28 = k28 #  H2+ + gam -> H + H+.
    k29 = fa * 3.4e-16 + k29#  H2 + gam -> H2+ + e-
    ##TODO, update calculation of k30
    k30 = k30 #  H2+ + gam -> H+ + H+ + e-

    #Photoheating
    piHI = fb * 1.16e-15 + heatH
    piHeI = fb * 2.8e-15 + heatHe
    piHeII = fb * 1.54e-20 + heatHep
    update_dict['Chemistry'] = {'k24':k24, 'k25':k25, 'k26':k26, 'k27':k27, 'k28':k28, 'k29':k29, 'k30':k30, 'k31':k31}
    update_dict['Photoheating'] = {'piHI':piHI, 'piHeI':piHeI, 'piHeII':piHeII}
    type_list = list(update_dict.keys())
    for i in type_list:
        reaction_list = list(update_dict[i].keys())
        for j in reaction_list:
            set_name = i + '/' + j
            set_value = update_dict[i][j]
            dataset_setup(set_value, set_name, folder_path)
    construct_rs_database(rs, folder_path)
