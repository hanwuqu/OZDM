import shutil
import os

def data_initial(folder_path):
# This path is the path of data folder, in general, it's in the decay
# or annihilation, dark matter mass, lifetime or cross section and 
# mass of cloud since I won't run self-shielding and non self-
# shielding cases at the same time
    folder_path = folder_path + '/input'
    origin_path = 'input/CloudyData_UVB=HM2012.h5'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    shutil.copy(origin_path, folder_path)
