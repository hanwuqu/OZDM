import numpy as np
import os
from one_zone import internal_energy
from files import new_initial, save_dict

def grackle_initial(data_path, rs, UVB=0):
    rs, density, temperature, HI, HII, HeI, HeII, HeIII, de, H2I, H2II, HM, energy, pressure, mu, gamma = new_initial(data_path)
    from pygrackle import chemistry_data, FluidContainer

    from pygrackle.utilities.physical_constants import \
        mass_hydrogen_cgs, \
        mass_electron_cgs, \
        sec_per_Myr, \
        cm_per_mpc, \
        gravitational_constant_cgs
    tiny_number = 1e-60
#    current_redshift = 0.
    current_redshift = rs
    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 1
    my_chemistry.primordial_chemistry = 2
    my_chemistry.UVbackground = UVB
#    my_dir = os.path.dirname(os.path.abspath(__file__))
    my_dir = data_path
    grackle_data_file = bytearray(os.path.join(my_dir, "input", "CloudyData_UVB=HM2012.h5"), 'utf-8')
#    print(grackle_data_file)
    my_chemistry.grackle_data_file = grackle_data_file
#    my_chemistry.UVbackground_redshift_on = 200
    my_chemistry.self_shielding_method = 0
    my_chemistry.H2_self_shielding = 0
    my_chemistry.Gamma = gamma
    my_chemistry.CaseBRecombination = 1
    my_chemistry.cie_cooling = 1
    my_chemistry.h2_optical_depth_approximation = 1
    my_chemistry.interstellar_radiation_field = 0.
    my_chemistry.h2_h_cooling_rate = 2
    my_chemistry.metal_cooling = 0
    my_chemistry.three_body_rate = 0
    # Set units
    my_chemistry.cmb_temperature_floor = 1
    my_chemistry.comoving_coordinates = 0# proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1. / (1. + current_redshift) / \
        my_chemistry.a_units
    my_chemistry.density_units  = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units   = cm_per_mpc        # 1 Mpc in cm
    my_chemistry.time_units     = sec_per_Myr       # 1 Myr in s
    my_chemistry.set_velocity_units()
    rval = my_chemistry.initialize()
    fc = FluidContainer(my_chemistry, 1)

    fc["density"][:] = density / my_chemistry.density_units
    fc["temperature"][:] = temperature
    fc["HI"][:] = HI / 1.67e-24
    fc["HII"][:] = HII / 1.67e-24
    fc["HeI"][:] = HeI / 1.67e-24
    fc["HeII"][:] = HeII / 1.67e-24
    fc["HeIII"][:] = HeIII / 1.67e-24
#    fc["de"][:] = de / 1.67e-24
    fc["de"][:] = de / 9.109e-28 
    if my_chemistry.primordial_chemistry > 1:
        fc["H2I"][:] = H2I / 1.67e-24
#        print(fc['H2I'])
        fc["H2II"][:] = H2II / 1.67e-24
        fc["HM"][:] = HM / 1.67e-24
    fc["x-velocity"][:] = 0.0
    fc["y-velocity"][:] = 0.0
    fc["z-velocity"][:] = 0.0
    safety_factor = 0.01
    fc["energy"][:] = energy
    return fc

# Add a function to calculate the mean molecular mass, may store that in the
# raw data file for the further calculation

def mean_molecular_weight(fc, density):
    HI = fc['HI'][0]
    HII = fc['HII'][0]
    HeI = fc['HeI'][0] / 4 # because of the initial of grackle, the unit
    HeII = fc['HeII'][0] / 4
    HeIII = fc['HeIII'][0] / 4
    H2I = fc['H2I'][0] / 2
    H2II = fc['H2II'][0] / 2
    Hm = fc['HM'][0]
    de = fc['de'][0]
    mu = HI + HII + HeI + HeII + HeIII + H2I + H2II + Hm + de
    return density / mu


# Evolution Functions
def evolve_step(fc, rs, current_time, time_step, final_density, safety_factor=0.01, include_pressure=True):
    my_chemistry = fc.chemistry_data
    current_time -= time_step
    density_old = fc['density'][0]
    density_new = final_density
    density_ratio = density_ratio = density_new / fc["density"][0]

    # update densities
    for field in fc.density_fields:
        fc[field] *= density_ratio

    # now update energy for adiabatic heating from collapse
    if density_ratio != 1:
        fc["energy"][0] *= internal_energy(density_old, density_new, my_chemistry.Gamma)
    else:
#        fc['energy'][0] = fc['temperature'][0] / fc.chemistry_data.temperature_units / (1.297750723207076 - 1) 
#        fc['energy'][0] = 3886.9828409699967 / fc.chemistry_data.temperature_units * 1.2 
        fc['energy'][0] *= 1
    fc.solve_chemistry(time_step)
    fc.calculate_pressure()
    fc.calculate_temperature()
    fc.calculate_cooling_time()
    mu = mean_molecular_weight(fc, density_new)
    fc.calculate_gamma()
#    fc.solve_chemistry(time_step)
    data_new = save_dict(fc, rs, mu)
#    cooling_info = [rs, fc['density'][0], fc['energy'][0], fc['cooling_time'][0]]
#    print(fc['cooling_time'][0])
    cooling_time = np.abs(fc['cooling_time'][0])
    temperature = fc['temperature'][0]
    return data_new, cooling_time, temperature, mu

