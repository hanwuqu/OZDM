# Functions about time
from files import read_data
def time(rs):
    from astropy.cosmology import Planck15
    # Unit of output is Gyr
    z = rs - 1
    return Planck15.age(0).value - Planck15.age(z).value

def time_step(rs0, rs1):
    # Unit of output is Gyr
    return time(rs0) - time(rs1)

def rs_time(rs):
    from astropy.cosmology import Planck15
    # Unit of output is Gyr
    z = rs - 1
#    return Planck15.age(0).value - Planck15.age(z).value
    return Planck15.age(z).value

def data_time(rs):
    z = rs - 1
    from astropy.cosmology import Planck15
    t0 = Planck15.age(99).value
    time = Planck15.age(z).value - t0
    return time * 1e3

def rs_point(time):
# This function is used to find the important redshift in data, like the redshift when halo formation and the
# min of redshift
    from scipy import interpolate
    rss = list(i for i in np.linspace(100, 4, 200))
    data = list(data_time(i) for i in rss)
    func = interpolate.interp1d(data, rss)
    return func(time).item()

def rs_to_data(rs):
    rs_min = rs_point(698.96) #from data file
    rs_form = rs_point(300) # redshift that halo formation
    if rs >= rs_min and rs <= rs_form:
        return data_time(rs)
    else:
        return False

def time_to_rs(myr):
# Convert time to redshift, input unit in Myr
    rs, yrs = read_data('input/time_convert_new.txt')
    from scipy import interpolate
    func = interpolate.interp1d(yrs, rs)
    return func(myr).item()
    
def new_drs(rs, timestep):
# This function obtains the new redshift step, timestep refers to 
# the new timestep from the cooling time in Myr
    current_age = rs_time(rs) * 1e3
    new_age = current_age + timestep
    new_rs = time_to_rs(new_age)
    drs = abs(new_rs - rs)
    return drs
     
