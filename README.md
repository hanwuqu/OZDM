# OZDM
This is the code to simulate the thermal and ionization history of gas cloud evolution via Grackle.
Initial conditions of gas cloud come from CLASS.
nss_runner.py and ss_runner.py are the runners of the code, the parameters like mass, lifetime/ cross-section of DM and gas cloud mass can be adjusted via:
python (n)ss_runner.py --mchi --tau(xsection) --Mcloud
Currently the sources of photon flux include CMB, DM in background and DM halo.
To run the code, CLASS, Grackle, h5py, numpy are required.
