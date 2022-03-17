"""
Generates N-18 compatible catalogues of potentially detectable sources (i.e. with a higher
approximate signal to noise than a given one), including SOBBH background amplitude.
"""

import os
import numpy as np
from tqdm import tqdm

# User parameters ########################################################################
n_realisations = 50
SNR_min = 6  # slightly smaller SNR than desired (e.g. 4-6 if target is 8)
output_folder = f"output/N18_SNR_min_{SNR_min}"
max_coal_time_yr = 2000
max_z = 0.5
max_z_background = 5
verbose = True

# Load and prepare data ##################################################################

try:
    from extrapops.population import Population
except ImportError:
    raise ImportError("You need `extrapops` to run this script. Contact Jesus Torrado")

try:
    from sgwb_common import LISA_noise as Ln
except ImportError:
    raise ImportError("You need `sgwb_common` to run this script. "
                      "Contact Jesus Torrado or Mauro Pieroni.")

# N18 parameters
population_params = {
    "cosmo_params": {
        "H_0": 67.9,  # Hubble constant in Km/(s*MPc)
        "Omega_m": 0.3065,
        "Omega_r": 9e-5,
        "Omega_l": 0.7,
        "Omega_k": 0.},
    "redshift_params": {
        "z_range": [1e-5, max_z],
        "z_perdecade": 500,
        "T_yr": max_coal_time_yr,
        "merger_rate_model": "madau",
        "merger_rate_params": {"R_0": 28.1, "z_0": 0.2, "d": 2.7}},
    "mass_params": {
        "m_range": [2.5, 100.],
        "alpha": 3.4,
        "beta_q": 1.1,
        "delta_m": 7.8,
        "lambda_peak": 0.039,
        "mu_m": 34,
        "sigma_m": 5.09},
    "spin_params": {
        "expected_a": 0.25,
        "var_a": 0.03,
        "a_max": 1.,
        "zeta_spin": 0.76,
        "sigma_t": 0.8}
}

# Background
f_min, f_max = 3e-5, 0.5  # Hz
fs = np.logspace(np.log10(f_min), np.log10(f_max), 100)
resp_AA = Ln.LISA_response_function(fs, channel='AA')

# Iterate, generate and save #############################################################

for i in tqdm(range(n_realisations)):
    # Create folder containing files
    this_dir = os.path.join(output_folder, f"{i}")
    try:
        os.makedirs(this_dir)
    except FileExistsError:
        raise FileExistsError(
            f"Folder {this_dir} exists. Check that you are not accidently overwriting a "
            "previous calculation. Delete the output tree manually before re-runnning.")
    # Generate and save
    # NB: without SNR and with max_tauc=2000 and max_z=0.5, it's 2M events.
    # TODO: for now, applying SNR cut post-generation
    this_pop = Population(lazy=False, **population_params)
    n_pre_cut = len(this_pop)
    this_pop.add_snr_avg_inclination()  # for later consistency checks, but not needed
    this_pop.add_snr_max_inclination()
    # Apply SNR cut -- don't miss detectable events! <-- using max-over-inclination SNR
    this_pop._data = this_pop._data[this_pop._data["snr_max_inclination"] > SNR_min]
    if verbose:
        print(f"{n_pre_cut} events generated before SNR cut.")
        print(f"{len(this_pop)} kept after SNR_max > {SNR_min} cut.")
        for c in [4, 6, 8]:
            print(f"- SNR_avg > {c}:",
                  len(this_pop._data[this_pop._data["snr_avg_inclination"] > c]))
    # Save population and parameters,
    # both LISA and plain formats, since plain one preserves approx SNR columns
    this_pop.save(os.path.join(this_dir, "population"), LISA=True)
    this_pop.save(os.path.join(this_dir, "population"), LISA=False)
    # Add background
    if i == 0:
        h2s = this_pop.total_strain_squared_avg(
            fs, z_max=max_z_background, use_cached=False)
        h2s_TDI_units = h2s * resp_AA / 2 / fs
    np.savetxt(os.path.join(this_dir, "background.txt"), np.array([fs, h2s_TDI_units]).T)
