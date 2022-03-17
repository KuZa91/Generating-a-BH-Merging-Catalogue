"""
Loads a LIGO population posterior table, including SOBBH background amplitude,
and generates events with a higher approximate signal to noise than a given one.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# User parameters ########################################################################
posterior_filename = "data/o1o2o3_post_background.pkl"
SNR_min = 6  # slightly smaller SNR than desired (e.g. 4-6 if target is 8)
output_folder = f"output/SNR_min_{SNR_min}"
max_coal_time_yr = 2000
max_z = 0.5
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

# Columns for background
_h2_1em2_colname = "h2_1em2"
_Omh2_1em2_colname = "Omh2_1em2"

# Load posterior containing amplitudes
posterior_df = pd.read_pickle(posterior_filename)
n_total = len(posterior_df)
# Select the computed part only
i_computed = np.argwhere(
    np.isfinite(posterior_df[_Omh2_1em2_colname].to_numpy())).T[0]
posterior_df = posterior_df.iloc[i_computed]
# Exclude failed computations (for now, Omh2 > 1)
posterior_df = posterior_df[posterior_df[_Omh2_1em2_colname] < 1]
print(f"Number of samples: {len(i_computed)} "
      f"(out of {n_total}; {len(i_computed) - len(posterior_df)} failed).")

merger_rate_labels = {"R_0": "rate", "d": "lamb"}
mass_labels = {"m_min": "mmin", "m_max": "mmax", "alpha": "alpha", "beta_q": "beta",
               "delta_m": "delta_m", "lambda_peak": "lam", "mu_m": "mpp",
               "sigma_m": "sigpp"}
spin_labels = {"expected_a": "mu_chi", "var_a": "sigma_chi", "a_max": "amax",
               "zeta_spin": "xi_spin", "sigma_t": "sigma_spin"}


def update_params(params_dict, params_labels, point):
    """Helper functions to turn LIGO into extrapops paramters."""
    params_dict.update({p: point[p_ligo] for p, p_ligo in params_labels.items()})


# Common parameters
# Values used is LIGO/Virgo O3 Populations paper:
# https://arxiv.org/abs/2111.03634
cosmo_params = {
    "H_0": 67.9,  # Km / (s MPc)
    "Omega_m": 0.3065,
    "Omega_r": 9e-5,
    "Omega_l": 0.7,
    "Omega_k": 0.}
redshift_params = {
    # Redshift range and precision:
    # min set to avoid SNR divergence due to extremely close events
    "z_range": [1e-5, max_z],
    "z_perdecade": 500,
    # Total time over which events are generated. By default,
    # = max years of coalescence time for a BBH mergine event
    # in detector frame
    "T_yr": max_coal_time_yr,
    # Merger rate model: "const" or "evol"
    "merger_rate_model": "madau",
    "merger_rate_params": {"R_0": None, "d": None,  # per posterior sample
                           "z_0": 0, "r": -2.9, "z_peak": 1.86, "c": None}  # common
}
mass_params = {}
spin_params = {}

# Background
f_reference = 0.01  # Hz
f_min, f_max = 3e-5, 0.5  # Hz
fs = np.logspace(np.log10(f_min), np.log10(f_max), 100)
resp_AA = Ln.LISA_response_function(fs, channel='AA')


def generate_char_strain_sq_unitless_SOBBH(fs, A, f_reference):
    """
    Generates characteristic unitless strain-squared for the SOBBH background at the given
    frequencies.
    """
    h2 = A * (fs / f_reference)**(-4 / 3)
    # TDI units
    h2 *= resp_AA / 2 / fs
    return h2


# Iterate over samples, generate and save ################################################

for i, row in tqdm(posterior_df.iterrows(), total=len(posterior_df)):
    # Create folder containing files
    this_dir = os.path.join(output_folder, f"{i}")
    try:
        os.makedirs(this_dir)
    except FileExistsError:
        raise FileExistsError(
            f"Folder {this_dir} exists. Check that you are not accidently overwriting a "
            "previous calculation. Delete the output tree manually before re-runnning.")
    # Prepare and individual population parameters
    update_params(redshift_params["merger_rate_params"], merger_rate_labels, row)
    update_params(mass_params, mass_labels, row)
    mass_params["m_range"] = [mass_params.pop("m_min"), mass_params.pop("m_max")]
    update_params(spin_params, spin_labels, row)
    # Generate and save
    # NB: without SNR and with max_tauc=2000 and max_z=0.5, it's 2M events.
    # TODO: for now, applying SNR cut post-generation
    this_pop = Population(cosmo_params=cosmo_params, redshift_params=redshift_params,
                          mass_params=mass_params, spin_params=spin_params, lazy=False)
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
    h2s_TDI_units = generate_char_strain_sq_unitless_SOBBH(
        fs, row[_h2_1em2_colname], f_reference)
    np.savetxt(os.path.join(this_dir, "background.txt"), np.array([fs, h2s_TDI_units]).T)
