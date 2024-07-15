import torch
import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from BioMime.utils.params import coeff_r_a, coeff_r_b, coeff_fb_a, coeff_fb_b, coeff_a_a, coeff_a_b, coeff_iz_a, coeff_iz_b, coeff_cv_a, coeff_cv_b, coeff_len_a, coeff_len_b, w_amp

def get_curr_angle_muap(curr_angle, unit_muaps_angles, angle_labels):
    """Get the MUAP corresponding to the current angle

    Args:
        curr_angle (float): current angle
        unit_muaps_angles (np.ndarray): Array of MUAPs with shape (units, morphs, ch_rows, ch_cols, samples)
        angle_labels (np.ndarray): Range of generated angles

    Returns:
        np.ndarray: MUAPs corresponding to the current angle
    """    

    idx = np.argmin( np.abs(angle_labels - curr_angle) )
    muap = unit_muaps_angles[idx]

    return muap


def generate_emg(muaps, spikes, muap_angle_labels, angle_profile):
    """
    Generate EMG signal based on MUAPs, spikes, muap_angle_labels, and angle_profile.

    Parameters:
    - muaps (ndarray): Array of MUAPs with shape (_, _, ch_rows, ch_cols, win).
    - spikes (list): List of spike timings for each unit.
    - muap_angle_labels (list): List of angle labels for each MUAP.
    - angle_profile (list): List of angles corresponding to each spike timing.

    Returns:
    - emg (ndarray): Generated EMG signal with shape (ch_rows, ch_cols, time_samples).
    """

    # Initialise dimensions
    _, _, ch_rows, ch_cols, win = muaps.shape
    offset = win//2
    
    # Check number of active units
    units_active = 0
    for sp in spikes:
        if len(sp) > 0:
            units_active += 1

    # Initialise emg
    time_samples = len(angle_profile)
    emg = np.zeros((ch_rows, ch_cols, time_samples))

    # Add each unit's contribution
    for unit in range(units_active):
        unit_firings = spikes[unit]

        if len(unit_firings) == 0:
            continue

        for firing in unit_firings:
            # Get the corresponding MUAP morphing for each firing
            curr_angle = angle_profile[firing]
            curr_muap = get_curr_angle_muap(curr_angle, muaps[unit], muap_angle_labels)

            # Deal with edge cases
            init_emg = np.max([0, firing-offset])
            end_emg = np.min([firing+offset, time_samples])

            init_muap = init_emg - (firing-offset)          # 0 if the window is inside the range
            end_muap = end_emg - (firing+offset) + offset*2   # win if the window is inside the range

            # Add contribution to EMG
            emg[:,:, init_emg:end_emg] += curr_muap[:,:,init_muap:end_muap]

    return emg


def generate_emg_mu(muaps, spikes, time_samples):
    """
    Args:
        muaps (np.array): [time_steps, nrow, ncol, duration]
        spikes (list): indices of spikes
        time_samples (int): fs * movement_time

    Return:
        EMG (np.array): [nrow, ncol, time_samples]
    """

    muap_steps, nrow, ncol, time_length = muaps.shape
    emg = np.zeros((nrow, ncol, time_samples + time_length))
    for t in spikes:
        muap_time_id = get_cur_muap(muap_steps, t, time_samples)
        emg[:, :, t:t + time_length] = muaps[muap_time_id]

    return emg


def normalise_properties(db, num_mus, steps=1):

    num = torch.from_numpy((db['num_fibre_log'] + coeff_fb_a) * coeff_fb_b).reshape(num_mus, 1).repeat(1, steps)
    depth = torch.from_numpy((db['mu_depth'] + coeff_r_a) * coeff_r_b).reshape(num_mus, 1).repeat(1, steps)
    angle = torch.from_numpy((db['mu_angle'] + coeff_a_a) * coeff_a_b).reshape(num_mus, 1).repeat(1, steps)
    iz = torch.from_numpy((db['iz'] + coeff_iz_a) * coeff_iz_b).reshape(num_mus, 1).repeat(1, steps)
    cv = torch.from_numpy((db['velocity'] + coeff_cv_a) * coeff_cv_b).reshape(num_mus, 1).repeat(1, steps)
    length = torch.from_numpy((db['len'] + coeff_len_a) * coeff_len_b).reshape(num_mus, 1).repeat(1, steps)

    base_muap = db['muap'].transpose(0, 3, 1, 2) * w_amp
    base_muap = torch.from_numpy(base_muap).unsqueeze(1).float()

    return num, depth, angle, iz, cv, length, base_muap


def get_cur_muap(muap_steps, cur_step, time_samples):
    return int(muap_steps * cur_step / time_samples)

    
def plot_spike_trains(spikes, fs=None, ax=None, pth=None):
    """
    Plot spike trains.

    Args:
        spikes (list): List of spike times for each unit.
        fs (float, optional): Sampling frequency. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.
        pth (str, optional): Path to save the figure. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The plotted axes object.
    """

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,4))
    num_mu = len(spikes)
    for mu in range(num_mu):
        spike = spikes[mu] #np.array(spikes[mu])
        if fs is None:
            ax.vlines(spike, mu, mu + 0.5, linewidth=1.0)
        else:
            ax.vlines(spike/fs, mu, mu + 0.5, linewidth=1.0)
    if fs is None:
        ax.set_xlabel('Time (samples)')
    else:
        ax.set_xlabel('Time (s)')
    ax.set_ylabel('MU Index')
    if pth:
        plt.savefig(pth)

    return ax


def ensure_spikes_in_range(spikes, samples):
    units = len(spikes)
    for unit in range(units):
        curr_spikes = np.array(copy(spikes[unit]))
        valid_spikes = np.logical_and( curr_spikes>=0, curr_spikes<samples)
        spikes[unit] = curr_spikes[valid_spikes]

    return spikes


def spikes_to_array(spikes):
    """
    Converts a list of firings times for each unit into an array of spikes.

    Parameters:
    spikes (list): A list of spike times for each unit.

    Returns:
    (np.ndarray): An array of spikes, where each element represents the firing
        times for a unit.

    """
    units = len(spikes)
    out = np.empty((units), dtype=object)
    for unit in range(units):
        out[unit] = np.array(spikes[unit])
    return out


def spikes_to_bin(spikes, time_samples):
    """
    Formats the given spike data into a binary matrix.

    Parameters:
    spikes (list): A list of spike times for each unit.
    time_samples (int): The total number of time samples.

    Returns:
    numpy.ndarray: A binary matrix with shape (time_samples, units), where
        1s represent firing times.
    """
    units = len(spikes)
    bin_spikes = np.zeros((time_samples, units)).astype(int)
    for unit in range(units):
        bin_spikes[spikes[unit].astype(int), unit] = 1
    return bin_spikes


def bin_to_spikes(bin_spikes):
    """
    Converts a binary spike array to a list of firing times for each unit.

    Parameters:
    bin_spikes (ndarray): A binary matrix with shape (time_samples, units),
        where 1s represent firing times.

    Returns:
    spikes (ndarray): List of firing times for each unit.
    """
    _, units = bin_spikes.shape
    spikes = np.empty((units), dtype=object)
    for unit in range(units):
        spikes[unit] = np.nonzero( bin_spikes[:, unit] )[0]
    return spikes
