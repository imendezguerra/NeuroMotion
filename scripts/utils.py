import h5py
import numpy as np

def save_gen_data(file_name, muaps, mode, muscle, num_mus, fs_mov, poses, durations, changes, fs, num, depth, angle, iz, cv, length):

    # Save all
    with h5py.File(file_name, 'w') as h5:
        
        # MUAPs
        h5.create_dataset('muaps', data = muaps)
        
        # BioMime parameters
        biomime_params = h5.create_group('BioMime_params')
        biomime_params.create_dataset('mode', data = mode)
        biomime_params.create_dataset('ms_label', data = muscle)
        biomime_params.create_dataset('num_mus', data = num_mus)
        biomime_params.create_dataset('fs_muaps', data = fs_mov)
        biomime_params.create_dataset('poses', data = poses)
        biomime_params.create_dataset('durations', data = durations)
        biomime_params.create_dataset('unit_lengths', data = changes['len'])
        biomime_params.create_dataset('unit_cvs', data = changes['cv'])
        biomime_params.create_dataset('unit_depths', data = changes['depth'])

        # Neural model parameters
        neural_params = h5.create_group('neural_params')
        neural_params.create_dataset('fs_spikes', data = fs)
        neural_params.create_dataset('num_fibres', data = num)
        neural_params.create_dataset('depth', data = depth)
        neural_params.create_dataset('angle', data = angle)
        neural_params.create_dataset('iz', data = iz)
        neural_params.create_dataset('cv', data = cv)
        neural_params.create_dataset('length', data = length)

def load_gen_data(file_name):

    with h5py.File(file_name, 'r') as h5:

        sim = {}

        # MUAPs
        sim['muaps'] = h5['muaps'][()]

        sim['BioMime_params'] = {}
        for key in h5['BioMime_params'].keys():
            sim['BioMime_params'][key] = h5['BioMime_params'][key][()]

        sim['neural_params'] = {}
        for key in h5['neural_params'].keys():
            sim['neural_params'][key] = h5['neural_params'][key][()]
    
    return sim