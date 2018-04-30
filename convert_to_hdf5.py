import h5py
import numpy as np
import json

if __name__ == '__main__':
    # Converts the unwieldy binaries into nicer hdf5 files

    with open('input/config.json', 'r') as fp:
        config = json.load(fp)

    num_gathers = 19
    num_receivers = config['parameters']['num_receivers']
    num_time_steps = config['parameters']['num_time_steps']

    cmp1900_data = np.fromfile('input/cmp_gathers/19cmp_gather.bin', dtype='float32').astype('float64')
    cmp1900_data = cmp1900_data.reshape((num_receivers * num_gathers, num_time_steps)).T

    reshaped = np.zeros((num_time_steps, num_receivers, num_gathers))

    for i in range(num_gathers):
        reshaped[:, :, i] = cmp1900_data[:, i * num_receivers:(i + 1) * num_receivers]

    with h5py.File('input/cmp19_original.hdf5', 'w') as fp:
        fp.create_dataset('cmp_gathers', data=reshaped)