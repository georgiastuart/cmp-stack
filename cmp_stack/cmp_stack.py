from cmp_stack import RadonTransform, NormalMoveOut
from cmp_stack.utilities import master
import numpy as np


# Assumes data is in 3D numpy array
def cmp_stack(data, config):
    nmo = NormalMoveOut(config)
    radon_transform = RadonTransform(config, mode='multiples')

    num_time_steps = data.shape[0]
    num_receivers = data.shape[1]
    num_gathers = data.shape[2]

    stack_section = np.zeros((num_time_steps, num_gathers))

    for i in range(num_gathers):
        if master:
            print('Running stack {}...'.format(i), flush=True)
        data_slice = data[:, :, i]
        nmo(data_slice)
        radon_transform(nmo.data_nmo)
        stack = np.sum(nmo.data_nmo - radon_transform.inverted_data, axis=1)
        stack_section[:, i] = stack

    return stack_section






