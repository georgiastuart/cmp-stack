from cmp_stack import RadonTransform, NormalMoveOut, Mute
from cmp_stack.utilities import master
import numpy as np


def stack(data):
    num_nonzero = (data != 0).sum(axis=1)
    num_nonzero[num_nonzero == 0] = 1
    vec = np.sum(data, axis=1) / num_nonzero
    return vec


# Assumes data is in 3D numpy array
def cmp_stack(data, config, velocity, mode='multiples'):
    nmo = NormalMoveOut(config)
    radon_transform = RadonTransform(config, mode=mode)
    mute = Mute(config)

    num_time_steps = data.shape[0]
    num_receivers = data.shape[1]
    num_gathers = data.shape[2]

    stack_section = np.zeros((num_time_steps, num_gathers))

    for i in range(num_gathers):
        if master:
            print('Running stack {}...'.format(i), flush=True)
        data_slice = data[:, :, i]
        nmo(data_slice, velocity[:, i])
        mute.mute(nmo.data_nmo)
        radon_transform(nmo.data_nmo)

        if mode == 'multiples':
            diff = nmo.data_nmo - radon_transform.inverted_data
        elif mode == 'primaries':
            diff = radon_transform.inverted_data
        else:
            raise AttributeError

        mute.mute(diff, taper=False)
        stack_section[:, i] = stack(diff)

    return stack_section






