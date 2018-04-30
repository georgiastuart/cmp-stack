from cmp_stack import RadonTransform, NormalMoveOut, Mute
from cmp_stack.utilities import master
import numpy as np


def stack(data):
    """ Stacks a treated CMP gather
    Parameters
    ----------
    data : Numpy array
        A post-treatment CMP gather with axis 0 time and axis 1 offset
    """
    num_nonzero = (data != 0).sum(axis=1)
    num_nonzero[num_nonzero == 0] = 1
    vec = np.sum(data, axis=1) / num_nonzero
    return vec


def cmp_stack(data, config, velocity, mode='multiples'):
    """ Creates a CMP  Stack Section.
    1. Compute Normal Move Out for each gather using the velocity profiles provided
    2. Mutes the data via spline interpolation with a cosine taper
    3. Models either primaries or multiples with a Radon transform
    4. Mute the data via spline interpolation with no taper
    5. Stacks the treated data into a stack section

    Parameters
    ----------
    data : Numpy array
        An array of CMP gathers with axes (time, offset, gather).
        Assumes the direct wave has already been muted
    config : dict
            Config dictionary as specified in generate_config.py
    velocity : Numpy array
        An array of velocity profiles for each CMP gather. Axes are (time, gather)
    mode : str
        Whether to generate the CMP stack by modeling multiples and subtracting ('multiples') or
        by modeling primary reflections ('primaries')
    """

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
        mute(nmo.data_nmo)
        radon_transform(nmo.data_nmo)

        if mode == 'multiples':
            diff = nmo.data_nmo - radon_transform.inverted_data
        elif mode == 'primaries':
            diff = radon_transform.inverted_data
        else:
            raise AttributeError

        mute(diff, taper=False)
        stack_section[:, i] = stack(diff)

    return stack_section






