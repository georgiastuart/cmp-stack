import numpy as np


def mute(data, config):
    """ Mutes a CMP gather along a hyperbola with a cosine taper

    Parameters
    ----------
    config : dict
        Config dictionary as specified in generate_config.py
    """

    taper_length = config['mute_gain_parameters']['taper_length']
    delta_t = config['parameters']['delta_t']
    offset = config['parameters']['min_offset']
    delta_offset = config['parameters']['delta_offset']
    num_receivers = config['parameters']['num_receivers']
    first_arrival = config['mute_gain_parameters']['first_arrival']

    taper = np.arange(0, taper_length + delta_t, delta_t)
    taper = np.cos(taper / taper_length * np.pi / 2)**2
    taper = taper[::-1]

    water_velocity = 1500

    t_0 = np.sqrt(first_arrival**2 - offset**2 / water_velocity**2)
    mute_line = np.zeros(num_receivers)

    for i in range(num_receivers):
        mute_line[i] = np.sqrt(t_0**2 + (offset + i * delta_offset)**2 / water_velocity**2)

    mute_indices = np.ceil((mute_line - taper_length) / delta_t).astype('int_')

    for i in range(num_receivers):
        it_mute = mute_indices[i]
        taper_start = it_mute - len(taper)

        if taper_start < 0:
            taper_start = 0

        index_taper_start = len(taper) - (it_mute - taper_start)

        data[:taper_start, i] = 0
        data[taper_start:it_mute, i] *= taper[index_taper_start:].T

    return mute_line
