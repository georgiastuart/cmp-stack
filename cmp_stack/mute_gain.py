import numpy as np
from scipy.interpolate import UnivariateSpline


def gain(data, config, gain_power=2):
    """Gains data with a :math:`t^{p}` gain function.

    Parameters
    ----------
    data : Numpy array
        CMP Gather to gain
    config : dict
        Config dictionary as specified in generate_config.py
    gain_power : float
        power, :math:`p`, to raise time to for the gain

    """
    num_time_steps = config['parameters']['num_time_steps']
    delta_t = config['parameters']['delta_t']

    gain_coef = np.arange(0, num_time_steps * delta_t, delta_t)**gain_power

    gained_data = (data.T * gain_coef).T

    return gained_data


class Mute:
    """ Creates a mute to apply to CMP gathers

    Parameters
    ----------
    config : dict
        Config dictionary as specified in generate_config.py
    mute_type : str
        Whether to use a spline interpolation mute or a hyperbola mute. Options: 'hyperbola', 'spline' (default)
    """
    def __init__(self, config, mute_type='spline'):

        self.taper_length = config['mute_gain_parameters']['taper_length']
        self.delta_t = config['parameters']['delta_t']
        self.offset = config['parameters']['min_offset']
        self.delta_offset = config['parameters']['delta_offset']
        self.num_receivers = config['parameters']['num_receivers']
        self.first_arrival = config['mute_gain_parameters']['first_arrival']

        self.taper = np.arange(0, self.taper_length + self.delta_t, self.delta_t)
        self.taper = np.cos(self.taper / self.taper_length * np.pi / 2) ** 2
        self.taper = self.taper[::-1]

        self.water_velocity = 1500

        if mute_type == 'spline':
            mute_t = np.loadtxt(config['mute_gain_parameters']['mute_t_file'])
            mute_x = np.loadtxt(config['mute_gain_parameters']['mute_x_file'])
            x = np.arange(self.offset, self.delta_offset * self.num_receivers + self.offset, self.delta_offset)
            spline = UnivariateSpline(mute_x, mute_t)
            self.mute_line = spline(x)
        elif mute_type == 'hyperbola':
            t_0 = np.sqrt(self.first_arrival ** 2 - self.offset ** 2 / self.water_velocity ** 2)
            self.mute_line = np.zeros(self.num_receivers)

            for i in range(self.num_receivers):
                self.mute_line[i] = np.sqrt(t_0**2 + (self.offset + i * self.delta_offset)**2 / self.water_velocity**2)

        else:
            raise AttributeError

    def __call__(self, data, taper=True):
        """ Mutes a CMP gather with an optional cosine taper

        Parameters
        ----------
        data : Numpy array
            CMP gather to mute
        taper : bool
            Whether or not to taper the mute (default - True)
        """

        mute_indices = np.ceil(self.mute_line / self.delta_t).astype('int_')

        if taper:
            taper_len = len(self.taper)
        else:
            taper_len = 0

        for i in range(self.num_receivers):
            it_mute = mute_indices[i]
            taper_start = it_mute - taper_len

            if taper_start < 0:
                taper_start = 0

            if it_mute < taper_start:
                it_mute = taper_start

            index_taper_start = taper_len - (it_mute - taper_start)

            data[:taper_start, i] = 0

            if taper:
                data[taper_start:it_mute, i] *= self.taper[index_taper_start:].T
