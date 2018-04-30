from cmp_stack.utilities import wrap_function, master
from ctypes import c_int, c_double, POINTER, Structure
import numpy as np
from numpy.fft import irfft, rfft
from math import ceil


class RadonTransform(Structure):
    """ Serves as both an object to run the radon transform and inverse radon transform
        and a radon_params_t struct as specified in cmp_c_library/library.h
        """
    _fields_ = [('num_time_steps', c_int), ('num_receivers', c_int), ('delta_t', c_double), ('delta_offset', c_double),
                ('min_offset', c_double), ('p_min', c_double), ('p_max', c_double), ('p_cutoff', c_double), ('delta_p', c_double),
                ('num_p', c_int)]

    def __init__(self, config, mode='all'):
        """ Initializes a RadonTransform object
        Parameters
        ----------
        config : dict
            Config dictionary as specified in generate_config.py
        """

        super().__init__()

        # Wraps void radon_transform(radon_parameters_t *params, const double *data, double *rad_domain_out, int mode)
        # from cmp_c_library/library.h
        self._c_radon_transform = wrap_function('radon_transform', None, [POINTER(RadonTransform), POINTER(c_double),
                                                                          POINTER(c_double), c_int])
        self.mode = mode

        self.num_time_steps = config['parameters']['num_time_steps']
        self.num_receivers = config['parameters']['num_receivers']
        self.delta_t = config['parameters']['delta_t']
        self.delta_offset = config['parameters']['delta_offset']
        self.min_offset = config['parameters']['min_offset']

        self.p_min = config['radon_parameters']['p_min']
        self.p_max = config['radon_parameters']['p_max']
        self.p_cutoff = config['radon_parameters']['p_cutoff']
        self.delta_p = config['radon_parameters']['delta_p']
        self.num_p = ceil((self.p_max - self.p_min) / self.delta_p) + 1

        # Sets p_values based on the mode of the Radon Transform
        if mode == 'all':
            self.p_values = np.arange(self.p_min, self.p_max + self.delta_p, self.delta_p)
            self.num_p = self.num_p
        elif mode == 'primaries':
            self.p_values = np.arange(self.p_min, self.p_cutoff + self.delta_p, self.delta_p)
            self.num_p = len(self.p_values)
        elif mode == 'multiples':
            self.p_values = np.arange(self.p_cutoff, self.p_max + self.delta_p, self.delta_p)
            self.num_p = len(self.p_values)
            print(self.num_p)
        else:
            raise ValueError

        self._np_radon_domain_out = np.zeros(self.num_p * self.num_time_steps)
        self._radon_domain_out = self._np_radon_domain_out.ctypes.data_as(POINTER(c_double))

        self.offsets = np.arange(self.min_offset, self.min_offset + self.delta_offset * self.num_receivers,
                                 self.delta_offset)
        self.inverted_data = None

    def __call__(self, data):
        """
        Applies the radon transform over the p range specified in init, then inverts back to time/offset space

        Parameters
        ----------
        data : np.ndarray
            CMP gather (after NMO) to apply normal move out to
        """
        self.radon_transform(data)
        self.inverse_radon_transform(data)

    def radon_transform(self, data):
        """
        Applies the radon transform over the p range specified in init

        Parameters
        ----------
        data : np.ndarray
            CMP gather (after NMO) to apply normal move out to
        """
        data = data.flatten()
        _data = data.ctypes.data_as(POINTER(c_double))
        mode_num = {'all': 0, 'primaries': 1, 'multiples': 2}

        self._c_radon_transform(self, _data, self._radon_domain_out, mode_num[self.mode])

    def inverse_radon_transform(self, data):
        """
        Inverts self.radon_transform_out

        Parameters
        ----------
        data : np.ndarray
            The pre-Radon transform data, for computing the scale factor
        """
        num_freq = 2**(int(ceil(np.log2(self.num_time_steps)) + 1))
        delta_freq = 1 / self.delta_t

        data_fft = np.zeros((self.num_receivers, num_freq), dtype='complex_')

        shift = np.tile(self.p_values, (self.num_receivers, 1))
        shift = (shift.T * (self.offsets**2)).T

        data_radon_fft = rfft(self.radon_domain_out.T, num_freq, axis=1)

        for i in range((num_freq + 1) // 2):
            freq = (i / num_freq) * delta_freq * -1
            l_matrix = np.exp(shift * 2j * np.pi * freq)
            data_fft[:, i] = l_matrix @ data_radon_fft[:, i]

            if i != 0:
                data_fft[:, num_freq - i] = np.conj(data_fft[:, i])

        inv_data = irfft(data_fft, num_freq, axis=1)
        inv_data = inv_data[:, :self.num_time_steps].T

        data = data.reshape((self.num_time_steps, self.num_receivers))

        scale = np.sum(inv_data * inv_data) / np.sum(data * inv_data)

        self.inverted_data = inv_data / scale

    @property
    def radon_domain_out(self):
        """ Reshapes the flat Radon Transform Out array """
        return self._np_radon_domain_out.reshape((self.num_time_steps, self.num_p))
