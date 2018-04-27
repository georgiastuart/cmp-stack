from cmp_stack.utilities import wrap_function, master
from ctypes import c_int, c_double, POINTER, Structure
import numpy as np


class RadonTransform(Structure):
    _fields_ = [('num_time_steps', c_int), ('num_receivers', c_int), ('delta_t', c_double), ('delta_offset', c_double),
                ('min_offset', c_double), ('p_min', c_double), ('p_max', c_double), ('delta_p', c_double)]

    def __init__(self, config):
        super().__init__()
        self._c_radon_transform = wrap_function('radon_transform', None, [POINTER(RadonTransform), POINTER(c_double),
                                                                          POINTER(c_double)])

        self.num_time_steps = config['parameters']['num_time_steps']
        self.num_receivers = config['parameters']['num_receivers']
        self.delta_t = config['parameters']['delta_t']
        self.delta_offset = config['parameters']['delta_offset']
        self.min_offset = config['parameters']['min_offset']

        self.p_min = config['radon_parameters']['p_min']
        self.p_max = config['radon_parameters']['p_max']
        self.delta_p = config['radon_parameters']['delta_p']
        self.num_p = int((self.p_max - self.p_min) / self.delta_p)

        self._np_radon_domain_out = np.zeros(self.num_p * self.num_time_steps)
        self._radon_domain_out = self._np_radon_domain_out.ctypes.data_as(POINTER(c_double))

    def __call__(self, data):
        _data = data.ctypes.data_as(POINTER(c_double))
        self._c_radon_transform(self, _data, self._radon_domain_out)

    @property
    def radon_domain_out(self):
        return self._np_radon_domain_out.reshape((self.num_time_steps, self.num_p))
