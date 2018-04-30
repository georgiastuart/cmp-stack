import numpy as np
from cmp_stack.utilities import wrap_function, master
from ctypes import c_int, c_double, POINTER, Structure


class NormalMoveOut(Structure):
    """ Serves as both an object to run normal move out
    and a nmo_params_t struct as specified in cmp_c_library/library.h
    """
    _fields_ = [('num_time_steps', c_int), ('num_receivers', c_int), ('delta_t', c_double), ('delta_offset', c_double),
                ('min_offset', c_double), ('seafloor', c_int), ('vnmo_interp', POINTER(c_double))]

    def __init__(self, config):
        """ Initializes a NormalMoveOut object

        Parameters
        ----------
        config : dict
            Config dictionary as specified in generate_config.py
        """

        super().__init__()
        self.num_time_steps = config['parameters']['num_time_steps']
        self.num_receivers = config['parameters']['num_receivers']
        self.delta_t = config['parameters']['delta_t']
        self.delta_offset = config['parameters']['delta_offset']
        self.min_offset = config['parameters']['min_offset']
        self.seafloor = config['parameters']['seafloor']

        self.t_values = np.linspace(0, self.delta_t * self.num_time_steps, self.num_time_steps)

        # Interpolates vnmo from a selection of vnmo-tau pairs from semblance analysis
        vnmo = np.fromfile(config['nmo_parameters']['vnmo_file'], dtype='float32').astype('float64')
        vnmo_tau = np.fromfile(config['nmo_parameters']['tau_file'], dtype='float32').astype('float64')
        self._np_vnmo_interp = np.interp(self.t_values, vnmo_tau, vnmo)
        self.vnmo_interp = self._np_vnmo_interp.ctypes.data_as(POINTER(c_double))

        self._np_data_nmo = np.zeros(self.num_time_steps * self.num_receivers)
        self._data_nmo = self._np_data_nmo.ctypes.data_as(POINTER(c_double))

        # Wraps void normal_move_out(nmo_parameters_t *params, const double *data, double *nmo_data)
        # from cmp_c_library/library.h
        self._c_nmo = wrap_function('normal_move_out', None, [POINTER(NormalMoveOut), POINTER(c_double),
                                                              POINTER(c_double)])

    def __call__(self, data):
        """
        Applies normal move out to the input data

        Parameters
        ----------
        data : np.ndarray
            CMP gather to apply normal move out to
        """
        data = data.flatten()
        _data = data.ctypes.data_as(POINTER(c_double))
        self._c_nmo(self, _data, self._data_nmo)

    @property
    def data_nmo(self):
        """ Reshapes the flat NMO data """
        return self._np_data_nmo.reshape((self.num_time_steps, self.num_receivers))