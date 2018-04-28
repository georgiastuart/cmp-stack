import numpy as np


class NormalMoveOut:

    def __init__(self, config):
        self.num_time_steps = config['parameters']['num_time_steps']
        self.num_receivers = config['parameters']['num_receivers']
        self.delta_t = config['parameters']['delta_t']
        self.delta_offset = config['parameters']['delta_offset']
        self.min_offset = config['parameters']['min_offset']
        self.seafloor = config['parameters']['seafloor']

        self.t_values = np.linspace(0, self.delta_t * self.num_time_steps, self.num_time_steps)

        vnmo = np.fromfile(config['nmo_parameters']['vnmo_file'], dtype='float32').astype('float64')
        vnmo_tau = np.fromfile(config['nmo_parameters']['tau_file'], dtype='float32').astype('float64')
        self.vnmo_interp = np.interp(self.t_values, vnmo_tau, vnmo)

        self.data_nmo = np.zeros((self.num_time_steps, self.num_receivers))

    def __call__(self, data):

        tmax = self.delta_t * (self.num_time_steps - 1)
        self.data_nmo[:self.seafloor, :] = data[:self.seafloor, :]

        for it in range(self.seafloor, self.num_time_steps):
            t0 = self.t_values[it]
            velocity = self.vnmo_interp[it]

            for ir in range(self.num_receivers):
                offset = self.min_offset + self.delta_offset * ir
                t_nmo = np.sqrt(t0**2 + (offset**2) / (velocity**2))

                if t_nmo > tmax:
                    amp = 0
                else:
                    times_at_rec = data[:, ir]
                    amp = np.interp(t_nmo, self.t_values, times_at_rec)
                self.data_nmo[it, ir] = amp

    @property
    def data_nmo_flat(self):
        return self.data_nmo.flatten()