import json


def generate_config():
    """ Generates a new config file from defaults """
    parameters = {'num_time_steps': 1500,
                  'delta_t': 0.004,
                  'num_receivers': 60,
                  'delta_offset': 50,
                  'min_offset': 262,
                  'seafloor': 50,
                  'num_gathers': 1900,
                  'cmp_offset': 1618.5,
                  'cmp_start_num': 125,
                  'cmp_bin_size': 12.5}

    nmo_parameters = {'vnmo_file': 'input/velocity.bin'}

    radon_parameters = {'p_min': -6.0e-7,
                        'p_max': 6.0e-7,
                        'p_cutoff': 3.0e-8,
                        'delta_p': 5.0e-9}

    mute_gain_parameters = {'taper_length': 0.25,
                            'first_arrival': 0.52,
                            'gain_exponent': 2,
                            'mute_t_file': 'input/mute_t.txt',
                            'mute_x_file': 'input/mute_x.txt'}

    return {'parameters': parameters, 'nmo_parameters': nmo_parameters, 'radon_parameters': radon_parameters,
            'mute_gain_parameters': mute_gain_parameters}


if __name__ == '__main__':
    with open('input/config.json', 'w') as fp:
        json.dump(generate_config(), fp, indent=4)
