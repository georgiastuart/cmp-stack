import json


def generate_config():
    parameters = {'num_time_steps': 1500,
                  'delta_t': 0.004,
                  'num_receivers': 60,
                  'delta_offset': 50,
                  'min_offset': 262,
                  'seafloor': 50,
                  'num_gathers': 1900}

    nmo_parameters = {'vnmo_file': 'input/Vnmo.bin',
                      'tau_file': 'input/tau.bin'}

    radon_parameters = {'p_min': -6.0e-7,
                        'p_max': 6.0e-7,
                        'p_cutoff': 2.0e-8,
                        'delta_p': 5.0e-9}

    return {'parameters': parameters, 'nmo_parameters': nmo_parameters, 'radon_parameters': radon_parameters}


if __name__ == '__main__':
    with open('input/config.json', 'w') as fp:
        json.dump(generate_config(), fp, indent=4)
