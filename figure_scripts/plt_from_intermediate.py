import matplotlib.pyplot as plt
import numpy as np
import json

if __name__ == '__main__':
    # data = np.fromfile('intermediate_data/stack_after_nmo_mute_1900.bin').reshape((1500, 1900))
    # data_min = np.min(data)
    # data_max = np.max(data)
    # clip = 0.7
    #
    # fig, ax = plt.subplots(1, figsize=(15, 8))
    #
    # ax.pcolormesh(data, vmin=clip*data_min, vmax=clip*data_max, cmap='gray')
    # ax.invert_yaxis()
    #
    # fig.savefig('figures/stack_after_nmo_mute_1900.png', dpi=300)

    mode = 'multiples'
    fname_dict = {'multiples': 'multiples_suppressed', 'primaries': 'modeled_primaries'}
    title_dict = {'multiples': 'Multiples Suppressed', 'primaries': 'Modeled Primaries'}

    with open('input/config.json', 'r') as fp:
        config = json.load(fp)

    num_time_steps = config['parameters']['num_time_steps']
    num_gathers = config['parameters']['num_gathers']
    cmp_offset = config['parameters']['cmp_offset']
    cmp_start_num = config['parameters']['cmp_start_num']
    cmp_bin_size = config['parameters']['cmp_bin_size']

    full_stack = np.fromfile('intermediate_data/{}_final_stack.bin'.format(fname_dict[mode])).reshape((num_time_steps,
                                                                                                       num_gathers))

    fig, ax = plt.subplots(1, figsize=(15, 8))
    data_max = np.max(full_stack)
    data_min = np.min(full_stack)
    clip = 0.09
    ax.pcolormesh(full_stack, vmin=data_min * clip, vmax=data_max * clip, cmap='gray')
    ax.invert_yaxis()
    ax.set_xticklabels((ax.get_xticks() + cmp_start_num - 1) * cmp_bin_size + cmp_offset)
    ax.set_yticklabels(np.round(ax.get_yticks() * config['parameters']['delta_t'], 1))
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title('CMP Stack with {}'.format(title_dict[mode].title()))

    fig.savefig('figures/{}_stack_1900.png'.format(fname_dict[mode]), dpi=300)