import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data = np.fromfile('intermediate_data/stack_after_nmo_mute_1900.bin').reshape((1500, 1900))
    data_min = np.min(data)
    data_max = np.max(data)
    clip = 0.7

    fig, ax = plt.subplots(1, figsize=(15, 8))

    ax.pcolormesh(data, vmin=clip*data_min, vmax=clip*data_max, cmap='gray')
    ax.invert_yaxis()

    fig.savefig('figures/stack_after_nmo_mute_1900.png', dpi=300)