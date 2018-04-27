import json
import numpy as np
from cmp_stack import RadonTransform
import matplotlib.pyplot as plt




if __name__ == '__main__':
    with open('input/config.json', 'r') as fp:
        config = json.load(fp)

    data = np.fromfile('input/CMP_nmo_mute.bin', dtype='float32')
    data = data.astype('float64')
    data = data.reshape(config['parameters']['num_receivers'], config['parameters']['num_time_steps']).T
    flat_data = data.flatten()

    plt.pcolormesh(data)
    plt.savefig('figures/data.png')
    plt.clf()

    rt = RadonTransform(config)
    rt(flat_data)

    plt.pcolormesh(rt.radon_domain_out, vmin=-3500, vmax=3500, cmap='gray')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('figures/radon_domain.png')

