import json
import numpy as np
from cmp_stack import RadonTransform, NormalMoveOut
import matplotlib.pyplot as plt
import h5py




if __name__ == '__main__':
    with open('input/config.json', 'r') as fp:
        config = json.load(fp)

    num_gathers = config['parameters']['num_gathers']

    nmo = NormalMoveOut(config)

    with h5py.File('input/cmp1900_mute_gain.hdf5', 'r') as data_file:

        muted_data = data_file['cmp_gathers'][:, :, 0]
        # plt.pcolormesh(data_file['cmp_gathers'][:, :, 1899], vmin=-100, vmax=100, cmap='gray')
        # plt.gca().invert_yaxis()
        # plt.colorbar()
        # plt.savefig('figures/one_cmp_1900.png')
        # plt.clf()

    nmo(muted_data)

    plt.pcolormesh(nmo.data_nmo, vmin=-100, vmax=100, cmap='gray')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig('figures/one_cmp_nmo.png')
    plt.clf()

    # cmp1_data = np.fromfile('input/1cmp_gather.bin', dtype='float32').astype('float64').reshape(nmo.num_receivers,
    #                                                                                             nmo.num_time_steps).T
    #
    # plt.pcolormesh(cmp1_data, cmap='gray')
    # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.savefig('figures/cmp1.png')
    # plt.clf()


    # data = np.fromfile('input/CMP_nmo_mute.bin', dtype='float32')
    # data = data.astype('float64')
    # data = data.reshape(config['parameters']['num_receivers'], config['parameters']['num_time_steps']).T
    # flat_data = data.flatten()
    #
    # plt.pcolormesh(data, cmap='gray')
    # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.savefig('figures/data.png')
    # plt.clf()
    #
    rt = RadonTransform(config)
    rt(nmo.data_nmo_flat)

    plt.pcolormesh(rt.radon_domain_out, vmin=-3500, vmax=3500, cmap='gray')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('figures/radon_domain.png')
    plt.clf()

    plt.pcolormesh(rt.inverted_data, cmap='gray')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('figures/inverted_data.png')


