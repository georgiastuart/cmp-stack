import matplotlib.pyplot as plt
import numpy as np
import h5py
import json
from cmp_stack import NormalMoveOut, Mute, RadonTransform

if __name__ == '__main__':
    gather_num = 1599

    with open('input/config.json', 'r') as fp:
        config = json.load(fp)

    with h5py.File('input/cmp1900_mute_gain.hdf5', 'r') as fp:
        cmp_gather = fp['cmp_gathers'][:, :, gather_num]

    velocity = np.fromfile(config['nmo_parameters']['vnmo_file'], dtype='float32').astype('float64')
    velocity = velocity.reshape((config['parameters']['num_gathers'], config['parameters']['num_time_steps'])).T
    velocity = velocity[:, gather_num]

    min_offset = config['parameters']['min_offset']
    delta_offset = config['parameters']['delta_offset']
    delta_t = config['parameters']['delta_t']

    nmo = NormalMoveOut(config)
    nmo(cmp_gather, velocity)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 8))

    ax1.pcolormesh(nmo.data_nmo, vmin=-400, vmax=400, cmap='gray')
    ax1.set_xticklabels((ax1.get_xticks() * delta_offset + min_offset).astype('int_'))
    ax1.set_yticklabels(np.round(ax1.get_yticks() * delta_t, 1))
    ax1.set_ylim(0, 1499)
    ax1.set_xlabel('Offset (m)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('After NMO')
    ax1.invert_yaxis()

    muted_data = np.copy(nmo.data_nmo)
    mute = Mute(config, mute_type='spline')
    mute.__call__(muted_data)

    ax2.pcolormesh(nmo.data_nmo, vmin=-400, vmax=400, cmap='gray')
    ax2.plot(mute.mute_line / delta_t, zorder=1, color='red')
    ax2.set_title('Mute Line')
    ax2.set_xlabel('Offset (m)')

    ax3.set_title('After Second Mute')
    ax3.pcolormesh(muted_data, vmin=-400, vmax=400, cmap='gray')
    ax3.set_xlabel('Offset (m)')

    fig.savefig('figures/second_mute.png', dpi=300)

    rt = RadonTransform(config, mode='all')
    rt(muted_data)
    all_radon_domain = np.copy(rt.radon_domain_out)
    all_inverted = np.copy(rt.inverted_data)

    rt = RadonTransform(config, mode='multiples')
    rt(muted_data)
    mult_radon_domain = np.copy(rt.radon_domain_out)
    mult_inverted = np.copy(rt.inverted_data)

    rt = RadonTransform(config, mode='primaries')
    rt(muted_data)
    prim_radon_domain = np.copy(rt.radon_domain_out)
    prim_inverted = np.copy(rt.inverted_data)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 8))

    ax1.pcolormesh(all_radon_domain, vmin=-3000, vmax=3000, cmap='gray')
    ax1.set_xticklabels(np.round((ax1.get_xticks() * rt.delta_p + rt.p_min) * 1.0e6, 2))
    ax1.set_yticklabels(np.round(ax1.get_yticks() * delta_t, 1))
    ax1.set_xlabel('P (1.0e-6)')
    ax1.set_ylabel('Tau (s)')
    ax1.set_title('Full Radon Domain')
    ax1.invert_yaxis()

    ax2.pcolormesh(mult_radon_domain, vmin=-3000, vmax=3000, cmap='gray')
    ax2.set_xticklabels(np.round((ax2.get_xticks() * rt.delta_p + rt.p_cutoff) * 1.0e6, 2))
    ax2.set_title('Multiples in the Radon Domain')
    ax2.set_xlabel('P (1.0e-6)')

    ax3.pcolormesh(prim_radon_domain, vmin=-3000, vmax=3000, cmap='gray')
    ax3.set_xticklabels(np.round((ax3.get_xticks() * rt.delta_p + rt.p_min) * 1.0e6, 2))
    ax3.set_title('Primaries in the Radon Domain')
    ax3.set_xlabel('P (1.0e-6)')

    fig.savefig('figures/radon_domain.png', dpi=300)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15, 8))

    ax1.pcolormesh(muted_data, vmin=-400, vmax=400, cmap='gray')
    ax1.set_xticklabels((ax1.get_xticks() * delta_offset + min_offset).astype('int_'))
    ax1.set_yticklabels(np.round(ax1.get_yticks() * delta_t, 1))
    ax1.set_xlabel('Offset (m)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Data after NMO')
    ax1.invert_yaxis()

    ax2.pcolormesh(muted_data - mult_inverted, vmin=-400, vmax=400, cmap='gray')
    ax2.set_title('Subtracted Multiples from Data')
    ax2.set_xlabel('Offset (m)')

    ax3.pcolormesh(prim_inverted, vmin=-400, vmax=400, cmap='gray')
    ax3.set_title('Modeled Primaries')
    ax3.set_xlabel('Offset (m)')

    ax4.pcolormesh(mult_inverted, vmin=-400, vmax=400, cmap='gray')
    ax4.set_title('Modeled Multiples')
    ax4.set_xlabel('Offset (m)')

    fig.savefig('figures/radon_inverted.png', dpi=300)

    diff = muted_data - mult_inverted
    mute.__call__(diff, taper=False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 8))

    ax1.pcolormesh(muted_data - mult_inverted, vmin=-400, vmax=400, cmap='gray')
    ax1.set_xticklabels((ax1.get_xticks() * delta_offset + min_offset).astype('int_'))
    ax1.set_yticklabels(np.round(ax1.get_yticks() * delta_t, 1))
    ax1.set_ylim(0, 1499)
    ax1.set_xlabel('Offset (m)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Multiples Suppressed')
    ax1.invert_yaxis()

    ax2.pcolormesh(muted_data - mult_inverted, vmin=-400, vmax=400, cmap='gray')
    ax2.plot(mute.mute_line / delta_t, zorder=1, color='red')
    ax2.set_title('Mute Line')
    ax2.set_xlabel('Offset (m)')

    ax3.set_title('After Third Mute')
    ax3.pcolormesh(diff, vmin=-400, vmax=400, cmap='gray')
    ax3.set_xlabel('Offset (m)')


    fig.savefig('figures/third_mute.png', dpi=300)
