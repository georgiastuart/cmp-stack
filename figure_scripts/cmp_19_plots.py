import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
from cmp_stack import gain, Mute, NormalMoveOut, RadonTransform, interpolate_missing_data


def plot_gathers(data_tensor, num_ts, num_cmp_gathers, num_recs, title):
    flat = np.zeros((num_ts, num_cmp_gathers * num_recs))
    for i in range(num_cmp_gathers):
        flat[:, i * num_recs:(i + 1) * num_recs] = data_tensor[:, :, i]

    figure, axis= plt.subplots(1, figsize=(15, 8))
    data_max = np.max(flat)
    data_min = np.min(flat)
    clip = 0.09
    axis.pcolormesh(flat, vmin=data_min * clip, vmax=data_max * clip, cmap='gray')
    axis.set_ylim(0, num_ts)
    axis.invert_yaxis()
    axis.set_yticklabels(np.round(axis.get_yticks() * config['parameters']['delta_t'], 1))
    axis.set_xticklabels(['' for i in axis.get_xticks()])
    axis.set_xlabel('Gathers')
    axis.set_ylabel('Time (s)')
    axis.set_title(title)

    return figure, axis


if __name__ == '__main__':
    with open('input/config.json', 'r') as fp:
        config = json.load(fp)

    num_time_steps = config['parameters']['num_time_steps']
    num_receivers = config['parameters']['num_receivers']
    delta_t = config['parameters']['delta_t']
    num_gathers = 9

    with h5py.File('input/cmp19_original.hdf5', 'r') as fp:
        full_data = fp['cmp_gathers'][:, :, :]

    indices = []
    data = np.zeros((num_time_steps, num_receivers, num_gathers))
    for i in range(num_gathers):
        data[:, :, i] = full_data[:, :, 2 * i]
        indices.append(2 * i * 100)

    fig, ax = plot_gathers(data, num_time_steps, num_gathers, num_receivers, 'Nine CMP Gathers from the Original Data')
    fig.savefig('figures/cmp9_original.png')

    for i in range(num_gathers):
        data[:, :, i] = gain(data[:, :, i], config)

    fig, ax = plot_gathers(data, num_time_steps, num_gathers, num_receivers, 'CMP Gathers after t^2 Gain')
    fig.savefig('figures/cmp9_gained.png')

    for i in range(num_gathers):
        interpolate_missing_data(data[:, :, i])

    fig, ax = plot_gathers(data, num_time_steps, num_gathers, num_receivers, 'CMP Gathers after t^2 Gain')
    fig.savefig('figures/cmp9_interpolated.png')

    mute = Mute(config, mute_type='hyperbola')

    for i in range(num_gathers):
        mute(data[:, :, i])

    fig, ax = plot_gathers(data, num_time_steps, num_gathers, num_receivers, 'CMP Gathers after Hyperbolic Mute')
    ax.plot(mute.mute_line / delta_t, color='red')
    fig.savefig('figures/cmp9_muted.png')

    velocity = np.fromfile(config['nmo_parameters']['vnmo_file'], dtype='float32').astype('float64')
    velocity = velocity.reshape((config['parameters']['num_gathers'], config['parameters']['num_time_steps'])).T

    nmo = NormalMoveOut(config)

    for i in range(num_gathers):
        nmo(data[:, :, i], velocity[:, indices[i]])
        data[:, :, i] = nmo.data_nmo

    fig, ax = plot_gathers(data, num_time_steps, num_gathers, num_receivers, 'CMP Gathers after Normal Moveout')
    fig.savefig('figures/cmp9_nmo.png')

    mute = Mute(config)

    for i in range(num_gathers):
        mute(data[:, :, i])

    fig, ax = plot_gathers(data, num_time_steps, num_gathers, num_receivers, 'CMP Gathers after Spline Muting')
    ax.plot(mute.mute_line / delta_t, color='red')
    fig.savefig('figures/cmp9_second_mute.png')

    radon = RadonTransform(config, mode='multiples')

    for i in range(num_gathers):
        radon(data[:, :, i])
        data[:, :, i] -= radon.inverted_data

    fig, ax = plot_gathers(data, num_time_steps, num_gathers, num_receivers, 'CMP Gathers after Multiples Suppressed' +
                           ' via Radon Transform')
    fig.savefig('figures/cmp9_multiples_suppressed.png')

    for i in range(num_gathers):
        mute(data[:, :, i], taper=False)

    fig, ax = plot_gathers(data, num_time_steps, num_gathers, num_receivers, 'CMP Gathers after Spline Mute with No Taper')
    ax.plot(mute.mute_line / delta_t, color='red')
    fig.savefig('figures/cmp9_third_mute.png')
