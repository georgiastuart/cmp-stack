import matplotlib
matplotlib.use("Agg")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import h5py
import json
from cmp_stack import NormalMoveOut, mute
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master = rank == 0
world_size = comm.Get_size()

if __name__ == '__main__':
    gather_num = 1599
    config = None

    if master:
        with open('input/config.json', 'r') as fp:
            config = json.load(fp)

    config = comm.bcast(config, root=0)

    min_offset = config['parameters']['min_offset']
    delta_offset = config['parameters']['delta_offset']
    delta_t = config['parameters']['delta_t']
    num_time_steps = config['parameters']['num_time_steps']
    all_num_gathers = config['parameters']['num_gathers']

    muted_data = None
    gathers_per_rank = None

    if master:
        with h5py.File('input/cmp1900_mute_gain.hdf5', 'r') as data_file:
            gathers_per_rank = [all_num_gathers // world_size + int(i < all_num_gathers % world_size) for i in range(world_size)]
            print(gathers_per_rank)
            print(sum(gathers_per_rank))

            muted_data = data_file['cmp_gathers'][:, :, :gathers_per_rank[0]]

            for i in range(1, world_size):
                temp_data = data_file['cmp_gathers'][:, :, sum(gathers_per_rank[:i]):sum(gathers_per_rank[:i+1])]
                comm.send(temp_data, dest=i, tag=0)
                print('Sent to rank {}'.format(i), flush=True)

    if not master:
        muted_data = comm.recv(source=0, tag=0)
        print('Rank {} Received Data'.format(rank), flush=True)

    num_gathers = muted_data.shape[2]
    stack_section = np.zeros((muted_data.shape[0], num_gathers))

    nmo = NormalMoveOut(config)

    if master:
        print('Starting NMO', flush=True)

    for i in range(num_gathers):
        if master:
            print('Gather: {}'.format(i), flush=True)
        cmp_gather = muted_data[:, :, i]
        nmo(cmp_gather)
        mute(nmo.data_nmo, config)

        num_nonzero = (nmo.data_nmo != 0).sum(axis=1)
        num_nonzero[num_nonzero == 0] = 1

        stack_section[:, i] = np.sum(nmo.data_nmo, axis=1) / num_nonzero

    if not master:
        comm.send(stack_section, dest=0, tag=1)

    if master:
        full_stack = np.zeros((num_time_steps, all_num_gathers))
        full_stack[:, :gathers_per_rank[0]] = stack_section
        for i in range(1, world_size):
            temp_stack = comm.recv(source=i, tag=1)
            full_stack[:, sum(gathers_per_rank[:i]):sum(gathers_per_rank[:(i + 1)])] = temp_stack

        full_stack.tofile('intermediate_data/stack_after_nmo_mute_1900.bin')

        fig, ax = plt.subplots(1, figsize=(15, 8))
        data_max = np.max(full_stack)
        data_min = np.min(full_stack)
        clip = 0.09
        ax.pcolormesh(full_stack, vmin=data_min*clip, vmax=data_max*clip,  cmap='gray')

        ax.invert_yaxis()

        fig.savefig('figures/stack_after_nmo_mute_1900.png', dpi=300)
