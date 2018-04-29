import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import numpy as np
from cmp_stack import cmp_stack
import matplotlib.pyplot as plt
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master = rank == 0
world_size = comm.Get_size()


if __name__ == '__main__':

    config = None
    muted_data = None
    gathers_per_rank = None

    if master:
        with open('input/config.json', 'r') as fp:
            config = json.load(fp)

    config = comm.bcast(config, root=0)

    num_gathers = config['parameters']['num_gathers']
    num_time_steps = config['parameters']['num_time_steps']

    if master:
        with h5py.File('input/cmp1900_mute_gain.hdf5', 'r') as data_file:
            gathers_per_rank = [num_gathers // world_size + int(i < num_gathers % world_size) for i in range(world_size)]
            print(gathers_per_rank)
            print(sum(gathers_per_rank))

            muted_data = data_file['cmp_gathers'][:, :, :gathers_per_rank[0]]

            for i in range(1, world_size):
                temp_data = data_file['cmp_gathers'][:, :, sum(gathers_per_rank[:i]):sum(gathers_per_rank[:i+1])]
                comm.send(temp_data, dest=i, tag=0)

    if not master:
        muted_data = comm.recv(source=0, tag=0)

    stack = cmp_stack(muted_data, config)

    if not master:
        comm.send(stack, dest=0, tag=1)

    if master:
        full_stack = np.zeros((num_time_steps, num_gathers))
        full_stack[:, :gathers_per_rank[0]] = stack
        for i in range(1, world_size):
            temp_stack = comm.recv(source=i, tag=1)
            full_stack[:, sum(gathers_per_rank[:i]):sum(gathers_per_rank[:(i + 1)])] = temp_stack

        np.savetxt('stack.txt', full_stack)

        plt.pcolormesh(full_stack, cmap='gray')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('figures/stack_1900.png')
        plt.clf()


