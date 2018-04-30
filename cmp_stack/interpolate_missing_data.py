import numpy as np
from scipy.interpolate import interp2d


# Didn't work well!
def interpolate_missing_data(cmp_gather):
    num_time_steps = cmp_gather.shape[0]
    num_receivers = cmp_gather.shape[1]

    times = np.array(range(num_time_steps))
    offsets = np.array(range(num_receivers))
    zero_receiver_indices = []

    for i in range(num_receivers):
        if not len(cmp_gather[cmp_gather[:, i] != 0, i]):
            zero_receiver_indices.append(i)

    print(zero_receiver_indices)

    if len(zero_receiver_indices):
        print(np.delete(offsets, zero_receiver_indices))
        interp_func = interp2d(np.delete(offsets, zero_receiver_indices), times,
                               np.delete(cmp_gather, zero_receiver_indices, axis=1), kind='cubic')
        cmp_gather[:, zero_receiver_indices] = interp_func(offsets[zero_receiver_indices], times)


