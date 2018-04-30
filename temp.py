import numpy as np
import matplotlib.pyplot as plt

# t = np.array([0, 0.224,  0.9, 1.224, 1.736, 2.055, 2.259, 2.5])
# x = np.array([262, 862, 1062,  1512, 1962, 2312, 2662, 3162])
#
# np.savetxt('input/mute_t.txt', t)
# np.savetxt('input/mute_x.txt', x)
#
# tau = np.array([0.516, 0.948, 1.824, 2.075, 2.512, 2.936, 3.124, 3.660, 4.348, 5.276, 5.760])
# print(tau)
# np.savetxt('input/tau.txt', tau)
# vnmo = np.array([[1482, 1683, 2025, 2105, 2306, 2527, 2608, 2788, 2929, 3110, 3251]])
# np.savetxt('input/vnmo.txt', vnmo)

velocity = np.fromfile('input/velocity.bin', dtype='float32')
print(velocity.reshape((1900,1500)))