import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = (15, 8)

stack = np.loadtxt('stack.txt')

plt.pcolormesh(stack, vmin=-3000, vmax=3000, cmap='gray')
plt.gca().invert_yaxis()
plt.savefig('figures/stack_1900_2.png', dpi=300)
