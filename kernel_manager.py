# --------------------------------------------
# SAVE AND LOAD KERNEL FILE (.txt)
# --------------------------------------------

import numpy as np

def save_kernel(path, kernel):
    np.savetxt(path, kernel, fmt="%.6f")


def load_kernel(path):
    return np.loadtxt(path)