# --------------------------------------------
# SALVARE SI INCARCARE KERNEL (NUCLEU) TEXT
# --------------------------------------------

import numpy as np

# Salvare kernel in fisier text
def save_kernel(path, kernel):
    np.savetxt(path, kernel, fmt="%.6f")


# Incarcare kernel din fisier text
def load_kernel(path):
    return np.loadtxt(path)