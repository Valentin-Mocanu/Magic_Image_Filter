# --------------------------------
# GENERAREA NUCLEULUI GAUSSIAN
# --------------------------------

import numpy as np # lucrul cu matrice

def generate_gaussian_kernel(size, sigma):

    if size % 2 == 0:
        raise ValueError("Dimensiunea trebuie sa fie impara!")

    # Cream matricea
    kernel = np.zeros((size, size))
    # Gasim centrul
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            # Formula Gaussian
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalizare
    kernel /= np.sum(kernel)

    return kernel