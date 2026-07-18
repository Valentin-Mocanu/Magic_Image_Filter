# --------------------------------
# GENERATE GAUSSIAN KERNEL
# --------------------------------

import numpy as np

def generate_gaussian_kernel(size, sigma):

    if size % 2 == 0:
        raise ValueError("The size must be odd!")

    # Create the matrix
    kernel = np.zeros((size, size))
    # Find the center of the matrix
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            # Apply the Gaussian formula
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel
    kernel /= np.sum(kernel)

    return kernel