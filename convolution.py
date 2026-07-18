# -----------------------------------------------
# CONVOLUTION AND PADDING (THE FILTERING ENGINE)
# -----------------------------------------------

import numpy as np

# Adds padding around the image
def pad_image(image, pad, mode):

    if mode == "zero":
        # Add zero-valued pixels around the image
        return np.pad(image, pad, mode='constant')

    elif mode == "mirror":
        # Mirror the image at the borders
        return np.pad(image, pad, mode='reflect')

    elif mode == "replicate":
        # Replicate the border pixels
        return np.pad(image, pad, mode='edge')

    else:
        raise ValueError("Invalid padding!")


# Applies the selected kernel using convolution
def convolve2d(image, kernel, padding_type):

    # Get the kernel size (for example: 5x5)
    k = kernel.shape[0]
    # Calculate the required padding
    pad = k // 2

    # Apply padding to the image
    padded = pad_image(image, pad, padding_type)

    # Create the output image with the same dimensions as the original
    output = np.zeros_like(image, dtype=float)

    # Process each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the local neighborhood
            region = padded[i:i+k, j:j+k]
            # Perform the convolution
            output[i, j] = np.sum(region * kernel)

    return output