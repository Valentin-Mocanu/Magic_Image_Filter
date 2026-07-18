# -------------------------------
# SAVE AND LOAD IMAGE
# -------------------------------

from PIL import Image # convert NumPy array to image
import numpy as np

def load_image(path):
    img = Image.open(path)

    # Check if the image is in RGB format
    if img.mode == "RGB":
        bit_depth = 24
    else:
        # Convert the image to grayscale
        img = img.convert("L")
        bit_depth = 8

    return np.array(img), bit_depth


def save_image(path, array, bit_depth):

    if bit_depth == 24:
        img = Image.fromarray(array.astype(np.uint8), "RGB")
    else:
        img = Image.fromarray(array.astype(np.uint8), "L")

    img.save(path)