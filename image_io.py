# -------------------------------
# INCARCARE + SALVARE IMAGINE
# -------------------------------

from PIL import Image # conversie numpy in imagine
import numpy as np # lucrul cu matrice

# Functie de incarcare imagine
def load_image(path):
    # Deschidem imaginea
    img = Image.open(path)

    # Daca e color
    if img.mode == "RGB":
        bit_depth = 24
    else:
        # Convertim la grayscale
        img = img.convert("L")
        bit_depth = 8

    # Returnam matricea + bit depth
    return np.array(img), bit_depth


# Functie de salvare imagine
def save_image(path, array, bit_depth):

    if bit_depth == 24:
        img = Image.fromarray(array.astype(np.uint8), "RGB")
    else:
        img = Image.fromarray(array.astype(np.uint8), "L")

    img.save(path)