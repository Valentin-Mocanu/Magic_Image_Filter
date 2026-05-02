# -----------------------------------------------
# CONVOLUTIA SI PADDING-UL (MOTORUL FILTRARII)
# -----------------------------------------------

import numpy as np # lucrul cu matrice

# Functie care adauga margini artificiale in jurul imaginii
def pad_image(image, pad, mode):

    if mode == "zero":
        # Adaugam pixeli 0 la margini
        return np.pad(image, pad, mode='constant')

    elif mode == "mirror":
        # Reflectam imaginea la margini (mai natural pentru procesarea imaginii)
        return np.pad(image, pad, mode='reflect')

    elif mode == "replicate":
        # Repetam pixelii de la margine
        return np.pad(image, pad, mode='edge')

    else:
        raise ValueError("Tip padding invalid!")


# Functie care aplica filtrul Gaussian prin convolutie
def convolve2d(image, kernel, padding_type):

    # Luam dimensiunea kernel-ului, care este patrat (de exemplu: 5x5)
    k = kernel.shape[0]
    # Calculam padding-ul
    pad = k // 2

    # Aplicam padding-ul si obtinem o imagine mai mare
    padded = pad_image(image, pad, padding_type)

    # Cream matricea rezultat, care are acceasi dimensiune cu originalul
    output = np.zeros_like(image, dtype=float)

    # Parcurgem fiecare pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extragem regiunea locala
            region = padded[i:i+k, j:j+k]
            # Aplicam formula convolutiei
            output[i, j] = np.sum(region * kernel)

    return output