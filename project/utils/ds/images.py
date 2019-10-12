import numpy as np
from PIL import Image


def is_grayscale(image: Image.Image) -> bool:
    rgba = image.convert("RGBA")
    return np.all(np.asarray(rgba) == np.asarray(rgba.convert("LA").convert("RGBA")))

