import numpy as np

__all__ = ["invert_intensity"]


def invert_intensity(image, max_value=None):
    image = np.asarray(image)

    if max_value is None:

        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max

        else:
            max_value = 1.0

    inverted = max_value - image

    return inverted.astype(image.dtype, copy=False)
