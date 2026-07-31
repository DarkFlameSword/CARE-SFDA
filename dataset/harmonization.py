import numpy as np
from skimage.transform import rescale as _sk_rescale

from util.intensity_harmonization import invert_intensity

__all__ = ["working_channel", "harmonize", "rescale_image", "rescale_mask"]


def working_channel(image, mode="green"):
    arr = np.asarray(image)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and arr.shape[2] >= 3:

        if mode == "green":
            return arr[..., 1]

        if mode == "luminance":
            luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

            return np.rint(luma).astype(arr.dtype)

        raise ValueError(f"Unknown working-channel mode {mode!r}; expected 'green' or 'luminance'.")

    raise ValueError(f"Unsupported image shape for working channel: {arr.shape}")


def _zscore_fov(green, fov, eps=1e-6):
    fov_bool = fov.astype(bool)

    if not fov_bool.any():
        return np.zeros_like(green, dtype=np.float32)

    vals = green[fov_bool]
    mean = float(vals.mean())
    std = float(vals.std())

    if std < eps:
        return np.zeros_like(green, dtype=np.float32)

    return ((green - mean) / std).astype(np.float32)


def harmonize(image, fov, invert, norm="zscore", photometric=None, eps=1e-6, channel="green"):
    green = working_channel(image, mode=channel)

    if invert:
        green = invert_intensity(green)

    if photometric is not None:
        green = photometric(green, fov)

    fov_f = fov.astype(np.float32)

    if norm == "zscore":
        work = _zscore_fov(green.astype(np.float32), fov, eps)

    elif norm == "unit":
        work = green.astype(np.float32) / 255.0

    elif norm == "raw":
        work = green.astype(np.float32)

    else:
        raise ValueError(f"Unknown norm {norm!r}; expected 'zscore', 'unit', or 'raw'.")

    return work * fov_f


def rescale_image(arr, scale):
    if abs(scale - 1.0) < 1e-6:
        return arr.astype(np.float32, copy=False)

    out = _sk_rescale(arr, scale, order=1, anti_aliasing=(scale < 1.0), preserve_range=True)

    return out.astype(np.float32)


def rescale_mask(mask, scale):
    if abs(scale - 1.0) < 1e-6:
        return np.asarray(mask, dtype=bool)

    out = _sk_rescale(np.asarray(mask, dtype=np.float32), scale, order=0,
                      anti_aliasing=False, preserve_range=True)

    return out > 0.5
