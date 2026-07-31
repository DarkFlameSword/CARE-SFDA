import numpy as np
from skimage.exposure import adjust_gamma
from skimage.exposure import equalize_adapthist

__all__ = ["photometric_augment"]


def _apply_gamma(work, rng, strength=1.0):
    gamma = float(rng.uniform(max(0.2, 1.0 - 0.3 * strength), 1.0 + 0.3 * strength))

    return adjust_gamma(work, gamma=gamma).astype(np.float32)


def _apply_clahe(work, rng, strength=1.0):
    clip_limit = float(rng.uniform(0.01, 0.03 * strength))

    return equalize_adapthist(work, clip_limit=clip_limit).astype(np.float32)


def _add_spots(work, fov, rng, max_spots=3, amp_scale=1.0):
    ys, xs = np.nonzero(fov)

    if ys.size == 0:
        return work

    h, w = work.shape
    n = int(rng.integers(0, max_spots + 1))
    out = work

    for _ in range(n):
        pick = int(rng.integers(0, ys.size))
        cy = int(ys[pick])
        cx = int(xs[pick])
        radius = float(rng.uniform(0.01, 0.05)) * max(h, w)
        amp = float(rng.uniform(0.1, 0.4)) * amp_scale
        yy, xx = np.ogrid[:h, :w]
        blob = amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * radius ** 2)))
        out = out + blob.astype(np.float32)

    return np.clip(out, 0.0, 1.0)


def photometric_augment(green, fov, rng, strength=1.0):
    prob = min(1.0, 0.5 * strength)
    work = green.astype(np.float32) / 255.0
    fov_bool = fov.astype(bool)

    if rng.random() < prob:
        work = _apply_gamma(work, rng, strength)

    if rng.random() < prob:
        work = _apply_clahe(work, rng, strength)

    if rng.random() < prob:
        work = _add_spots(work, fov_bool, rng, max_spots=int(round(3 * strength)), amp_scale=strength)

    work = np.clip(work, 0.0, 1.0)

    return (work * 255.0).astype(np.uint8)
