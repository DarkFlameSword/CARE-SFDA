import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import binary_closing
from skimage.morphology import binary_erosion
from skimage.morphology import disk

__all__ = ["derive_fov", "load_fov_mask", "fov_diameter"]


def _largest_component(foreground):
    labeled = label(foreground, connectivity=2)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    return labeled == counts.argmax()


def _disk_mask(shape, center, radius):
    h, w = shape
    rr, cc = np.ogrid[:h, :w]
    dist_sq = (rr - center[0]) ** 2 + (cc - center[1]) ** 2

    return dist_sq <= radius ** 2


def derive_fov(gray, low_thresh=16, close_radius=5, erode_px=5,
               full_frame_coverage=0.98, circle_fill_ratio=0.70):
    gray = np.asarray(gray)

    if gray.ndim != 2:
        raise ValueError("derive_fov expects a 2D working-channel image.")

    if gray.dtype != np.uint8:
        raise ValueError(f"derive_fov expects a uint8 [0,255] working channel; got {gray.dtype}.")

    h, w = gray.shape
    foreground = gray > low_thresh

    if foreground.mean() >= full_frame_coverage:
        return np.ones((h, w), dtype=bool)

    if not foreground.any():
        return np.ones((h, w), dtype=bool)

    largest = _largest_component(foreground)
    filled = binary_fill_holes(largest)
    closed = binary_closing(filled, disk(close_radius))
    rows = np.where(np.any(closed, axis=1))[0]
    cols = np.where(np.any(closed, axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1
    fill_ratio = closed.sum() / float(bbox_h * bbox_w)
    near_full = (bbox_h * bbox_w) >= full_frame_coverage * h * w

    if near_full and fill_ratio > 0.95:
        return np.ones((h, w), dtype=bool)

    if fill_ratio >= circle_fill_ratio and not near_full:
        center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        radius = max(bbox_h, bbox_w) / 2.0
        mask = _disk_mask((h, w), center, radius)

    else:
        mask = closed

    if erode_px > 0:
        mask = binary_erosion(mask, disk(erode_px))

    return mask


def load_fov_mask(fov_path, target_shape=None):
    from PIL import Image

    img = Image.open(fov_path).convert("L")

    if target_shape is not None and img.size != (target_shape[1], target_shape[0]):
        img = img.resize((target_shape[1], target_shape[0]), Image.NEAREST)

    arr = np.asarray(img)

    return arr > (arr.max() / 2.0 if arr.max() > 0 else 0)


def fov_diameter(mask):
    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        return 0.0

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    bbox_h = rows[-1] - rows[0] + 1
    bbox_w = cols[-1] - cols[0] + 1

    return float(max(bbox_h, bbox_w))
