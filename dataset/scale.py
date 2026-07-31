import numpy as np
from PIL import Image

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import index_domain
from dataset.fov import derive_fov
from dataset.fov import fov_diameter
from dataset.fov import load_fov_mask
from dataset.harmonization import working_channel

__all__ = ["measure_fov_diameter", "compute_domain_scale"]


def measure_fov_diameter(sample):
    if sample.fov_path:
        mask = load_fov_mask(sample.fov_path)

    else:
        green = working_channel(np.asarray(Image.open(sample.image_path).convert("RGB")))
        mask = derive_fov(green)

    return fov_diameter(mask)


def compute_domain_scale(name, target_diameter=1024.0, n_sample=5):
    samples = index_domain(name, require_label=False)
    step = max(1, len(samples) // n_sample)
    chosen = samples[::step][:n_sample]
    diameters = [measure_fov_diameter(s) for s in chosen]
    diameters = [d for d in diameters if d > 0]

    if not diameters:
        raise RuntimeError(f"Could not measure a FOV diameter for domain {name!r}.")

    return target_diameter / float(np.median(diameters))
