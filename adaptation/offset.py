import numpy as np

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import index_domain
from dataset.harmonization import working_channel
from dataset.retinal_vessel_dataset import EvalDataset
from dataset.retinal_vessel_dataset import _load_label
from dataset.retinal_vessel_dataset import _load_rgb
from dataset.retinal_vessel_dataset import _resolve_fov
from dataset.splits import domain_split
from dataset.tiling import sliding_window_predict
from dataset.tiling import upsample_to_native
from evaluation.evaluator import make_predict_fn

__all__ = ["source_vessel_prevalence", "estimate_domain_offset"]


def source_vessel_prevalence(seed=0, val_fraction=0.15):
    train_ids = set(domain_split(SOURCE_DOMAIN, eval_fraction=val_fraction, seed=seed)[0])
    fractions = []

    for sample in index_domain(SOURCE_DOMAIN, split="train", require_label=True):

        if sample.sample_id not in train_ids:
            continue

        green = working_channel(_load_rgb(sample.image_path))
        fov = _resolve_fov(sample, green).astype(bool)
        label = _load_label(sample.label_path)
        fractions.append(float(label[fov].mean()))

    return float(np.mean(fractions))


def estimate_domain_offset(model, name, device, pi_s, tile=512, overlap=0.25, use_amp=False,
                           grid=None, n_est=40, seed=0, eval_fraction=0.30):
    if grid is None:
        grid = np.round(np.linspace(0.02, 0.98, 49), 4)

    spec = DOMAINS[name]
    adapt_ids = set(domain_split(name, eval_fraction=eval_fraction, seed=seed)[0])
    samples = [s for s in index_domain(name, split="train", require_label=False)
               if s.sample_id in adapt_ids][:n_est]
    dataset = EvalDataset(name, split="train", sample_ids=[s.sample_id for s in samples])
    predict = make_predict_fn(model, device, use_amp)
    fgfrac = np.zeros(len(grid))
    n = 0

    for i in range(len(dataset)):
        item = dataset[i]
        prob = sliding_window_predict(predict, item["image"][0].numpy(), tile=tile, overlap=overlap)
        prob_native = upsample_to_native(prob, item["native_shape"])
        fov = item["fov_native"].astype(bool)
        total = float(fov.sum())
        fgfrac += np.array([float(((prob_native > t) & fov).sum()) / max(1.0, total) for t in grid])
        n += 1

    fgfrac /= max(1, n)
    tau_prior = float(grid[int(np.argmin(np.abs(fgfrac - pi_s)))])

    return max(0.5, tau_prior), tau_prior
