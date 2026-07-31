import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import index_domain
from dataset.fov import derive_fov
from dataset.fov import load_fov_mask
from dataset.harmonization import _zscore_fov
from dataset.harmonization import harmonize
from dataset.harmonization import rescale_image
from dataset.harmonization import rescale_mask
from dataset.harmonization import working_channel
from dataset.photometric import photometric_augment
from dataset.scale import compute_domain_scale

__all__ = ["SourceDataset", "TargetTrainDataset", "TargetAdaptDataset", "EvalDataset"]


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _load_label(path):
    arr = np.asarray(Image.open(path).convert("L"))
    threshold = arr.max() / 2.0 if arr.max() > 0 else 0

    return arr > threshold


def _resolve_fov(sample, green):
    if sample.fov_path:
        return load_fov_mask(sample.fov_path, target_shape=green.shape)

    return derive_fov(green)


def _prepare_image_side(sample, spec, scale, photometric=None, invert=None, norm="zscore",
                        channel="green"):
    rgb = _load_rgb(sample.image_path)
    green = working_channel(rgb)
    fov = _resolve_fov(sample, green)
    inv = spec.invert if invert is None else invert
    work = harmonize(rgb, fov, inv, norm=norm, photometric=photometric, channel=channel)
    work = rescale_image(work, scale)
    fov = rescale_mask(fov, scale)

    return work, fov


def fdh_dataset_kwargs(norm="zscore", channel="green", caliber=True, polarity=True):
    kwargs = {"norm": norm, "channel": channel}

    if not caliber:
        kwargs["scale"] = 1.0

    if not polarity:
        kwargs["invert_override"] = False

    return kwargs


def _pad_to_size(work, fov, label, size):
    h, w = work.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)

    if pad_h or pad_w:
        mode = "reflect" if (h > 1 and w > 1 and pad_h < h and pad_w < w) else "edge"
        work = np.pad(work, ((0, pad_h), (0, pad_w)), mode=mode)
        fov = np.pad(fov, ((0, pad_h), (0, pad_w)), mode="constant")

        if label is not None:
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode="constant")

    return work, fov, label


def _random_fov_crop(work, fov, label, size, rng):
    h, w = work.shape
    max_y = h - size
    max_x = w - size
    best_cov = -1.0
    best = (0, 0)

    for _ in range(20):
        y = int(rng.integers(0, max_y + 1))
        x = int(rng.integers(0, max_x + 1))
        cov = float(fov[y:y + size, x:x + size].mean())

        if cov > best_cov:
            best_cov = cov
            best = (y, x)

        if cov >= 0.5:
            break

    y, x = best
    work = work[y:y + size, x:x + size]
    fov = fov[y:y + size, x:x + size]

    if label is not None:
        label = label[y:y + size, x:x + size]

    return work, fov, label


def _augment_geometric(work, fov, label, rng):
    if rng.random() < 0.5:
        work = work[:, ::-1]
        fov = fov[:, ::-1]
        label = label[:, ::-1] if label is not None else None

    if rng.random() < 0.5:
        work = work[::-1, :]
        fov = fov[::-1, :]
        label = label[::-1, :] if label is not None else None

    k = int(rng.integers(0, 4))

    if k:
        work = np.rot90(work, k)
        fov = np.rot90(fov, k)
        label = np.rot90(label, k) if label is not None else None

    return work, fov, label


def _make_patch(work, fov, label, size, augment, rng):
    if augment:
        jitter = float(rng.uniform(0.9, 1.1))

        if abs(jitter - 1.0) > 1e-3:
            work = rescale_image(work, jitter)
            fov = rescale_mask(fov, jitter)
            label = rescale_mask(label, jitter) if label is not None else None

    work, fov, label = _pad_to_size(work, fov, label, size)
    work, fov, label = _random_fov_crop(work, fov, label, size, rng)

    if augment:
        work, fov, label = _augment_geometric(work, fov, label, rng)

    return work, fov, label


def _to_tensor(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    return torch.from_numpy(arr).unsqueeze(0)


class _BaseDomainDataset(Dataset):
    def __init__(self, name, split="train", patch_size=512, augment=True,
                 target_diameter=1024.0, scale=None, require_label=True, sample_ids=None,
                 invert_override=None, norm="zscore", channel="green"):
        self.name = name
        self.spec = DOMAINS[name]
        self.invert = self.spec.invert if invert_override is None else bool(invert_override)
        self.norm = norm
        self.channel = channel
        self.split = split
        self.samples = index_domain(name, split=split, require_label=require_label)

        if sample_ids is not None:
            allowed = set(sample_ids)
            self.samples = [s for s in self.samples if s.sample_id in allowed]

        self.patch_size = patch_size
        self.augment = augment
        self.target_diameter = target_diameter
        self.scale = float(scale) if scale is not None else compute_domain_scale(name, target_diameter)
        self._draw_count = 0

    def __len__(self):
        return len(self.samples)

    def _rng(self, idx):
        count = self._draw_count
        self._draw_count += 1

        return np.random.default_rng(np.random.SeedSequence(
            entropy=int(torch.initial_seed()), spawn_key=(idx, count)))


class SourceDataset(_BaseDomainDataset):
    def __init__(self, name, split="train", patch_size=512, augment=True,
                 target_diameter=1024.0, scale=None, sample_ids=None, photometric=False,
                 invert_override=None, norm="zscore", channel="green"):
        super().__init__(name, split=split, patch_size=patch_size, augment=augment,
                         target_diameter=target_diameter, scale=scale, require_label=True,
                         sample_ids=sample_ids, invert_override=invert_override,
                         norm=norm, channel=channel)
        self.photometric = photometric

    def __getitem__(self, idx):
        rng = self._rng(idx)
        sample = self.samples[idx]
        photometric = (lambda g, f: photometric_augment(g, f, rng)) if self.photometric else None
        work, fov = _prepare_image_side(sample, self.spec, self.scale,
                                        photometric=photometric, invert=self.invert,
                                        norm=self.norm, channel=self.channel)
        label = rescale_mask(_load_label(sample.label_path), self.scale)
        work, fov, label = _make_patch(work, fov, label, self.patch_size, self.augment, rng)

        return {
            "image": _to_tensor(work),
            "label": _to_tensor(label),
            "fov": _to_tensor(fov),
            "domain": self.name,
        }


class TargetTrainDataset(_BaseDomainDataset):
    def __init__(self, name, split="train", patch_size=512, augment=True,
                 target_diameter=1024.0, scale=None, sample_ids=None,
                 invert_override=None, norm="zscore", channel="green"):
        super().__init__(name, split=split, patch_size=patch_size, augment=augment,
                         target_diameter=target_diameter, scale=scale, require_label=False,
                         sample_ids=sample_ids, invert_override=invert_override,
                         norm=norm, channel=channel)

    def __getitem__(self, idx):
        rng = self._rng(idx)
        sample = self.samples[idx]

        assert sample.label_path is None, "Target-train samples must carry no label handle."

        work, fov = _prepare_image_side(sample, self.spec, self.scale, invert=self.invert,
                                        norm=self.norm, channel=self.channel)
        work, fov, _ = _make_patch(work, fov, None, self.patch_size, self.augment, rng)

        return {
            "image": _to_tensor(work),
            "fov": _to_tensor(fov),
            "domain": self.name,
            "sample_id": sample.sample_id,
        }


class TargetAdaptDataset(_BaseDomainDataset):
    def __init__(self, name, split="train", patch_size=512, target_diameter=1024.0,
                 scale=None, sample_ids=None, n_views=2, invert_override=None, intervention_strength=1.0,
                 norm="zscore", channel="green"):
        super().__init__(name, split=split, patch_size=patch_size, augment=False,
                         target_diameter=target_diameter, scale=scale, require_label=False,
                         sample_ids=sample_ids, invert_override=invert_override,
                         norm=norm, channel=channel)
        self.n_views = n_views
        self.intervention_strength = intervention_strength

    def _normalize_patch(self, raw, fov):
        if self.norm == "zscore":
            return _zscore_fov(raw, fov)

        if self.norm == "unit":
            return raw.astype(np.float32) / 255.0

        raise ValueError(f"Unknown norm {self.norm!r} for the adaptation stream.")

    def __getitem__(self, idx):
        rng = self._rng(idx)
        sample = self.samples[idx]
        raw, fov = _prepare_image_side(sample, self.spec, self.scale, invert=self.invert, norm="raw",
                                       channel=self.channel)
        raw, fov, _ = _make_patch(raw, fov, None, self.patch_size, augment=False, rng=rng)
        fov_f = fov.astype(np.float32)
        clean = self._normalize_patch(raw, fov) * fov_f
        views = []

        for _ in range(self.n_views):
            aug = photometric_augment(np.clip(raw, 0.0, 255.0).astype(np.uint8), fov, rng,
                                      strength=self.intervention_strength)
            views.append(_to_tensor(self._normalize_patch(aug.astype(np.float32), fov) * fov_f))

        return {
            "image": _to_tensor(clean),
            "views": torch.stack(views, dim=0),
            "fov": _to_tensor(fov),
            "domain": self.name,
            "sample_id": sample.sample_id,
        }


class EvalDataset(_BaseDomainDataset):
    def __init__(self, name, split="train", patch_size=512, target_diameter=1024.0, scale=None,
                 sample_ids=None, invert_override=None, norm="zscore", channel="green"):
        super().__init__(name, split=split, patch_size=patch_size, augment=False,
                         target_diameter=target_diameter, scale=scale, sample_ids=sample_ids,
                         invert_override=invert_override, norm=norm, channel=channel)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb = _load_rgb(sample.image_path)
        green = working_channel(rgb)
        fov_native = _resolve_fov(sample, green)
        work = harmonize(rgb, fov_native, self.invert, norm=self.norm, channel=self.channel)
        work_scaled = rescale_image(work, self.scale)
        label_native = _load_label(sample.label_path)

        return {
            "image": _to_tensor(work_scaled),
            "label_native": label_native,
            "fov_native": fov_native,
            "native_shape": tuple(green.shape),
            "scale": self.scale,
            "domain": self.name,
            "sample_id": sample.sample_id,
        }
