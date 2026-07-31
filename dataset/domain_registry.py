import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

__all__ = [
    "DomainSpec",
    "Sample",
    "DOMAINS",
    "TARGET_DOMAINS",
    "SOURCE_DOMAIN",
    "index_domain",
]


@dataclass(frozen=True)
class DomainSpec:
    name: str
    role: str
    modality: str
    invert: bool
    image_subdir: str
    label_subdir: str
    fov_subdir: Optional[str]
    image_exts: Tuple[str, ...]
    label_pattern: str
    fov_pattern: Optional[str]
    id_strip_suffix: Optional[str] = None
    eval_image_subdir: Optional[str] = None
    eval_label_subdir: Optional[str] = None


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    label_path: Optional[str]
    fov_path: Optional[str]


SOURCE_DOMAIN = "DRIVEN"

TARGET_DOMAINS = (
    "FIVES",
    "CHASEDB1",
    "IOSTAR",
    "JRCFA",
    "JRCFAF",
    "JRCIR",
    "STARE_AH",
)

DOMAINS = {
    "FIVES": DomainSpec(
        name="FIVES",
        role="target",
        modality="color_fundus",
        invert=False,
        image_subdir="train/img",
        label_subdir="train/label",
        fov_subdir=None,
        image_exts=(".png",),
        label_pattern="{id}.png",
        fov_pattern=None,
        eval_image_subdir="test/img",
        eval_label_subdir="test/label",
    ),
    "CHASEDB1": DomainSpec(
        name="CHASEDB1",
        role="target",
        modality="color_fundus",
        invert=False,
        image_subdir="img",
        label_subdir="label",
        fov_subdir="mask",
        image_exts=(".jpg",),
        label_pattern="{id}_1stHO.png",
        fov_pattern="{id}_mask.png",
    ),
    "DRIVEN": DomainSpec(
        name="DRIVEN",
        role="source",
        modality="color_fundus",
        invert=False,
        image_subdir="train/img",
        label_subdir="train/label",
        fov_subdir="train/mask",
        image_exts=(".tif",),
        label_pattern="{id}_manual1.gif",
        fov_pattern="{id}_training_mask.gif",
        id_strip_suffix="_training",
    ),
    "IOSTAR": DomainSpec(
        name="IOSTAR",
        role="target",
        modality="slo",
        invert=False,
        image_subdir="img",
        label_subdir="label",
        fov_subdir="mask",
        image_exts=(".jpg",),
        label_pattern="{id}_GT.tif",
        fov_pattern="{id}_Mask.tif",
    ),
    "JRCFA": DomainSpec(
        name="JRCFA",
        role="target",
        modality="fluorescein_angiography",
        invert=True,
        image_subdir="img",
        label_subdir="label",
        fov_subdir=None,
        image_exts=(".png",),
        label_pattern="{id}.png",
        fov_pattern=None,
    ),
    "JRCFAF": DomainSpec(
        name="JRCFAF",
        role="target",
        modality="autofluorescence",
        invert=False,
        image_subdir="img",
        label_subdir="label",
        fov_subdir=None,
        image_exts=(".png",),
        label_pattern="{id}.png",
        fov_pattern=None,
    ),
    "JRCIR": DomainSpec(
        name="JRCIR",
        role="target",
        modality="infrared_reflectance",
        invert=False,
        image_subdir="img",
        label_subdir="label",
        fov_subdir=None,
        image_exts=(".png",),
        label_pattern="{id}.png",
        fov_pattern=None,
    ),
    "STARE_AH": DomainSpec(
        name="STARE_AH",
        role="target",
        modality="color_fundus",
        invert=False,
        image_subdir="img",
        label_subdir="label",
        fov_subdir=None,
        image_exts=(".png",),
        label_pattern="{id}.png",
        fov_pattern=None,
    ),
}


def _list_images(directory, exts):
    paths = []

    for entry in sorted(os.listdir(directory)):

        if entry.startswith("."):
            continue

        full = directory / entry

        if full.is_file() and full.suffix.lower() in exts:
            paths.append(full)

    return paths


def index_domain(name, split="train", require_label=True):
    spec = DOMAINS[name]
    image_subdir = spec.image_subdir
    label_subdir = spec.label_subdir

    if split == "test":

        if spec.eval_image_subdir is None:
            raise ValueError(f"Domain {name!r} has no 'test' evaluation split.")

        image_subdir = spec.eval_image_subdir
        label_subdir = spec.eval_label_subdir

    domain_root = DATA_ROOT / name
    image_dir = domain_root / image_subdir
    label_dir = domain_root / label_subdir
    fov_dir = domain_root / spec.fov_subdir if spec.fov_subdir else None
    samples = []

    for image_path in _list_images(image_dir, spec.image_exts):
        stem = image_path.stem
        sample_id = stem

        if spec.id_strip_suffix and stem.endswith(spec.id_strip_suffix):
            sample_id = stem[: -len(spec.id_strip_suffix)]

        label_path = None

        if require_label:
            label_file = label_dir / spec.label_pattern.format(id=sample_id)

            if not label_file.is_file():
                raise FileNotFoundError(f"Missing label for {name}/{sample_id}: {label_file}")

            label_path = str(label_file)

        fov_path = None

        if fov_dir is not None:
            candidate = fov_dir / spec.fov_pattern.format(id=sample_id)

            if not candidate.is_file():
                raise FileNotFoundError(f"Missing FOV for {name}/{sample_id}: {candidate}")

            fov_path = str(candidate)

        samples.append(Sample(sample_id, str(image_path), label_path, fov_path))

    return samples
