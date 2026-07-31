import argparse
import contextlib
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_CARE_ROOT = os.path.dirname(_HERE)
_UPL_ROOT = os.path.join(_CARE_ROOT, "model", "UPL-SFDA")
sys.path.insert(0, _CARE_ROOT)
sys.path.insert(0, _UPL_ROOT)

sys.modules.setdefault("SimpleITK", types.ModuleType("SimpleITK"))

from unet import UNet

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.domain_registry import index_domain
from dataset.harmonization import working_channel
from dataset.retinal_vessel_dataset import _load_label
from dataset.retinal_vessel_dataset import _load_rgb
from dataset.retinal_vessel_dataset import _resolve_fov
from dataset.retinal_vessel_dataset import _to_tensor
from dataset.splits import domain_split
from util.evaluate_metrics import evaluate as care_evaluate

SOURCE_VAL_FRACTION = 0.15
TARGET_EVAL_FRACTION = 0.30
INPUT_SIZE = 512
PL_THRESHOLD = 0.9


def _make_upl_config(lr):
    return {
        "train": {"lr": lr, "dataset": "fb", "pl_threshold_fb": PL_THRESHOLD},
        "network": {"in_chns": 1, "n_classes_fb": 2, "ft_chns_fb": [8, 16, 32, 64, 128],
                    "dropout_p": [0, 0.5], "up_mode": "upsample"},
    }


def _winadj_fov(green, fov):
    fov_bool = fov.astype(bool)
    vals = green[fov_bool]

    if vals.size == 0:

        return np.zeros_like(green, dtype=np.float32)

    v0 = float(np.percentile(vals, 1))
    v1 = float(np.percentile(vals, 99))

    if v1 - v0 < 1e-6:

        return np.zeros_like(green, dtype=np.float32)

    clipped = np.clip(green, v0, v1)
    norm = (clipped - v0) / (v1 - v0) * 2.0 - 1.0

    return (norm * fov_bool.astype(np.float32)).astype(np.float32)


def _upl_input(sample, invert):
    rgb = _load_rgb(sample.image_path)
    green = working_channel(rgb).astype(np.float32)
    fov = _resolve_fov(sample, working_channel(rgb))

    if invert:
        green = (255.0 - green) * fov.astype(np.float32)

    return _winadj_fov(green, fov), fov


def _infinite(loader):
    while True:

        for batch in loader:

            yield batch


def modality_groups(targets):
    groups = {}

    for domain in targets:
        groups.setdefault(DOMAINS[domain].modality, []).append(domain)

    return groups


def summarize(per_domain, groups):
    group_value = {m: float(np.mean([per_domain[d] for d in members])) for m, members in groups.items()}
    values = list(group_value.values())

    return {"mean": float(np.mean(list(per_domain.values()))),
            "worst": float(min(values)), "spread": float(max(values) - min(values)),
            "group_dice": group_value}


class UPLPatchDataset(Dataset):
    def __init__(self, name, sample_ids, invert, size=INPUT_SIZE, with_label=True, augment=True, seed=0):
        allowed = set(sample_ids)
        self.samples = [s for s in index_domain(name, split="train", require_label=with_label)
                        if s.sample_id in allowed]
        self.invert = invert
        self.size = size
        self.with_label = with_label
        self.augment = augment
        self.seed = seed
        self._draw = 0

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed, spawn_key=(idx, self._draw)))
        self._draw += 1
        sample = self.samples[idx]
        norm, fov = _upl_input(sample, self.invert)
        label = _load_label(sample.label_path).astype(np.float32) if self.with_label else np.zeros(fov.shape, np.float32)
        size = (self.size, self.size)
        norm = sk_resize(norm, size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        label = (sk_resize(label, size, order=0, preserve_range=True, anti_aliasing=False) > 0.5).astype(np.float32)

        if self.augment:

            if rng.random() < 0.5:
                norm = norm[:, ::-1]
                label = label[:, ::-1]

            k = int(rng.integers(0, 4))

            if k:
                norm = np.rot90(norm, k)
                label = np.rot90(label, k)

        return {"image": _to_tensor(norm), "label": _to_tensor(label)}


def pretrain_source(source_ids, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    upl = UNet(_make_upl_config(1e-3)).to(device)
    upl.train()
    dataset = UPLPatchDataset(SOURCE_DOMAIN, source_ids, invert=DOMAINS[SOURCE_DOMAIN].invert,
                              with_label=True, augment=True, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))

    for step in range(iters):
        batch = next(loader)
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        upl.forward(image)
        loss = upl.segloss(upl.aux_seg_1, label, one_hot=True)
        upl.enc_opt.zero_grad(set_to_none=True)
        upl.aux_dec1_opt.zero_grad(set_to_none=True)
        loss.backward()
        upl.enc_opt.step()
        upl.aux_dec1_opt.step()

        if (step + 1) % 1000 == 0:
            print(f"[upl-pretrain] iter {step + 1}/{iters} dice {float(loss):.4f}", flush=True)

    return upl.state_dict()


def adapt_target(name, source_state, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    upl = UNet(_make_upl_config(1e-5)).to(device)
    upl.load_state_dict(source_state, strict=False)
    decoder_state = upl.aux_dec1.state_dict()
    upl.aux_dec2.load_state_dict(decoder_state)
    upl.aux_dec3.load_state_dict(decoder_state)
    upl.aux_dec4.load_state_dict(decoder_state)
    upl.train()
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    dataset = UPLPatchDataset(name, adapt_ids, invert=DOMAINS[name].invert,
                              with_label=False, augment=False, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True))

    with open(os.devnull, "w") as devnull:

        for step in range(iters):
            batch = next(loader)
            image = batch["image"].to(device)

            with contextlib.redirect_stdout(devnull):
                upl.save_nii(image)
                dice_loss, entropy_loss = upl.trian_target(image)

            if (step + 1) % 200 == 0:
                print(f"[upl-adapt:{name}] iter {step + 1}/{iters} dice {dice_loss:.3f} ent {entropy_loss:.3f}",
                      flush=True)

    return upl


def evaluate_target(upl, name, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[1]
    invert = DOMAINS[name].invert
    id_to_sample = {s.sample_id: s for s in index_domain(name, split="train", require_label=True)}
    upl.train()
    per_image = []

    for sid in eval_ids:
        sample = id_to_sample[sid]
        norm, fov = _upl_input(sample, invert)
        label_native = _load_label(sample.label_path)
        native_shape = fov.shape
        norm_eval = sk_resize(norm, (INPUT_SIZE, INPUT_SIZE), order=1, preserve_range=True,
                              anti_aliasing=True).astype(np.float32)
        image = _to_tensor(norm_eval).unsqueeze(0).to(device)

        with torch.no_grad():
            mean_map = upl.test_with_name(image)

        prob = mean_map[0, 1].cpu().numpy()
        prob_native = sk_resize(prob, native_shape, order=1, preserve_range=True, anti_aliasing=False)
        per_image.append(care_evaluate(prob_native, label_native, roi=fov, threshold=0.5))
        torch.cuda.empty_cache()

    keys = ["dice", "cldice", "iou", "sensitivity", "specificity"]

    return {k: float(np.mean([m[k] for m in per_image])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="B-5 UPL-SFDA native-design baseline.")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--pretrain-iters", type=int, default=6000)
    parser.add_argument("--adapt-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/baselines_7targets") / f"upl_sfda_native_{args.date}"
    (run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    source_ids = domain_split(SOURCE_DOMAIN, eval_fraction=SOURCE_VAL_FRACTION, seed=args.seed)[0]

    config = {
        "experiment": "B-5 UPL-SFDA (IEEE TMI 2023) native-design reproduction",
        "method": "UPL 4-decoder rotation-consistency multi-head U-Net; pseudo-label = 4-head mean gated at "
                  "0.9 + largest connected component; adapt = masked-Dice across heads + mean-map entropy min "
                  "(verbatim model/UPL-SFDA/unet.py)",
        "regime": "home-field: green channel -> polarity flip (JRCFA) + re-zero FOV -> winadj[1,99]->[-1,1] "
                  "over FOV -> resize whole to 512x512; train-mode (per-image BN) inference",
        "controlled_only": {"source": f"DRIVEN train split seed {args.seed} ({len(source_ids)} imgs)",
                            "splits": f"domain_split(t, {TARGET_EVAL_FRACTION}, {args.seed}) identical to CARE",
                            "evaluation": "CARE util.evaluate_metrics on native GT/FOV (dice/cldice/iou)",
                            "label_free": "fixed adapt budget, final checkpoint, NO target-label selection "
                                          "(UPL's valid-best selection dropped)",
                            "shared_physical_necessity": "JRCFA polarity flip + re-zero FOV only"},
        "learning_rates": {"source_pretrain": "Adam 1e-3 (matched to AIF-native weak source)",
                           "adaptation": "Adam 1e-5 (UPL native), constant (per-epoch StepLR omitted "
                                         "under fixed-iteration budget)"},
        "protocol": f"pretrain {args.pretrain_iters} / adapt {args.adapt_iters} iters, batch 2 (source) / 1 (adapt)",
        "seed": args.seed,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    source_ckpt = run_dir / "checkpoint" / "source.pt"

    if source_ckpt.exists():
        print(f"[upl] loading existing source checkpoint {source_ckpt}", flush=True)
        source_state = torch.load(source_ckpt, map_location=device)

    else:
        print(f"[upl] pretraining source on DRIVEN ({len(source_ids)} imgs, {args.pretrain_iters} iters)", flush=True)
        source_state = pretrain_source(source_ids, args.pretrain_iters, device, args.seed)
        torch.save(source_state, source_ckpt)

    metric_path = run_dir / "metric.json"
    results = json.load(open(metric_path, encoding="utf-8"))["per_domain"] if metric_path.exists() else {}

    for name in targets:

        if name in results:
            print(f"[upl] {name} already done, skipping", flush=True)
            continue

        print(f"[upl] adapting {name} ({args.adapt_iters} iters, invert={DOMAINS[name].invert})", flush=True)
        model = adapt_target(name, source_state, args.adapt_iters, device, args.seed)
        results[name] = evaluate_target(model, name, device, args.seed)
        print(f"[upl] {name}: dice {results[name]['dice']:.4f} cldice {results[name]['cldice']:.4f} "
              f"iou {results[name]['iou']:.4f}", flush=True)
        torch.save(model.state_dict(), run_dir / "checkpoint" / f"{name}_adapted.pt")

        groups = modality_groups(list(results.keys()))
        payload = {"config": config, "per_domain": results,
                   "dice_summary": summarize({d: results[d]["dice"] for d in results}, groups),
                   "cldice_summary": summarize({d: results[d]["cldice"] for d in results}, groups),
                   "iou_summary": summarize({d: results[d]["iou"] for d in results}, groups)}

        with open(metric_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"[upl] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
