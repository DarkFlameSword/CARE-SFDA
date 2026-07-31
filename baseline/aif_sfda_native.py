import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from aif_sfda_baseline import AIFAdapt
from aif_sfda_baseline import build_unet
from aif_sfda_baseline import _infinite
from aif_sfda_baseline import _linear_lr
from aif_sfda_baseline import _pad_to_multiple
from aif_sfda_baseline import _resolve_fov
from aif_sfda_baseline import modality_groups
from aif_sfda_baseline import summarize
from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.domain_registry import index_domain
from dataset.retinal_vessel_dataset import _load_label
from dataset.retinal_vessel_dataset import _load_rgb
from dataset.retinal_vessel_dataset import _to_tensor
from dataset.harmonization import working_channel
from dataset.splits import domain_split
from util.evaluate_metrics import evaluate as care_evaluate

SOURCE_VAL_FRACTION = 0.15
TARGET_EVAL_FRACTION = 0.30
EVAL_SIZE = 512


def _native_rgb(sample, invert):
    rgb = _load_rgb(sample.image_path)
    fov = _resolve_fov(sample, working_channel(rgb))

    if invert:
        rgb = (255 - rgb) * fov[..., None]

    return rgb.astype(np.float32) / 255.0, fov


class NativePatchDataset(Dataset):
    def __init__(self, name, sample_ids, invert, patch=512, with_label=True, seed=0):
        allowed = set(sample_ids)
        self.samples = [s for s in index_domain(name, split="train", require_label=with_label)
                        if s.sample_id in allowed]
        self.invert = invert
        self.patch = patch
        self.with_label = with_label
        self.seed = seed
        self._draw = 0

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed, spawn_key=(idx, self._draw)))
        self._draw += 1
        sample = self.samples[idx]
        rgb, fov = _native_rgb(sample, self.invert)
        label = _load_label(sample.label_path).astype(np.float32) if self.with_label else np.zeros(fov.shape, np.float32)
        size = (self.patch, self.patch)
        rgb = sk_resize(rgb, size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        fov = (sk_resize(fov.astype(np.float32), size, order=0, preserve_range=True, anti_aliasing=False) > 0.5
               ).astype(np.float32)
        label = (sk_resize(label, size, order=0, preserve_range=True, anti_aliasing=False) > 0.5).astype(np.float32)

        if rng.random() < 0.5:
            rgb, fov, label = rgb[:, ::-1], fov[:, ::-1], label[:, ::-1]

        k = int(rng.integers(0, 4))

        if k:
            rgb, fov, label = np.rot90(rgb, k), np.rot90(fov, k), np.rot90(label, k)

        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))

        return {"image": image, "label": _to_tensor(label), "mask": _to_tensor(fov)}


def pretrain_source(source_ids, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_unet().to(device)
    dataset = NativePatchDataset(SOURCE_DOMAIN, source_ids, invert=DOMAINS[SOURCE_DOMAIN].invert,
                                 with_label=True, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    net.train()

    for step in range(iters):
        batch = next(loader)
        logits = net(batch["image"].to(device))[0]
        loss = criterion(logits, batch["label"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _linear_lr(optimizer, 1e-3, step, iters)

        if (step + 1) % 1000 == 0:
            print(f"[native-pretrain] iter {step + 1}/{iters} bce {float(loss):.4f}", flush=True)

    return net.state_dict()


def adapt_target(name, source_state, iters, device, seed, patch=512):
    torch.manual_seed(seed)
    np.random.seed(seed)
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    dataset = NativePatchDataset(name, adapt_ids, invert=DOMAINS[name].invert, patch=patch,
                                 with_label=False, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    model = AIFAdapt(source_state, device)
    model.net_filter.train()
    model.net_student.train()

    for step in range(iters):
        batch = next(loader)
        losses = model.step(batch["image"].to(device), batch["mask"].to(device))
        _linear_lr(model.opt_filter, 1e-3, step, iters)
        _linear_lr(model.opt_student, 1e-3, step, iters)

        if (step + 1) % 200 == 0:
            print(f"[native-adapt:{name}] iter {step + 1}/{iters} seg {losses['seg']:.3f} "
                  f"mi {losses['mi']:.3f} con {losses['con']:.3f}", flush=True)

    return model


def evaluate_target(model, name, device, seed):
    eval_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[1]
    invert = DOMAINS[name].invert
    id_to_sample = {s.sample_id: s for s in index_domain(name, split="train", require_label=True)}
    per_image = []

    for sid in eval_ids:
        sample = id_to_sample[sid]
        rgb, fov = _native_rgb(sample, invert)
        label_native = _load_label(sample.label_path)
        native_shape = fov.shape
        rgb_eval = sk_resize(rgb, (EVAL_SIZE, EVAL_SIZE), order=1, preserve_range=True,
                             anti_aliasing=True).astype(np.float32)
        fov_eval = sk_resize(fov.astype(np.float32), (EVAL_SIZE, EVAL_SIZE), order=0, preserve_range=True,
                             anti_aliasing=False) > 0.5
        image = torch.from_numpy(np.ascontiguousarray(rgb_eval.transpose(2, 0, 1))).unsqueeze(0).to(device)
        mask = _to_tensor(fov_eval.astype(np.float32)).unsqueeze(0).to(device)
        image, h, w = _pad_to_multiple(image)
        mask, _, _ = _pad_to_multiple(mask)
        prob = model.predict(image, mask)[0, 0, :h, :w].cpu().numpy()
        prob_native = sk_resize(prob, native_shape, order=1, preserve_range=True, anti_aliasing=False)
        per_image.append(care_evaluate(prob_native, label_native, roi=fov, threshold=0.5))
        torch.cuda.empty_cache()

    keys = ["dice", "cldice", "iou", "sensitivity", "specificity"]

    return {k: float(np.mean([m[k] for m in per_image])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="B-3 AIF-SFDA native-preprocessing baseline.")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--pretrain-iters", type=int, default=6000)
    parser.add_argument("--adapt-iters", type=int, default=2000)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/baselines_7targets") / f"aif_sfda_native_{args.date}"
    (run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    source_ids = domain_split(SOURCE_DOMAIN, eval_fraction=SOURCE_VAL_FRACTION, seed=args.seed)[0]

    config = {
        "experiment": "B-3 AIF-SFDA (AAAI 2025) native-preprocessing reproduction",
        "regime": "AIF paper protocol: RGB, resize-whole to 512x512 (load_size, train+eval), no green, "
                  "no scale-1024, [0,1]+InstanceNorm, flip/rot aug",
        "controlled_only": {"source": f"DRIVEN train split seed {args.seed} ({len(source_ids)} imgs)",
                            "splits": f"domain_split(t, {TARGET_EVAL_FRACTION}, {args.seed}) identical to CARE",
                            "evaluation": "CARE util.evaluate_metrics on native GT/FOV (dice/cldice/iou)",
                            "shared_physical_necessity": "JRCFA polarity flip only (green + scale stay native)"},
        "protocol": f"label-free, pretrain {args.pretrain_iters} / adapt {args.adapt_iters} iters, final ckpt",
        "seed": args.seed,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    source_ckpt = run_dir / "checkpoint" / "source.pt"

    if source_ckpt.exists():
        print(f"[native] loading existing source checkpoint {source_ckpt}", flush=True)
        source_state = torch.load(source_ckpt, map_location=device)

    else:
        print(f"[native] pretraining source on DRIVEN RGB ({len(source_ids)} imgs, {args.pretrain_iters} iters)", flush=True)
        source_state = pretrain_source(source_ids, args.pretrain_iters, device, args.seed)
        torch.save(source_state, source_ckpt)

    metric_path = run_dir / "metric.json"
    results = json.load(open(metric_path, encoding="utf-8"))["per_domain"] if metric_path.exists() else {}

    for name in targets:

        if name in results:
            print(f"[native] {name} already done, skipping", flush=True)
            continue

        print(f"[native] adapting {name} ({args.adapt_iters} iters, invert={DOMAINS[name].invert})", flush=True)
        model = adapt_target(name, source_state, args.adapt_iters, device, args.seed, args.patch)
        results[name] = evaluate_target(model, name, device, args.seed)
        print(f"[native] {name}: dice {results[name]['dice']:.4f} cldice {results[name]['cldice']:.4f} "
              f"iou {results[name]['iou']:.4f}", flush=True)
        torch.save({"net_filter": model.net_filter.state_dict(), "net_student": model.net_student.state_dict()},
                   run_dir / "checkpoint" / f"{name}_adapted.pt")

        groups = modality_groups(list(results.keys()))
        payload = {"config": config, "per_domain": results,
                   "dice_summary": summarize({d: results[d]["dice"] for d in results}, groups),
                   "cldice_summary": summarize({d: results[d]["cldice"] for d in results}, groups),
                   "iou_summary": summarize({d: results[d]["iou"] for d in results}, groups)}

        with open(metric_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"[native] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
