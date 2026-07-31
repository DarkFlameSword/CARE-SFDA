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
_CARE_ROOT = os.path.dirname(_HERE)
_CBMT_ROOT = os.path.join(_CARE_ROOT, "model", "SFDA-CBMT")
sys.path.insert(0, _CARE_ROOT)
sys.path.insert(0, _CBMT_ROOT)

from networks.backbone import mobilenet as _cbmt_mobilenet

_cbmt_mobilenet.MobileNetV2._load_pretrained_model = lambda self: None

from networks.deeplabv3 import DeepLab

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.domain_registry import index_domain
from dataset.harmonization import working_channel
from dataset.retinal_vessel_dataset import _load_label
from dataset.retinal_vessel_dataset import _load_rgb
from dataset.retinal_vessel_dataset import _resolve_fov
from dataset.splits import domain_split
from util.evaluate_metrics import evaluate as care_evaluate

SOURCE_VAL_FRACTION = 0.15
TARGET_EVAL_FRACTION = 0.30
INPUT_SIZE = 512
PL_THRESHOLD = 0.75
EMA_RATE = 0.98
CB_BOUND_RATIO = 0.2


def _strong_aug(rgb, rng):
    img = rgb.copy()
    h, w, c = img.shape

    seed = rng.random()

    if seed > 0.75:
        n = int(np.ceil(0.004 * img.size * 0.2))
        ys = rng.integers(0, h - 1, n)
        xs = rng.integers(0, w - 1, n)
        img[ys, xs, :] = 1

    elif seed > 0.5:
        n = int(np.ceil(0.004 * img.size * 0.8))
        ys = rng.integers(0, h - 1, n)
        xs = rng.integers(0, w - 1, n)
        img[ys, xs, :] = 0

    if rng.random() > 0.5:
        gamma = rng.random() * 3 + 0.5
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype(np.uint8)
        img = table[img]

    if rng.random() > 0.5:
        s = rng.uniform(0.02, 0.06) * h * w
        r = rng.uniform(0.3, 0.6)
        ew = int(np.sqrt(s / r))
        eh = int(np.sqrt(s * r))

        if 0 < ew < w and 0 < eh < h:
            left = int(rng.integers(0, w - ew))
            top = int(rng.integers(0, h - eh))
            img[top:top + eh, left:left + ew, :] = int(rng.uniform(0, 255))

    return img


def _cbmt_input(sample, invert):
    rgb = _load_rgb(sample.image_path)
    fov = _resolve_fov(sample, working_channel(rgb))

    if invert:
        rgb = ((255 - rgb) * fov[..., None]).astype(np.uint8)

    return rgb.astype(np.uint8), fov


def _to_input_tensor(rgb_uint8):
    resized = sk_resize(rgb_uint8, (INPUT_SIZE, INPUT_SIZE), order=1, preserve_range=True,
                        anti_aliasing=True).astype(np.float32)
    normed = resized / 127.5 - 1.0

    return torch.from_numpy(np.ascontiguousarray(normed.transpose(2, 0, 1)))


class SourceDataset(Dataset):
    def __init__(self, sample_ids, invert, seed=0):
        allowed = set(sample_ids)
        self.samples = [s for s in index_domain(SOURCE_DOMAIN, split="train", require_label=True)
                        if s.sample_id in allowed]
        self.invert = invert
        self.seed = seed
        self._draw = 0

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed, spawn_key=(idx, self._draw)))
        self._draw += 1
        sample = self.samples[idx]
        rgb, fov = _cbmt_input(sample, self.invert)
        label = _load_label(sample.label_path).astype(np.float32)
        rgb = sk_resize(rgb, (INPUT_SIZE, INPUT_SIZE), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        label = (sk_resize(label, (INPUT_SIZE, INPUT_SIZE), order=0, preserve_range=True, anti_aliasing=False) > 0.5
                 ).astype(np.float32)

        if rng.random() < 0.5:
            rgb, label = rgb[:, ::-1], label[:, ::-1]

        k = int(rng.integers(0, 4))

        if k:
            rgb, label = np.rot90(rgb, k), np.rot90(label, k)

        rgb = _strong_aug(np.ascontiguousarray(rgb), rng)
        image = torch.from_numpy(np.ascontiguousarray((rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)))
        label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        return {"image": image, "label": label}


class AdaptDataset(Dataset):
    def __init__(self, name, sample_ids, invert, seed=0):
        allowed = set(sample_ids)
        self.samples = [s for s in index_domain(name, split="train", require_label=False)
                        if s.sample_id in allowed]
        self.invert = invert
        self.seed = seed
        self._draw = 0

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed, spawn_key=(idx, self._draw)))
        self._draw += 1
        sample = self.samples[idx]
        rgb, _ = _cbmt_input(sample, self.invert)
        rgb = sk_resize(rgb, (INPUT_SIZE, INPUT_SIZE), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        weak = torch.from_numpy(np.ascontiguousarray((rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)))
        strong_rgb = _strong_aug(np.ascontiguousarray(rgb), rng)
        strong = torch.from_numpy(np.ascontiguousarray((strong_rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)))

        return {"weak": weak, "strong": strong, "name": sample.sample_id}


def _infinite(loader):
    while True:

        for batch in loader:

            yield batch


def build_deeplab(device):
    return DeepLab(num_classes=1, backbone="mobilenet", output_stride=16, sync_bn=False, freeze_bn=False).to(device)


def pretrain_source(source_ids, iters, device, seed, batch_size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_deeplab(device)
    dataset = SourceDataset(source_ids, invert=DOMAINS[SOURCE_DOMAIN].invert, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    for step in range(iters):
        batch = next(loader)
        logit = model(batch["image"].to(device))[0]
        loss = criterion(logit, batch["label"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"[cbmt-pretrain] iter {step + 1}/{iters} bce {float(loss):.4f}", flush=True)

    return model.state_dict()


@torch.no_grad()
def _init_pred_bank(model, loader, device):
    model.eval()
    bank = {}

    for batch in loader:
        prob = torch.sigmoid(model(batch["weak"].to(device))[0])

        for i, name in enumerate(batch["name"]):
            bank[name] = prob[i, 0].detach().cpu()

    model.train()

    return bank


def _class_balance_weight(pred_bank):
    lower = PL_THRESHOLD * CB_BOUND_RATIO
    upper = 1.0 - (1.0 - PL_THRESHOLD) * CB_BOUND_RATIO
    fg_sum = 0.0
    fg_num = 0
    bg_sum = 0.0
    bg_num = 0

    for p in pred_bank.values():
        bg = (p < PL_THRESHOLD) & (p > lower)
        fg = (p > PL_THRESHOLD) & (p < upper)
        bg_sum += float(-torch.log(1.0 - p[bg]).sum())
        bg_num += int(bg.sum())
        fg_sum += float(-torch.log(p[fg]).sum())
        fg_num += int(fg.sum())

    if fg_num == 0 or bg_num == 0 or bg_sum == 0:

        return 1.0

    return (fg_sum / fg_num) / (bg_sum / bg_num)


def adapt_target(name, source_state, epochs, device, seed, batch_size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    student = build_deeplab(device)
    teacher = build_deeplab(device)
    student.load_state_dict(source_state)
    teacher.load_state_dict(source_state)
    student.train()
    teacher.train()

    for p in teacher.parameters():
        p.requires_grad_(False)

    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    dataset = AdaptDataset(name, adapt_ids, invert=DOMAINS[name].invert, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    bce = nn.BCELoss(reduction="none")
    pred_bank = _init_pred_bank(teacher, DataLoader(dataset, batch_size=batch_size, num_workers=0), device)

    for epoch in range(epochs):
        loss_weight = _class_balance_weight(pred_bank)

        for batch in loader:
            weak = batch["weak"].to(device)
            strong = batch["strong"].to(device)
            student_prob = torch.sigmoid(student(strong)[0])

            with torch.no_grad():
                teacher_prob = torch.sigmoid(teacher(weak)[0])

            pseudo = (teacher_prob > PL_THRESHOLD).float()
            weight_mask = torch.ones_like(pseudo)
            weight_mask[pseudo == 0] = loss_weight
            loss = (bce(student_prob, pseudo) * weight_mask).sum() / weight_mask.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():

                for p_s, p_t in zip(student.parameters(), teacher.parameters()):
                    p_t.data.mul_(EMA_RATE).add_(p_s.data, alpha=1.0 - EMA_RATE)

                for i, nm in enumerate(batch["name"]):
                    pred_bank[nm] = teacher_prob[i, 0].detach().cpu()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"[cbmt-adapt:{name}] epoch {epoch + 1}/{epochs} cb_weight {loss_weight:.3f}", flush=True)

    return teacher


@torch.no_grad()
def evaluate_target(model, name, device, seed):
    model.eval()
    eval_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[1]
    invert = DOMAINS[name].invert
    id_to_sample = {s.sample_id: s for s in index_domain(name, split="train", require_label=True)}
    per_image = []

    for sid in eval_ids:
        sample = id_to_sample[sid]
        rgb, fov = _cbmt_input(sample, invert)
        label_native = _load_label(sample.label_path)
        image = _to_input_tensor(rgb).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(image)[0])[0, 0].cpu().numpy()
        prob_native = sk_resize(prob, fov.shape, order=1, preserve_range=True, anti_aliasing=False)
        per_image.append(care_evaluate(prob_native, label_native, roi=fov, threshold=0.5))
        torch.cuda.empty_cache()

    keys = ["dice", "cldice", "iou", "sensitivity", "specificity"]

    return {k: float(np.mean([m[k] for m in per_image])) for k in keys}


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


def main():
    parser = argparse.ArgumentParser(description="B-4 SFDA-CBMT native-design baseline.")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--pretrain-iters", type=int, default=6000)
    parser.add_argument("--adapt-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/baselines_7targets") / f"sfda_cbmt_native_{args.date}"
    (run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    source_ids = domain_split(SOURCE_DOMAIN, eval_fraction=SOURCE_VAL_FRACTION, seed=args.seed)[0]

    config = {
        "experiment": "B-4 SFDA-CBMT (MICCAI 2023) native-design reproduction",
        "method": "DeepLabV3+/MobileNet class-balanced mean teacher: weak-teacher hard PL@0.75 + strong-student, "
                  "class-balanced BCE, EMA teacher 0.98, report final teacher (verbatim model/SFDA-CBMT DeepLab)",
        "regime": "native: RGB [-1,1] (Normalize_tf) @512x512, os16, weak/strong aug; backbone trained "
                  "FROM SCRATCH (ImageNet weights unavailable: jeff95.me 404, layout != torchvision)",
        "controlled_only": {"source": f"DRIVEN train split seed {args.seed} ({len(source_ids)} imgs)",
                            "splits": f"domain_split(t, {TARGET_EVAL_FRACTION}, {args.seed}) identical to CARE",
                            "evaluation": "CARE util.evaluate_metrics on native GT/FOV (dice/cldice/iou)",
                            "label_free": "fixed epoch budget, final teacher, NO target-label selection",
                            "shared_physical_necessity": "JRCFA polarity flip + re-zero FOV only"},
        "protocol": f"pretrain {args.pretrain_iters} iters / adapt {args.adapt_epochs} epochs, batch {args.batch_size}",
        "seed": args.seed,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    source_ckpt = run_dir / "checkpoint" / "source.pt"

    if source_ckpt.exists():
        print(f"[cbmt] loading existing source checkpoint {source_ckpt}", flush=True)
        source_state = torch.load(source_ckpt, map_location=device)

    else:
        print(f"[cbmt] pretraining source on DRIVEN ({len(source_ids)} imgs, {args.pretrain_iters} iters)", flush=True)
        source_state = pretrain_source(source_ids, args.pretrain_iters, device, args.seed, args.batch_size)
        torch.save(source_state, source_ckpt)

    metric_path = run_dir / "metric.json"
    results = json.load(open(metric_path, encoding="utf-8"))["per_domain"] if metric_path.exists() else {}

    for name in targets:

        if name in results:
            print(f"[cbmt] {name} already done, skipping", flush=True)
            continue

        print(f"[cbmt] adapting {name} ({args.adapt_epochs} epochs, invert={DOMAINS[name].invert})", flush=True)
        model = adapt_target(name, source_state, args.adapt_epochs, device, args.seed, args.batch_size)
        results[name] = evaluate_target(model, name, device, args.seed)
        print(f"[cbmt] {name}: dice {results[name]['dice']:.4f} cldice {results[name]['cldice']:.4f} "
              f"iou {results[name]['iou']:.4f}", flush=True)
        torch.save(model.state_dict(), run_dir / "checkpoint" / f"{name}_adapted.pt")

        groups = modality_groups(list(results.keys()))
        payload = {"config": config, "per_domain": results,
                   "dice_summary": summarize({d: results[d]["dice"] for d in results}, groups),
                   "cldice_summary": summarize({d: results[d]["cldice"] for d in results}, groups),
                   "iou_summary": summarize({d: results[d]["iou"] for d in results}, groups)}

        with open(metric_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"[cbmt] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
