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
_AIF_ROOT = os.path.join(_CARE_ROOT, "model", "AIF-SFDA")
sys.path.insert(0, _CARE_ROOT)
sys.path.insert(0, _AIF_ROOT)

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.domain_registry import index_domain
from dataset.fov import derive_fov
from dataset.fov import load_fov_mask
from dataset.harmonization import harmonize
from dataset.harmonization import rescale_image
from dataset.harmonization import rescale_mask
from dataset.harmonization import working_channel
from dataset.retinal_vessel_dataset import _load_label
from dataset.retinal_vessel_dataset import _load_rgb
from dataset.retinal_vessel_dataset import _make_patch
from dataset.retinal_vessel_dataset import _prepare_image_side
from dataset.retinal_vessel_dataset import _to_tensor
from dataset.scale import compute_domain_scale
from dataset.splits import domain_split
from util.evaluate_metrics import evaluate as care_evaluate

from others.backbones.unet import Unet
from others.frequency_transforms import get_frequency_transform
from others.losses.mi_upper_boundary import CLUB
from others.losses.vector_distance import CosineSimilarity
from utils import multi_class_segmentation_reduce_dim

SOURCE_VAL_FRACTION = 0.15
TARGET_EVAL_FRACTION = 0.30


def build_unet():
    return Unet(3, 1, last_layer="Identity", output_mode=0, norm_layer=nn.InstanceNorm2d)


def _resolve_fov(sample, green):
    if sample.fov_path:

        return load_fov_mask(sample.fov_path, target_shape=green.shape)

    return derive_fov(green)


class AIFPatchDataset(Dataset):
    def __init__(self, name, sample_ids, scale, invert, patch=512, with_label=True, seed=0):
        self.spec = DOMAINS[name]
        allowed = set(sample_ids)
        self.samples = [s for s in index_domain(name, split="train", require_label=with_label)
                        if s.sample_id in allowed]
        self.scale = scale
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
        work, fov = _prepare_image_side(sample, self.spec, self.scale, invert=self.invert, norm="unit")

        if self.with_label:
            label = rescale_mask(_load_label(sample.label_path), self.scale)

        else:
            label = np.zeros_like(fov)

        work, fov, label = _make_patch(work, fov, label, self.patch, augment=True, rng=rng)
        image = _to_tensor(work).repeat(3, 1, 1)

        return {"image": image, "label": _to_tensor(label), "mask": _to_tensor(fov)}


def _infinite(loader):
    while True:

        for batch in loader:

            yield batch


def _linear_lr(optimizer, base_lr, step, total):
    half = total / 2.0
    lr = base_lr if step < half else base_lr * max(0.0, (total - step) / half)

    for group in optimizer.param_groups:
        group["lr"] = lr


def pretrain_source(source_ids, scale, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_unet().to(device)
    dataset = AIFPatchDataset(SOURCE_DOMAIN, source_ids, scale, invert=DOMAINS[SOURCE_DOMAIN].invert,
                              with_label=True, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    net.train()

    for step in range(iters):
        batch = next(loader)
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = net(image)[0]
        loss = criterion(logits, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _linear_lr(optimizer, 1e-3, step, iters)

        if (step + 1) % 500 == 0:
            print(f"[aif-pretrain] iter {step + 1}/{iters} bce {float(loss):.4f}", flush=True)

    return net.state_dict()


class AIFAdapt:
    def __init__(self, source_state, device, label_threshold=0.8, ema=0.9995,
                 alphas=(1.0, 0.5, 1.0, 1.0), smooth_sigma=1.0, smooth_kernel=5):
        self.device = device
        self.alphas = alphas
        self.label_threshold = label_threshold
        self.ema = ema
        self.net_filter = Unet(3, 1, depth=3, last_layer="Sigmoid", norm_layer=nn.InstanceNorm2d).to(device)
        self.net_student = build_unet().to(device)
        self.net_teacher = build_unet().to(device)
        self.net_student.load_state_dict(source_state)
        self.net_teacher.load_state_dict(source_state)

        for p in self.net_teacher.parameters():
            p.requires_grad_(False)

        self.club = CLUB(1024).to(device)
        self.dct = get_frequency_transform("dct")()
        self.criterion_ce = nn.BCEWithLogitsLoss()
        self.criterion_con = CosineSimilarity()
        kernel = torch.arange(smooth_kernel) - smooth_kernel // 2
        kernel = torch.exp(-kernel ** 2 / (2 * smooth_sigma ** 2))
        kernel = kernel / kernel.sum()
        self.smooth_kernel = (kernel[:, None] @ kernel[None, :]).unsqueeze(0).unsqueeze(0).to(device)
        self.opt_filter = torch.optim.Adam(self.net_filter.parameters(), lr=1e-3)
        self.opt_student = torch.optim.Adam(list(self.net_student.parameters()) + list(self.club.parameters()),
                                            lr=1e-3)

    def _filter_image(self, image, mask):
        freq = self.dct(image)
        filter_map = self.net_filter(self.dct.normalize_frequency_map(freq))
        filter_map = nn.functional.conv2d(filter_map, self.smooth_kernel,
                                          padding=self.smooth_kernel.shape[2] // 2)
        image_filtered = self.dct(freq * filter_map, inverse=True) * mask

        return image_filtered * image.mean() / image_filtered.mean()

    def _forward_train(self, image, mask):
        image_filtered = self._filter_image(image, mask)
        logits_student, *_, embedding_student = self.net_student(image_filtered)

        with torch.no_grad():
            logits_teacher, *_, embedding_teacher = self.net_teacher(image)
            pseudo_logits, pseudo_label = multi_class_segmentation_reduce_dim(logits_teacher)
            pseudo_logits = pseudo_logits.detach()
            pseudo_label = pseudo_label.detach().float()

        return logits_student, embedding_student, embedding_teacher, pseudo_logits, pseudo_label

    def _seg_loss(self, logits_student, pseudo_logits, pseudo_label):
        high_confident = pseudo_logits > self.label_threshold

        if high_confident.sum() == 0:

            return torch.zeros((), device=self.device)

        return self.criterion_ce(logits_student[high_confident], pseudo_label[high_confident])

    def step(self, image, mask):
        a0, a1, a2, a3 = self.alphas

        self.opt_filter.zero_grad(set_to_none=True)
        ls, es, et, pl_logits, pl = self._forward_train(image, mask)
        loss_seg = self._seg_loss(ls, pl_logits, pl)
        loss_mi = self.club(es, et)
        (a0 * loss_seg + a1 * loss_mi).backward()
        self.opt_filter.step()

        self.opt_student.zero_grad(set_to_none=True)
        ls, es, et, pl_logits, pl = self._forward_train(image, mask)
        loss_seg = self._seg_loss(ls, pl_logits, pl)
        loss_llh = -self.club.loglikelihood(es, et)
        loss_con = self.criterion_con(es, et)
        (loss_seg + a2 * loss_llh + a3 * loss_con).backward()
        self.opt_student.step()

        with torch.no_grad():
            for p_t, p_s in zip(self.net_teacher.parameters(), self.net_student.parameters()):
                p_t.mul_(self.ema).add_(p_s, alpha=1.0 - self.ema)

        return {"seg": float(loss_seg.detach()), "mi": float(loss_mi.detach()),
                "llh": float(loss_llh.detach()), "con": float(loss_con.detach())}

    @torch.no_grad()
    def predict(self, image, mask):
        self.net_filter.eval()
        self.net_student.eval()
        logits_student, *_ = self.net_student(self._filter_image(image, mask))

        return torch.sigmoid(logits_student)


def adapt_target(name, source_state, scale, iters, device, seed, patch=512):
    torch.manual_seed(seed)
    np.random.seed(seed)
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    invert = DOMAINS[name].invert
    dataset = AIFPatchDataset(name, adapt_ids, scale, invert=invert, patch=patch, with_label=False, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    model = AIFAdapt(source_state, device)
    model.net_filter.train()
    model.net_student.train()

    for step in range(iters):
        batch = next(loader)
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        losses = model.step(image, mask)
        _linear_lr(model.opt_filter, 1e-3, step, iters)
        _linear_lr(model.opt_student, 1e-3, step, iters)

        if (step + 1) % 200 == 0:
            print(f"[aif-adapt:{name}] iter {step + 1}/{iters} "
                  f"seg {losses['seg']:.3f} mi {losses['mi']:.3f} con {losses['con']:.3f}", flush=True)

    return model


def _pad_to_multiple(tensor, multiple=16):
    h, w = tensor.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple

    if ph or pw:
        tensor = nn.functional.pad(tensor, (0, pw, 0, ph), mode="reflect")

    return tensor, h, w


def evaluate_target(model, name, scale, device, seed):
    eval_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[1]
    invert = DOMAINS[name].invert
    id_to_sample = {s.sample_id: s for s in index_domain(name, split="train", require_label=True)}
    per_image = []

    for sid in eval_ids:
        sample = id_to_sample[sid]
        rgb = _load_rgb(sample.image_path)
        green = working_channel(rgb)
        fov_native = _resolve_fov(sample, green)
        label_native = _load_label(sample.label_path)
        work = harmonize(rgb, fov_native, invert, norm="unit")
        work_scaled = rescale_image(work, scale)
        fov_scaled = rescale_mask(fov_native, scale)
        image = _to_tensor(work_scaled).repeat(3, 1, 1).unsqueeze(0).to(device)
        mask = _to_tensor(fov_scaled.astype(np.float32)).unsqueeze(0).to(device)
        image, h, w = _pad_to_multiple(image)
        mask, _, _ = _pad_to_multiple(mask)
        prob = model.predict(image, mask)[0, 0, :h, :w].cpu().numpy()
        prob_native = sk_resize(prob, green.shape, order=1, preserve_range=True, anti_aliasing=False)
        metrics = care_evaluate(prob_native, label_native, roi=fov_native, threshold=0.5)
        per_image.append(metrics)

    keys = ["dice", "cldice", "iou", "sensitivity", "specificity"]

    return {k: float(np.mean([m[k] for m in per_image])) for k in keys}


def modality_groups(targets):
    groups = {}

    for domain in targets:
        groups.setdefault(DOMAINS[domain].modality, []).append(domain)

    return groups


def summarize(per_domain_dice, groups):
    group_dice = {m: float(np.mean([per_domain_dice[d] for d in members])) for m, members in groups.items()}
    values = list(group_dice.values())

    return {"mean": float(np.mean(list(per_domain_dice.values()))),
            "worst": float(min(values)), "spread": float(max(values) - min(values)),
            "group_dice": group_dice}


def main():
    parser = argparse.ArgumentParser(description="B-3 AIF-SFDA controlled baseline.")
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
    run_dir = Path("result/baselines_7targets") / f"aif_sfda_{args.date}"
    (run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)

    source_scale = compute_domain_scale(SOURCE_DOMAIN, 1024.0)
    scales = {t: compute_domain_scale(t, 1024.0) for t in targets}
    source_ids = domain_split(SOURCE_DOMAIN, eval_fraction=SOURCE_VAL_FRACTION, seed=args.seed)[0]

    config = {
        "experiment": "B-3 AIF-SFDA (AAAI 2025) controlled reproduction",
        "method": "official AIF-SFDA modules (DCT filter + CLUB MI + EMA consistency), reimplemented orchestration",
        "controlled": {"source": f"DRIVEN train split seed {args.seed} ({len(source_ids)} imgs)",
                       "splits": f"domain_split(t, {TARGET_EVAL_FRACTION}, {args.seed}) identical to CARE",
                       "preprocessing": "green + polarity(JRCFA) + FOV + scale-1024, unit[0,1], 3ch-replicated",
                       "evaluation": "CARE util.evaluate_metrics on native GT/FOV (dice/cldice/iou)",
                       "protocol": f"label-free, pretrain {args.pretrain_iters} / adapt {args.adapt_iters} iters, final ckpt"},
        "aif_intrinsic": {"arch": "InstanceNorm UNet depth5 + filter depth3", "input_nc": 3,
                          "optimizer": "Adam lr 1e-3 linear-decay", "label_threshold": 0.8, "ema": 0.9995,
                          "alphas": [1.0, 0.5, 1.0, 1.0], "filter_smooth": "sigma1 kernel5",
                          "batch_size": 2, "patch": args.patch, "adaptation": "per-target (single-target)"},
        "source_scale": source_scale,
        "target_scales": scales,
        "seed": args.seed,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    source_ckpt = run_dir / "checkpoint" / "source.pt"

    if source_ckpt.exists():
        print(f"[aif] loading existing source checkpoint {source_ckpt}", flush=True)
        source_state = torch.load(source_ckpt, map_location=device)

    else:
        print(f"[aif] pretraining source on DRIVEN ({len(source_ids)} imgs, {args.pretrain_iters} iters)", flush=True)
        source_state = pretrain_source(source_ids, source_scale, args.pretrain_iters, device, args.seed)
        torch.save(source_state, source_ckpt)

    metric_path = run_dir / "metric.json"
    results = json.load(open(metric_path, encoding="utf-8"))["per_domain"] if metric_path.exists() else {}

    for name in targets:

        if name in results:
            print(f"[aif] {name} already done, skipping", flush=True)
            continue

        print(f"[aif] adapting {name} (scale {scales[name]:.3f}, {args.adapt_iters} iters)", flush=True)
        model = adapt_target(name, source_state, scales[name], args.adapt_iters, device, args.seed, args.patch)
        results[name] = evaluate_target(model, name, scales[name], device, args.seed)
        print(f"[aif] {name}: dice {results[name]['dice']:.4f} cldice {results[name]['cldice']:.4f} "
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

    print(f"[aif] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
