import argparse
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_CARE_ROOT = os.path.dirname(_HERE)
_SFDA_ROOT = os.path.join(_CARE_ROOT, "model", "SFDA")
sys.path.insert(0, _CARE_ROOT)
sys.path.insert(0, _SFDA_ROOT)

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from networks import UNet as BatesonUNet
from losses import CrossEntropy
from losses import DiceLoss
from losses import EntKLProp
from utils import class2one_hot

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
N_CLASS = 2
WEIGHTS_SE = [0.1, 0.9]
LAMB_SE = 1.0
LAMB_PRIOR = 1.0
SOURCE_LR = 5e-4
WEIGHT_DECAY = 1e-4


def _native_green(sample, invert):
    rgb = _load_rgb(sample.image_path)
    green = working_channel(rgb)
    fov = _resolve_fov(sample, green)
    green = green.astype(np.float32)

    if invert:
        green = (255.0 - green) * fov.astype(np.float32)

    return green, fov


def _zscore_fov(green, fov):
    fov_bool = fov.astype(bool)
    vals = green[fov_bool]

    if vals.size == 0:

        return np.zeros_like(green, dtype=np.float32)

    mean = float(vals.mean())
    std = float(vals.std())
    norm = (green - mean) / (std if std > 1e-6 else 1.0)

    return (norm * fov_bool.astype(np.float32)).astype(np.float32)


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


class BatesonDataset(Dataset):
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
        green, fov = _native_green(sample, self.invert)
        norm = _zscore_fov(green, fov)
        size = (self.size, self.size)
        norm = sk_resize(norm, size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        label = _load_label(sample.label_path).astype(np.float32) if self.with_label else np.zeros(fov.shape, np.float32)
        label = (sk_resize(label, size, order=0, preserve_range=True, anti_aliasing=False) > 0.5).astype(np.float32)

        if self.augment:

            if rng.random() < 0.5:
                norm = norm[:, ::-1]
                label = label[:, ::-1]

            k = int(rng.integers(0, 4))

            if k:
                norm = np.rot90(norm, k)
                label = np.rot90(label, k)

        image = torch.from_numpy(np.ascontiguousarray(norm)[None])
        label_t = torch.from_numpy(np.ascontiguousarray(label)).long()

        return {"image": image, "label": label_t, "sample_id": sample.sample_id}


def compute_source_prevalence(source_ids, seed):
    id_to_sample = {s.sample_id: s for s in index_domain(SOURCE_DOMAIN, split="train", require_label=True)}
    fractions = []

    for sid in source_ids:
        sample = id_to_sample[sid]
        label = _load_label(sample.label_path).astype(np.float32)
        label = (sk_resize(label, (INPUT_SIZE, INPUT_SIZE), order=0, preserve_range=True,
                           anti_aliasing=False) > 0.5).astype(np.float32)
        fractions.append(float(label.mean()))

    return float(np.mean(fractions))


def pretrain_source(source_ids, iters, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = BatesonUNet(1, N_CLASS).to(device)
    net.train()
    dataset = BatesonDataset(SOURCE_DOMAIN, source_ids, invert=DOMAINS[SOURCE_DOMAIN].invert,
                             with_label=True, augment=True, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.Adam(net.parameters(), lr=SOURCE_LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    ce_fn = CrossEntropy(idc=[0, 1], weights=[1.0, 1.0], dtype="torch.float32")
    dice_fn = DiceLoss(idc=[1])

    for step in range(iters):
        batch = next(loader)
        image = batch["image"].to(device)
        target_oh = class2one_hot(batch["label"].to(device), N_CLASS).float()
        probs = F.softmax(net(image), dim=1)
        loss = ce_fn(probs, target_oh, None) + dice_fn(probs, target_oh, None)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"[bateson-pretrain] iter {step + 1}/{iters} loss {float(loss):.4f}", flush=True)

    return net.state_dict()


def _prior_bounds(batch_size, prevalence, device):
    wh = INPUT_SIZE * INPUT_SIZE
    bounds = torch.zeros(batch_size, 2, 1, dtype=torch.float32, device=device)
    bounds[:, 0, 0] = (1.0 - prevalence) * wh
    bounds[:, 1, 0] = prevalence * wh

    return bounds


def adapt_target(name, source_state, prevalence, iters, lr, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = BatesonUNet(1, N_CLASS).to(device)
    net.load_state_dict(source_state)
    net.train()
    adapt_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[0]
    dataset = BatesonDataset(name, adapt_ids, invert=DOMAINS[name].invert,
                             with_label=False, augment=True, seed=seed)
    loader = _infinite(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    ent_fn = EntKLProp(power=1, fn="norm_soft_size", curi=True, idc_c=[1], ivd=False,
                       weights_se=WEIGHTS_SE, lamb_se=LAMB_SE, lamb_conspred=1, lamb_consprior=LAMB_PRIOR,
                       dtype="torch.float32")

    for step in range(iters):
        batch = next(loader)
        image = batch["image"].to(device)
        probs = F.softmax(net(image), dim=1)
        bounds = _prior_bounds(image.shape[0], prevalence, device)
        loss_se, loss_prior, est_prop = ent_fn(probs, probs.detach(), bounds)
        loss = loss_se + loss_prior
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            fg = float(est_prop[:, 1].mean())
            print(f"[bateson-adapt:{name}] iter {step + 1}/{iters} ent {float(loss_se):.3f} "
                  f"prior {float(loss_prior):.3f} pred_fg {fg:.4f} (pi_S {prevalence:.4f})", flush=True)

    return net


def evaluate_target(net, name, device, seed):
    net.eval()
    eval_ids = domain_split(name, eval_fraction=TARGET_EVAL_FRACTION, seed=seed)[1]
    invert = DOMAINS[name].invert
    id_to_sample = {s.sample_id: s for s in index_domain(name, split="train", require_label=True)}
    per_image = []

    for sid in eval_ids:
        sample = id_to_sample[sid]
        green, fov = _native_green(sample, invert)
        norm = _zscore_fov(green, fov)
        label_native = _load_label(sample.label_path)
        native_shape = fov.shape
        norm_eval = sk_resize(norm, (INPUT_SIZE, INPUT_SIZE), order=1, preserve_range=True,
                              anti_aliasing=True).astype(np.float32)
        image = torch.from_numpy(np.ascontiguousarray(norm_eval)[None, None]).to(device)

        with torch.no_grad():
            prob = F.softmax(net(image), dim=1)[0, 1].cpu().numpy()

        prob_native = sk_resize(prob, native_shape, order=1, preserve_range=True, anti_aliasing=False)
        per_image.append(care_evaluate(prob_native, label_native, roi=fov, threshold=0.5))
        torch.cuda.empty_cache()

    keys = ["dice", "cldice", "iou", "sensitivity", "specificity"]

    return {k: float(np.mean([m[k] for m in per_image])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="B-7 Bateson SFDA (entropy + class-ratio prior) baseline.")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--pretrain-iters", type=int, default=6000)
    parser.add_argument("--sfda-iters", type=int, default=2000)
    parser.add_argument("--sfda-lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/baselines_7targets") / f"sfda_bateson_native_{args.date}"
    (run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    source_ids = domain_split(SOURCE_DOMAIN, eval_fraction=SOURCE_VAL_FRACTION, seed=args.seed)[0]
    prevalence = compute_source_prevalence(source_ids, args.seed)

    config = {
        "experiment": "B-7 Bateson SFDA (Source-Relaxed/Source-Free DA, MICCAI 2020 + MedIA 2022) native reproduction",
        "method": "prior-guided SFDA: minimize weighted self-entropy + class-ratio prior KL "
                  "(EntKLProp, lamb_se=1, lamb_consprior=1, weights_se=[0.1,0.9], curi=True, power=1); "
                  "no source data, no pseudo-labels (verbatim model/SFDA networks.UNet + losses.EntKLProp)",
        "class_ratio_prior": {"pi_S_wholeimage_512": prevalence,
                              "source": "DRIVEN source-label whole-image (512^2) foreground fraction; "
                                        "label-free (== Bateson dumbpredwtags estimate, NOT the val_gt_size oracle)",
                              "note": "single source prior applied to all targets (this family's known fragility); "
                                      "whole-image norm_soft_size carries a bounded FOV-coverage confound"},
        "regime": "native: green channel -> polarity flip (JRCFA) + re-zero FOV -> z-score over FOV -> resize "
                  "whole to 512x512; networks.UNet(1,2) BN, nG=32, trained FROM SCRATCH; 2-class softmax",
        "controlled_only": {"source": f"DRIVEN train split seed {args.seed} ({len(source_ids)} imgs)",
                            "splits": f"domain_split(t, {TARGET_EVAL_FRACTION}, {args.seed}) identical to CARE",
                            "evaluation": "CARE util.evaluate_metrics on native GT/FOV (dice/cldice/iou)",
                            "label_free": "fixed adapt budget, final checkpoint, NO target-label selection",
                            "shared_physical_necessity": "JRCFA polarity flip + re-zero FOV only"},
        "learning_rates": {"source_pretrain": f"Adam {SOURCE_LR} (Bateson source recipe), wd {WEIGHT_DECAY}",
                           "adaptation": f"Adam {args.sfda_lr} (Bateson native 1e-6; see run log for budget)"},
        "protocol": f"pretrain {args.pretrain_iters} / SFDA {args.sfda_iters} iters, batch 2",
        "seed": args.seed,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    source_ckpt = run_dir / "checkpoint" / "source.pt"

    if source_ckpt.exists():
        print(f"[bateson] loading existing source checkpoint {source_ckpt}", flush=True)
        source_state = torch.load(source_ckpt, map_location=device)

    else:
        print(f"[bateson] pretraining source on DRIVEN ({len(source_ids)} imgs, {args.pretrain_iters} iters, "
              f"pi_S={prevalence:.4f})", flush=True)
        source_state = pretrain_source(source_ids, args.pretrain_iters, device, args.seed)
        torch.save(source_state, source_ckpt)

    metric_path = run_dir / "metric.json"
    results = json.load(open(metric_path, encoding="utf-8"))["per_domain"] if metric_path.exists() else {}

    for name in targets:

        if name in results:
            print(f"[bateson] {name} already done, skipping", flush=True)
            continue

        print(f"[bateson] adapting {name} ({args.sfda_iters} iters, lr {args.sfda_lr}, invert={DOMAINS[name].invert})", flush=True)
        net = adapt_target(name, source_state, prevalence, args.sfda_iters, args.sfda_lr, device, args.seed)
        results[name] = evaluate_target(net, name, device, args.seed)
        print(f"[bateson] {name}: dice {results[name]['dice']:.4f} cldice {results[name]['cldice']:.4f} "
              f"iou {results[name]['iou']:.4f}", flush=True)
        torch.save(net.state_dict(), run_dir / "checkpoint" / f"{name}_adapted.pt")
        del net
        torch.cuda.empty_cache()

        groups = modality_groups(list(results.keys()))
        payload = {"config": config, "per_domain": results,
                   "dice_summary": summarize({d: results[d]["dice"] for d in results}, groups),
                   "cldice_summary": summarize({d: results[d]["cldice"] for d in results}, groups),
                   "iou_summary": summarize({d: results[d]["iou"] for d in results}, groups)}

        with open(metric_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"[bateson] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
