import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.domain_registry import index_domain
from dataset.harmonization import working_channel
from dataset.splits import domain_split
from dataset.tiling import score_native
from dataset.tiling import sliding_window_predict
from dataset.retinal_vessel_dataset import _augment_geometric
from dataset.retinal_vessel_dataset import _load_label
from dataset.retinal_vessel_dataset import _load_rgb
from dataset.retinal_vessel_dataset import _pad_to_size
from dataset.retinal_vessel_dataset import _random_fov_crop
from dataset.retinal_vessel_dataset import _resolve_fov
from dataset.retinal_vessel_dataset import _to_tensor
from loss.segmentation_loss import dice_bce_loss
from model.unet import UNet
from script.multitarget_results import modality_groups
from script.multitarget_results import summarize
from util.intensity_harmonization import invert_intensity

SOURCE_VAL_FRACTION = 0.15
TARGET_EVAL_FRACTION = 0.30
TILE = 512
OVERLAP = 0.25
THRESHOLD = 0.5


def vanilla_working_image(sample, invert):
    rgb = _load_rgb(sample.image_path)
    green = working_channel(rgb)
    fov = _resolve_fov(sample, green)

    if invert:
        green = invert_intensity(green)

    return green.astype(np.float32) * fov, fov


class VanillaPatchDataset(Dataset):
    def __init__(self, name, sample_ids, invert, patch=512, seed=0):
        allowed = set(sample_ids)
        self.samples = [s for s in index_domain(name, split="train", require_label=True)
                        if s.sample_id in allowed]
        self.invert = invert
        self.patch = patch
        self.seed = seed
        self._draw = 0

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed,
                                                          spawn_key=(idx, self._draw)))
        self._draw += 1
        sample = self.samples[idx]
        work, fov = vanilla_working_image(sample, self.invert)
        label = _load_label(sample.label_path).astype(np.float32)
        work, fov, label = _pad_to_size(work, fov, label, self.patch)
        work, fov, label = _random_fov_crop(work, fov, label, self.patch, rng)
        work, fov, label = _augment_geometric(work, fov, label, rng)

        return {"image": _to_tensor(work), "label": _to_tensor(label), "fov": _to_tensor(fov)}


def predict_probability(model, work, device):
    def predict_fn(tile):
        x = torch.from_numpy(np.ascontiguousarray(tile, dtype=np.float32))[None, None].to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x))

        return prob[0, 0].float().cpu().numpy()

    return sliding_window_predict(predict_fn, work, tile=TILE, overlap=OVERLAP)


def evaluate_domain(model, name, sample_ids, device):
    invert = DOMAINS[name].invert
    id_to_sample = {s.sample_id: s for s in index_domain(name, split="train", require_label=True)}
    model.eval()
    per_image = []

    for sid in sample_ids:
        sample = id_to_sample[sid]
        work, fov = vanilla_working_image(sample, invert)
        label_native = _load_label(sample.label_path)
        prob = predict_probability(model, work, device)
        per_image.append(score_native(prob, work.shape, label_native, fov, threshold=THRESHOLD))
        torch.cuda.empty_cache()

    return {k: float(np.mean([m[k] for m in per_image])) for k in sorted(per_image[0])}


def train_source(model, train_ids, val_ids, args, device, run_dir, config):
    dataset = VanillaPatchDataset(SOURCE_DOMAIN, train_ids, invert=DOMAINS[SOURCE_DOMAIN].invert,
                                  patch=args.patch, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_scaler = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    trajectory = []
    best = {"dice": -1.0, "iter": -1, "state": None}
    step = 0

    def run_eval(at_step):
        dice = evaluate_domain(model, SOURCE_DOMAIN, val_ids, device)["dice"]
        trajectory.append({"iter": at_step, "source_val_dice": dice})

        if dice > best["dice"]:
            best["dice"] = dice
            best["iter"] = at_step
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best["state"], "iter": at_step, "config": config,
                        "val_dice": dice}, run_dir / "checkpoint" / "best_model.pt")

        model.train()
        print(f"[vanilla] iter {at_step}: {SOURCE_DOMAIN}-val dice {dice:.4f} | "
              f"best@{best['iter']} {best['dice']:.4f}", flush=True)

    run_eval(0)

    with open(run_dir / "log" / "train_log.jsonl", "w", encoding="utf-8") as train_log:

        while step < args.iters:

            for batch in loader:

                if step >= args.iters:
                    break

                step += 1
                image = batch["image"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                fov = batch["fov"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=args.amp and device.type == "cuda"):
                    loss = dice_bce_loss(model(image), label, fov)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_log.write(json.dumps({"iter": step, "loss": float(loss.detach())}) + "\n")
                train_log.flush()

                if step % args.eval_interval == 0:
                    run_eval(step)

    return best, trajectory


def main():
    parser = argparse.ArgumentParser(description="B-1 strict-vanilla source-only lower bound.")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-root", type=str, default="result")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ids, val_ids = domain_split(SOURCE_DOMAIN, eval_fraction=SOURCE_VAL_FRACTION, seed=args.seed)
    eval_ids = {t: domain_split(t, eval_fraction=TARGET_EVAL_FRACTION, seed=args.seed)[1] for t in targets}

    run_dir = Path(args.result_root) / "baselines_7targets" / f"vanilla_unet_source_only_{args.date}"

    if run_dir.exists():
        run_dir = run_dir.parent / f"{run_dir.name}_{datetime.now().strftime('%H%M%S')}"

    (run_dir / "checkpoint").mkdir(parents=True)
    (run_dir / "log").mkdir()

    config = {
        "experiment": "B-1 strict-vanilla source-only lower bound on 7 targets",
        "regime": "green channel + JRCFA polarity inversion (FOV re-zeroed) + FOV zeroing only; "
                  "raw 0-255 intensities (no per-image z-score), native resolution (no FOV-caliber "
                  "resampling), flip/rot90 augmentation with no scale jitter",
        "controlled_only": {
            "architecture": "model.unet.UNet in_channels=1 n_classes=1 norm=group head_bias=False",
            "source": f"{SOURCE_DOMAIN} train split seed {args.seed} ({len(train_ids)} imgs, "
                      f"{len(val_ids)} source-val)",
            "splits": f"domain_split(t, {TARGET_EVAL_FRACTION}, {args.seed}) identical to CARE",
            "evaluation": f"CARE overlap-tiled native scoring (tile {TILE}, overlap {OVERLAP}, "
                          f"thr {THRESHOLD}, FOV-masked), full held-out subset",
            "selection": "best source-val Dice (label-free; no target label enters selection)",
        },
        "adaptation": "none (source-only lower bound)",
        "targets": targets,
        "n_eval": {t: len(eval_ids[t]) for t in targets},
        "iters": args.iters,
        "eval_interval": args.eval_interval,
        "batch_size": args.batch_size,
        "patch_size": args.patch,
        "base_channels": args.base_channels,
        "lr": args.lr,
        "optimizer": "adam",
        "loss": "dice_bce",
        "amp": args.amp,
        "amp_dtype": "fp16" if args.amp else "fp32",
        "seed": args.seed,
        "source_val_ids": val_ids,
        "target_eval_ids": eval_ids,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    print(f"[vanilla] run dir: {run_dir}", flush=True)
    print(f"[vanilla] {SOURCE_DOMAIN} train/val = {len(train_ids)}/{len(val_ids)}; "
          f"target eval = {[len(eval_ids[t]) for t in targets]}", flush=True)

    model = UNet(in_channels=1, n_classes=1, base_channels=args.base_channels, norm="group",
                 head_bias=False).to(device)
    best, trajectory = train_source(model, train_ids, val_ids, args, device, run_dir, config)
    model.load_state_dict(best["state"])
    print(f"[vanilla] scoring best checkpoint (iter {best['iter']}, source-val "
          f"{best['dice']:.4f}) on {len(targets)} targets", flush=True)

    per_domain = {}

    for name in targets:
        per_domain[name] = evaluate_domain(model, name, eval_ids[name], device)
        print(f"  {name:>9}: dice {per_domain[name]['dice']:.4f}  "
              f"cldice {per_domain[name]['cldice']:.4f}  iou {per_domain[name]['iou']:.4f}",
              flush=True)

    groups = modality_groups(targets)
    payload = {
        "config": config,
        "trajectory": trajectory,
        "best": {"iter": best["iter"], "source_val_dice": best["dice"]},
        "per_domain": per_domain,
        "summary_dice": summarize({t: per_domain[t]["dice"] for t in targets}, groups),
        "summary_cldice": summarize({t: per_domain[t]["cldice"] for t in targets}, groups),
    }

    with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    s = payload["summary_dice"]
    print(f"[vanilla] Dice mean7 {s['mean']:.4f}  group-mean {s['group_mean']:.4f}  "
          f"worst-mod {s['worst']:.4f}  spread {s['spread']:.4f}", flush=True)
    print(f"[vanilla] done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
