import argparse
import json
import os
import sys
from datetime import datetime
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.tent import configure_tent
from baseline.tent import entropy_loss
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.retinal_vessel_dataset import EvalDataset
from dataset.retinal_vessel_dataset import TargetTrainDataset
from dataset.splits import domain_split
from evaluation.evaluator import Evaluator
from model.unet import UNet
from script.multitarget_results import modality_groups
from script.multitarget_results import summarize

DEFAULT_ANCHOR = "result/stage_a_source_driven_to_7targets/robust_2026-07-09/checkpoint/best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="B-2 per-domain TENT entropy-min baseline.")
    parser.add_argument("--anchor", type=str, default=DEFAULT_ANCHOR)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--steps", type=int, default=200, help="Entropy-min steps per domain.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--eval-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-only", type=str,
                        default="result/baselines_7targets/source_only_2026-07-16/metric.json",
                        help="Source-only reference metric.json for the delta column (optional).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-root", type=str, default="result")
    parser.add_argument("--theme", type=str, default="baselines_7targets")
    parser.add_argument("--note", type=str, default="tent")

    return parser.parse_args()


def occupancy(model, dataset, device, n=4):
    values = []

    for i in range(min(n, len(dataset))):
        item = dataset[i]
        image = item["image"].unsqueeze(0).to(device)
        fov = item["fov"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = (torch.sigmoid(model(image)) > 0.5).float()

        values.append(float((pred * fov).sum() / fov.sum().clamp_min(1.0)))

    return float(np.mean(values))


def adapt_domain(domain, args):
    device = args.device
    adapt_ids, eval_ids = domain_split(domain, eval_fraction=args.eval_fraction, seed=args.seed)

    model = UNet(in_channels=1, n_classes=1, base_channels=args.base_channels, norm="group", head_bias=False)
    model.load_state_dict(torch.load(args.anchor, map_location="cpu", weights_only=False)["model"])
    model.to(device)
    gn_params = configure_tent(model)
    optimizer = torch.optim.Adam(gn_params, lr=args.lr)

    adapt_ds = TargetTrainDataset(domain, split="train", patch_size=args.patch_size, augment=False,
                                  sample_ids=adapt_ids)
    loader = DataLoader(adapt_ds, batch_size=min(args.batch_size, len(adapt_ds)), shuffle=True,
                        num_workers=4, drop_last=len(adapt_ds) >= args.batch_size)
    stream = cycle(loader)

    occ_before = occupancy(model, adapt_ds, device)
    model.train()

    for _ in range(args.steps):
        batch = next(stream)
        image = batch["image"].to(device)
        fov = batch["fov"].to(device)
        loss = entropy_loss(model(image), fov)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    occ_after = occupancy(model, adapt_ds, device)

    eval_ds = {domain: EvalDataset(domain, split="train", sample_ids=eval_ids)}
    evaluator = Evaluator(SOURCE_DOMAIN, [domain], device=device, tile=args.patch_size,
                          use_amp=False, max_images=None, datasets=eval_ds)
    result = evaluator.evaluate(model)[domain]

    return {**result,
            "occ_before": occ_before, "occ_after": occ_after,
            "n_gn_params": sum(p.numel() for p in gn_params), "n_adapt": len(adapt_ids),
            "n_eval": len(eval_ids)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    source_dice = {}

    if args.source_only and Path(args.source_only).exists():
        ref = json.load(open(args.source_only, encoding="utf-8"))
        source_dice = {t: ref["per_domain"][t]["dice"] for t in targets if t in ref["per_domain"]}

    per_domain = {}

    for domain in targets:
        per_domain[domain] = adapt_domain(domain, args)
        r = per_domain[domain]
        delta = f"{r['dice'] - source_dice[domain]:+.4f}" if domain in source_dice else "n/a"
        print(f"  {domain:>9}: dice {r['dice']:.4f} (src {delta})  cldice {r['cldice']:.4f}  "
              f"occ {r['occ_before']:.3f}->{r['occ_after']:.3f}")

    groups = modality_groups(targets)
    dice = {t: per_domain[t]["dice"] for t in targets}
    cldice = {t: per_domain[t]["cldice"] for t in targets}

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(args.result_root) / args.theme / f"{args.note}_{date}"

    if run_dir.exists():
        run_dir = run_dir.parent / f"{args.note}_{date}_{datetime.now().strftime('%H%M%S')}"

    run_dir.mkdir(parents=True)
    config = {
        "experiment": "B-2 TENT entropy-minimization baseline (per-domain, GroupNorm affine)",
        "method": "TENT (Wang et al. 2021), GroupNorm variant: entropy-min on GN affine only; "
                  "BN-statistic update is a no-op for GroupNorm",
        "anchor": args.anchor,
        "targets": targets,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
        "transductive": "adapt on unlabeled adapt-pool, eval on held-out split, fixed budget (no selection)",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    payload = {
        "config": config,
        "per_domain": per_domain,
        "summary_dice": summarize(dice, groups),
        "summary_cldice": summarize(cldice, groups),
        "source_only_dice": source_dice,
    }

    with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    s = payload["summary_dice"]
    mean_delta = (np.mean([dice[t] - source_dice[t] for t in targets if t in source_dice])
                  if source_dice else float("nan"))
    print(f"\nTENT Dice mean7 {s['mean']:.4f} worst-mod {s['worst']:.4f} spread {s['spread']:.4f} "
          f"| mean delta vs source-only {mean_delta:+.4f}")
    print(f"[B-2] done -> {run_dir}")


if __name__ == "__main__":
    main()
