import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.source_only import SourceOnlyTrainer
from dataset.domain_registry import DOMAINS
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.retinal_vessel_dataset import EvalDataset
from dataset.retinal_vessel_dataset import fdh_dataset_kwargs
from dataset.splits import domain_split
from evaluation.evaluator import Evaluator
from model.unet import UNet

THEME = f"stage_a_source_{SOURCE_DOMAIN.lower()}_to_7targets"


def parse_args():
    parser = argparse.ArgumentParser(description="CARE-SFDA Stage-A source-anchor trainer.")
    parser.add_argument("--arm", choices=["vanilla", "robust"], required=True,
                        help="vanilla = geometric base only; robust = geometric + §3.3 photometric.")
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.30, help="Per-target held-out eval fraction.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Source-val fraction of the source train split.")
    parser.add_argument("--max-eval-images", type=int, default=25)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--fdh-norm", choices=["zscore", "unit"], default="zscore",
                        help="Intensity normalization axis of the deterministic alignment.")
    parser.add_argument("--fdh-channel", choices=["green", "luminance"], default="green",
                        help="Working-channel axis of the deterministic alignment.")
    parser.add_argument("--fdh-caliber", action=argparse.BooleanOptionalAction, default=True,
                        help="Resample each domain to the common FOV caliber.")
    parser.add_argument("--fdh-polarity", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the per-domain polarity sign (the FA inversion).")
    parser.add_argument("--note", type=str, default=None,
                        help="Descriptive suffix for the run folder, e.g. the FDH axis being ablated.")
    parser.add_argument("--result-root", type=str, default="result")
    parser.add_argument("--theme", type=str, default=THEME)
    return parser.parse_args()


def build_run_dir(result_root, theme, arm, note=None):
    date = datetime.now().strftime("%Y-%m-%d")
    run_name = f"{arm}_{note}_{date}" if note else f"{arm}_{date}"
    run_dir = Path(result_root) / theme / run_name

    if run_dir.exists():
        run_name = f"{run_name}_{datetime.now().strftime('%H%M%S')}"
        run_dir = Path(result_root) / theme / run_name

    (run_dir / "visualization").mkdir(parents=True)
    (run_dir / "log").mkdir()
    (run_dir / "checkpoint").mkdir()

    return run_dir


def build_splits(targets, val_fraction, eval_fraction, seed):
    train_ids, val_ids = domain_split(SOURCE_DOMAIN, eval_fraction=val_fraction, seed=seed)
    target_eval = {}

    for d in targets:
        _, eval_ids = domain_split(d, eval_fraction=eval_fraction, seed=seed)
        target_eval[d] = eval_ids

    return train_ids, val_ids, target_eval


def write_config(run_dir, args, targets, train_ids, val_ids, target_eval):
    config = {
        "method": "source_anchor",
        "arm": args.arm,
        "source_domain": SOURCE_DOMAIN,
        "target_domains": targets,
        "normalization": "zscore_fov",
        "head_bias": False,
        "geometric_base": True,
        "photometric": args.arm == "robust",
        "iters": args.iters,
        "eval_interval": args.eval_interval,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "fdh": {"norm": args.fdh_norm, "channel": args.fdh_channel,
                "caliber": args.fdh_caliber, "polarity": args.fdh_polarity},
        "base_channels": args.base_channels,
        "norm": "group",
        "in_channels": 1,
        "n_classes": 1,
        "lr": args.lr,
        "optimizer": "adam",
        "loss": "dice_bce",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "eval_fraction": args.eval_fraction,
        "max_eval_images": args.max_eval_images,
        "num_workers": args.num_workers,
        "device": args.device,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype,
        "n_source_train": len(train_ids),
        "n_source_val": len(val_ids),
        "target_eval_counts": {d: len(ids) for d, ids in target_eval.items()},
        "source_val_ids": val_ids,
        "target_eval_ids": target_eval,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return config


def build_eval_datasets(targets, val_ids, target_eval, fdh=None):
    fdh = dict(fdh or {})
    datasets = {"source_val": EvalDataset(SOURCE_DOMAIN, split="train", sample_ids=val_ids, **fdh)}

    for d in targets:
        datasets[d] = EvalDataset(d, split="train", sample_ids=target_eval[d], **fdh)

    return datasets


def plot_curves(records, targets, run_dir):
    iters = [r["iter"] for r in records]
    figure, axis = plt.subplots(figsize=(9, 6))

    for d in targets:
        axis.plot(iters, [r["targets"][d]["dice"] for r in records], marker="o", label=d)

    axis.plot(iters, [r["source_val"]["dice"] for r in records], marker="s",
              linewidth=2.5, color="black", linestyle="--", label=f"{SOURCE_DOMAIN} (source-val)")
    axis.set_xlabel("iteration")
    axis.set_ylabel("Dice")
    axis.set_title("Stage-A source anchor: per-target pseudo-label Dice vs iterations")
    axis.legend(loc="best", fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(run_dir / "visualization" / "pseudolabel_dice_curves.png", dpi=150)
    plt.close(figure)


def polarity_ablation(model, targets, target_eval, args):
    correct = {d: EvalDataset(d, split="train", sample_ids=target_eval[d]) for d in targets}
    flipped = {d: EvalDataset(d, split="train", sample_ids=target_eval[d],
                              invert_override=(not DOMAINS[d].invert)) for d in targets}
    ev_correct = Evaluator(SOURCE_DOMAIN, targets, device=args.device, tile=args.patch_size,
                           max_images=args.max_eval_images, use_amp=False, datasets=correct)
    ev_flipped = Evaluator(SOURCE_DOMAIN, targets, device=args.device, tile=args.patch_size,
                           max_images=args.max_eval_images, use_amp=False, datasets=flipped)
    res_correct = ev_correct.evaluate(model)
    res_flipped = ev_flipped.evaluate(model)
    ablation = {}

    for d in targets:
        ablation[d] = {
            "invert_default": DOMAINS[d].invert,
            "dice_correct": res_correct[d]["dice"],
            "dice_flipped": res_flipped[d]["dice"],
            "gap": res_correct[d]["dice"] - res_flipped[d]["dice"],
        }

    return ablation


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    targets = [d.strip() for d in args.targets.split(",") if d.strip()]
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    train_ids, val_ids, target_eval = build_splits(targets, args.val_fraction, args.eval_fraction, args.seed)
    run_dir = build_run_dir(args.result_root, args.theme, args.arm, note=args.note)
    config = write_config(run_dir, args, targets, train_ids, val_ids, target_eval)
    print(f"[stage_a:{args.arm}] run dir: {run_dir}")
    print(f"[stage_a:{args.arm}] {SOURCE_DOMAIN} train/val = {len(train_ids)}/{len(val_ids)}; "
          f"target eval = {[len(v) for v in target_eval.values()]}")

    model = UNet(in_channels=1, n_classes=1, base_channels=args.base_channels, norm="group", head_bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fdh = fdh_dataset_kwargs(norm=args.fdh_norm, channel=args.fdh_channel,
                             caliber=args.fdh_caliber, polarity=args.fdh_polarity)
    trainer = SourceOnlyTrainer(model, optimizer, source_domain=SOURCE_DOMAIN,
                                patch_size=args.patch_size, batch_size=args.batch_size,
                                device=args.device, use_amp=args.amp, amp_dtype=amp_dtype,
                                num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                grad_clip=args.grad_clip, augment=True,
                                photometric=(args.arm == "robust"), train_ids=train_ids, fdh=fdh)

    eval_datasets = build_eval_datasets(targets, val_ids, target_eval, fdh=fdh)
    evaluator = Evaluator(SOURCE_DOMAIN, targets, device=args.device, tile=args.patch_size,
                          max_images=args.max_eval_images, use_amp=False, datasets=eval_datasets)

    records = []
    best = {"val_dice": -1.0, "iter": -1}

    def run_eval(step):
        results = evaluator.evaluate(model)
        record = {"iter": step, "source_val": results["source_val"],
                  "targets": {d: results[d] for d in targets}}
        record["target_mean_dice"] = float(np.mean([results[d]["dice"] for d in targets]))
        record["target_worst_dice"] = float(min(results[d]["dice"] for d in targets))
        records.append(record)

        with open(run_dir / "metric.json", "w") as handle:
            json.dump({"config": config, "trajectory": records}, handle, indent=2)

        torch.save({"model": model.state_dict(), "iter": step, "config": config},
                   run_dir / "checkpoint" / "latest.pt")

        if record["source_val"]["dice"] > best["val_dice"]:
            best["val_dice"] = record["source_val"]["dice"]
            best["iter"] = step
            torch.save({"model": model.state_dict(), "iter": step, "config": config,
                        "val_dice": best["val_dice"]}, run_dir / "checkpoint" / "best_model.pt")

        print(f"[stage_a:{args.arm}] iter {step}: {SOURCE_DOMAIN}-val {record['source_val']['dice']:.4f} | "
              f"target mean {record['target_mean_dice']:.4f} | worst {record['target_worst_dice']:.4f} | "
              f"best-val@{best['iter']} {best['val_dice']:.4f}")
        return record

    run_eval(0)

    with open(run_dir / "log" / "train_log.jsonl", "w") as train_log:

        for step in range(1, args.iters + 1):
            log = trainer.step()
            train_log.write(json.dumps(log) + "\n")
            train_log.flush()

            if step % args.eval_interval == 0:
                run_eval(step)

    plot_curves(records, targets, run_dir)

    best_ckpt = torch.load(run_dir / "checkpoint" / "best_model.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    print(f"[stage_a:{args.arm}] polarity ablation (A-iv) on best model @ iter {best_ckpt['iter']} ...")
    ablation = polarity_ablation(model, targets, target_eval, args)

    for d in targets:
        a = ablation[d]
        print(f"  {d:10s} correct {a['dice_correct']:.4f} | flipped {a['dice_flipped']:.4f} | "
              f"gap {a['gap']:+.4f} (default invert={a['invert_default']})")

    payload = {"config": config, "trajectory": records,
               "best": {"iter": best["iter"], "val_dice": best["val_dice"]},
               "polarity_ablation": ablation}

    with open(run_dir / "metric.json", "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[stage_a:{args.arm}] done. best val-Dice {best['val_dice']:.4f} @ iter {best['iter']}; "
          f"artifacts under {run_dir}")


if __name__ == "__main__":
    main()
