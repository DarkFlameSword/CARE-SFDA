import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptation.peft import configure_peft
from dataset.domain_registry import SOURCE_DOMAIN
from dataset.domain_registry import TARGET_DOMAINS
from dataset.retinal_vessel_dataset import EvalDataset
from dataset.splits import domain_split
from evaluation.evaluator import Evaluator
from model.unet import UNet
from script.multitarget_results import modality_groups
from script.multitarget_results import summarize

DEFAULT_CKPT = "result/stage_a_source_driven_to_7targets/robust_2026-07-09/checkpoint/best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Full-eval a checkpoint on the 7 target held-out subsets.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--note", type=str, default="source_only", help="Run-folder note tag.")
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--eval-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0, help="Split seed (must match the adaptation runs' seed).")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-root", type=str, default="result")
    parser.add_argument("--theme", type=str, default="baselines_7targets")

    return parser.parse_args()


def main():
    args = parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    eval_ids = {t: domain_split(t, eval_fraction=args.eval_fraction, seed=args.seed)[1] for t in targets}

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = UNet(in_channels=1, n_classes=1, base_channels=args.base_channels, norm="group", head_bias=False)

    if "student" in checkpoint:
        peft_config = checkpoint.get("config", {})
        configure_peft(model, r=peft_config.get("lora_rank", 8), alpha=peft_config.get("lora_alpha", 16.0),
                       include_decoder=bool(peft_config.get("decoder_lora", False)))
        model.load_state_dict(checkpoint["student"])

    else:
        model.load_state_dict(checkpoint["model"])

    model.to(args.device)

    eval_datasets = {t: EvalDataset(t, split="train", sample_ids=eval_ids[t]) for t in targets}
    evaluator = Evaluator(SOURCE_DOMAIN, targets, device=args.device, tile=args.patch_size,
                          use_amp=False, max_images=None, datasets=eval_datasets)
    results = evaluator.evaluate(model)

    per_dice = {t: results[t]["dice"] for t in targets}
    per_cldice = {t: results[t]["cldice"] for t in targets}
    groups = modality_groups(targets)

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(args.result_root) / args.theme / f"{args.note}_{date}"

    if run_dir.exists():
        run_dir = run_dir.parent / f"{args.note}_{date}_{datetime.now().strftime('%H%M%S')}"

    run_dir.mkdir(parents=True)
    config = {
        "experiment": "source-only / checkpoint full-eval reference on 7 targets",
        "checkpoint": args.checkpoint,
        "targets": targets,
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
        "n_eval": {t: len(eval_ids[t]) for t in targets},
        "eval": "full held-out subset (max_images=None), tile 512 overlap 0.25 thr 0.5, FOV-masked",
        "modality_groups": groups,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    payload = {
        "config": config,
        "per_domain": {t: results[t] for t in targets},
        "summary_dice": summarize(per_dice, groups),
        "summary_cldice": summarize(per_cldice, groups),
    }

    with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"[eval:{args.note}] full-eval on {len(targets)} targets, checkpoint {args.checkpoint}")

    for t in targets:
        print(f"  {t:>9}: dice {per_dice[t]:.4f}  cldice {per_cldice[t]:.4f}  "
              f"(n_eval {len(eval_ids[t])})")

    s = payload["summary_dice"]
    print(f"  Dice mean7 {s['mean']:.4f}  group-mean {s['group_mean']:.4f}  "
          f"worst-mod {s['worst']:.4f}  spread {s['spread']:.4f}")
    print(f"[eval:{args.note}] done -> {run_dir}")


if __name__ == "__main__":
    main()
