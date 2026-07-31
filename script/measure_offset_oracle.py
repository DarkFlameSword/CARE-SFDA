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
from dataset.tiling import sliding_window_predict
from dataset.tiling import upsample_to_native
from evaluation.evaluator import make_predict_fn
from model.unet import UNet
from util.evaluate_metrics import dice_coefficient

DEFAULT_ANCHOR = "result/stage_a_source_driven_to_7targets/robust_2026-07-09/checkpoint/best_model.pt"
DEFAULT_REGRET = "result/multi_target_adapt_7targets/nativeF2_tau10_regret_2026-07-14/student_final.pt"
EVAL_FRACTION = 0.30
GRID = np.round(np.linspace(0.02, 0.98, 49), 4)


def load_source_model(anchor, device):
    model = UNet(in_channels=1, n_classes=1, base_channels=64, norm="group", head_bias=False)
    model.load_state_dict(torch.load(anchor, map_location="cpu", weights_only=False)["model"])

    return model.to(device).eval()


def load_care_model(student_ckpt, device):
    ckpt = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    r = int(config.get("lora_rank", 8))
    alpha = float(config.get("lora_alpha", 16))
    model = UNet(in_channels=1, n_classes=1, base_channels=64, norm="group", head_bias=False)
    configure_peft(model, r=r, alpha=alpha, include_decoder=False, use_lora=True)
    model.load_state_dict(ckpt["student"])

    return model.to(device).eval(), int(config.get("seed", 0))


def sweep_domain(model, name, seed, device, tile, overlap, max_images, use_amp):
    eval_ids = domain_split(name, eval_fraction=EVAL_FRACTION, seed=seed)[1]
    dataset = EvalDataset(name, split="train", sample_ids=eval_ids)
    predict = make_predict_fn(model, device, use_amp)
    count = len(dataset) if max_images is None else min(len(dataset), max_images)
    rows = []

    for i in range(count):
        item = dataset[i]
        prob_scaled = sliding_window_predict(predict, item["image"][0].numpy(), tile=tile, overlap=overlap)
        prob_native = upsample_to_native(prob_scaled, item["native_shape"])
        label = item["label_native"]
        fov = item["fov_native"]
        rows.append([dice_coefficient(prob_native, label, roi=fov, threshold=float(t)) for t in GRID])

    return np.asarray(rows, dtype=np.float64)


def summarize_domain(dice_matrix):
    curve = dice_matrix.mean(axis=0)
    idx_half = int(np.argmin(np.abs(GRID - 0.5)))
    dice_at_half = float(curve[idx_half])
    idx_star = int(np.argmax(curve))
    oracle_dice = float(curve[idx_star])
    tau_star = float(GRID[idx_star])
    per_image_ceiling = float(dice_matrix.max(axis=1).mean())

    return {
        "dice_at_0.5": dice_at_half,
        "oracle_dice": oracle_dice,
        "oracle_threshold": tau_star,
        "delta_oracle": oracle_dice - dice_at_half,
        "per_image_ceiling": per_image_ceiling,
        "per_image_ceiling_gap": per_image_ceiling - dice_at_half,
        "curve": {float(t): float(v) for t, v in zip(GRID, curve)},
        "n_images": int(dice_matrix.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Offset-oracle threshold sweep (FCA redesign gate).")
    parser.add_argument("--anchor", type=str, default=DEFAULT_ANCHOR)
    parser.add_argument("--regret", type=str, default=DEFAULT_REGRET)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_DOMAINS))
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=None, help="Cap eval images per domain (None=all).")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    device = torch.device(args.device)
    run_dir = Path("result/mechanism_fca_offset_7targets") / f"oracle_sweep_{args.date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    care_model, seed = load_care_model(args.regret, device)
    source_model = load_source_model(args.anchor, device)
    models = {"care_regret": care_model, "source_only": source_model}

    config = {
        "experiment": "FCA offset-oracle probe: per-domain Dice headroom from thresholding w^T z at tau vs 0",
        "gate": "Delta_oracle(d) = max_tau Dice_d(prob>tau) - Dice_d(prob>0.5); single oracle threshold per domain",
        "models": {"care_regret": args.regret, "source_only": args.anchor},
        "seed": seed,
        "eval_fraction": EVAL_FRACTION,
        "threshold_grid": [float(t) for t in GRID],
        "tile": args.tile,
        "overlap": args.overlap,
        "max_images": args.max_images,
        "scoring": "CARE dice_coefficient on native GT inside FOV (identical to main table)",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    results = {}

    for model_name, model in models.items():
        print(f"\n=== {model_name} (seed {seed}) ===", flush=True)
        results[model_name] = {}
        deltas = []

        for name in targets:
            matrix = sweep_domain(model, name, seed, device, args.tile, args.overlap, args.max_images, args.amp)
            summary = summarize_domain(matrix)
            results[model_name][name] = summary
            deltas.append(summary["delta_oracle"])
            print(f"  {name:10s} Dice@0.5 {summary['dice_at_0.5']:.4f}  oracle {summary['oracle_dice']:.4f} "
                  f"@tau={summary['oracle_threshold']:.2f}  Delta_oracle {summary['delta_oracle']:+.4f}  "
                  f"(per-img ceiling +{summary['per_image_ceiling_gap']:.4f}, n={summary['n_images']})", flush=True)

        results[model_name]["_mean7"] = {
            "dice_at_0.5": float(np.mean([results[model_name][d]["dice_at_0.5"] for d in targets])),
            "oracle_dice": float(np.mean([results[model_name][d]["oracle_dice"] for d in targets])),
            "mean_delta_oracle": float(np.mean(deltas)),
        }
        m = results[model_name]["_mean7"]
        print(f"  {'MEAN7':10s} Dice@0.5 {m['dice_at_0.5']:.4f}  oracle {m['oracle_dice']:.4f}  "
              f"mean Delta_oracle {m['mean_delta_oracle']:+.4f}", flush=True)

        with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
            json.dump({"config": config, "results": results}, handle, indent=2, ensure_ascii=False)

    print(f"\n[offset-oracle] done. artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
