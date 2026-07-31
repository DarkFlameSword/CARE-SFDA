import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.domain_registry import TARGET_DOMAINS
from dataset.multi_domain import MultiDomainAdaptDataset
from dataset.splits import domain_split
from model.unet import UNet

DEFAULT_ANCHOR = "result/stage_a_source_driven_to_7targets/robust_2026-07-09/checkpoint/best_model.pt"
RUN_ROOT = Path("result/multi_target_adapt_7targets")
DEFAULT_RUNS = {
    0: "nativeF2_tau10_regret_residual_noalign_2026-07-24",
    1: "S1_regret_residual_tau10_noalign_s1_2026-07-24",
    2: "S1_regret_residual_tau10_noalign_s2_2026-07-24",
    3: "S1_regret_residual_tau10_noalign_s3_2026-07-24",
    4: "S1_regret_residual_tau10_noalign_s4_2026-07-24",
}
TARGETS = list(TARGET_DOMAINS)
EVAL_FRACTION = 0.30
N_CONTENT_IMAGES = 16
N_BOOTSTRAP = 20000


@torch.no_grad()
def content_descriptor(model, dataset, n_images, device):
    values = []

    for i in range(min(n_images, len(dataset))):
        item = dataset[i]
        image = item["image"].unsqueeze(0).to(device)
        fov = item["fov"].unsqueeze(0).to(device)
        prob = torch.sigmoid(model(image))
        values.append(float(((prob > 0.5).float() * fov).sum() / fov.sum().clamp_min(1.0)))

    return float(np.mean(values))


def in_sample_residual(y, x):
    design = np.column_stack([np.ones(len(y)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    return y - design @ coefficients


def lodo_residual(y, x):
    n = len(y)
    out_of_sample = np.zeros(n)

    for i in range(n):
        keep = np.arange(n) != i
        design = np.column_stack([np.ones(keep.sum()), x[keep]])
        coefficients, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        out_of_sample[i] = y[i] - np.concatenate([[1.0], np.atleast_1d(x[i])]) @ coefficients

    return out_of_sample


def ci95(values):
    values = np.asarray(values, dtype=np.float64)

    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="C-3 content-residual difficulty-ordering check.")
    parser.add_argument("--anchor", type=str, default=DEFAULT_ANCHOR)
    parser.add_argument("--runs", type=str, default=",".join(f"{s}:{r}" for s, r in DEFAULT_RUNS.items()),
                        help="Comma-separated seed:run_folder pairs under result/multi_target_adapt_7targets.")
    parser.add_argument("--n-content-images", type=int, default=N_CONTENT_IMAGES)
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    runs = {int(pair.split(":")[0]): pair.split(":")[1] for pair in args.runs.split(",") if pair.strip()}
    device = torch.device(args.device)
    run_dir = Path("result/mechanism_fairness_signal_7targets") / f"f0_lodo_{args.date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "C-3: Spearman(signal, realized Dice) for the content-adjusted residual, "
                      "in-sample vs leave-one-domain-out cross-validated",
        "claim": "s_d orders domains by realized difficulty (negative rank correlation with Dice); "
                 "the unadjusted surrogate does not",
        "anchor": args.anchor,
        "runs": runs,
        "run_family": "regret-residual noalign (deployed §3.10 signal), seeds 0-4, @2000",
        "targets": TARGETS,
        "content_descriptor": f"frozen-source mean in-FOV predicted foreground fraction over "
                              f"{args.n_content_images} adaptation-pool images per domain",
        "surrogate": "rho_d logged by the runner at iteration 2000 (EMA, teacher-student "
                     "disagreement + gamma1*L_inv + gamma2*L_occ)",
        "difficulty": "held-out per-domain Dice logged at iteration 2000",
        "bootstrap": f"resample the 5 seed-level LODO correlations, B={N_BOOTSTRAP}",
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    model = UNet(in_channels=1, n_classes=1, base_channels=64, norm="group", head_bias=False)
    model.load_state_dict(torch.load(args.anchor, map_location="cpu", weights_only=False)["model"])
    model = model.to(device).eval()
    per_seed = {}

    for seed, folder in runs.items():
        record = json.loads((RUN_ROOT / folder / "metric.json").read_text(encoding="utf-8"))["trajectory"][-1]
        pools = {t: domain_split(t, eval_fraction=EVAL_FRACTION, seed=seed)[0] for t in TARGETS}
        multi = MultiDomainAdaptDataset(pools, patch_size=512, n_views=2, intervention_strength=1.0)
        content = {t: content_descriptor(model, multi.datasets[t], args.n_content_images, device) for t in TARGETS}
        per_seed[seed] = {
            "rho_d": {t: float(record["rho_d"][t]) for t in TARGETS},
            "dice": {t: float(record["dice"][t]) for t in TARGETS},
            "content_c_d": content,
            "logged_hedge_signal": {t: float(record["hedge_signal"][t]) for t in TARGETS},
        }
        print(f"[seed {seed}] content descriptor done", flush=True)
        del multi

    signals = {}

    for seed, record in per_seed.items():
        rho = np.array([record["rho_d"][t] for t in TARGETS])
        dice = np.array([record["dice"][t] for t in TARGETS])
        content = np.array([record["content_c_d"][t] for t in TARGETS])[:, None]
        logged = np.array([record["logged_hedge_signal"][t] for t in TARGETS])
        signals[seed] = {
            "raw_surrogate_in_sample": float(spearmanr(rho, dice)[0]),
            "content_residual_in_sample": float(spearmanr(in_sample_residual(rho, content), dice)[0]),
            "content_residual_lodo": float(spearmanr(lodo_residual(rho, content), dice)[0]),
            "logged_hedge_signal_in_sample": float(spearmanr(logged, dice)[0]),
        }

    summary = {}

    for key in ["raw_surrogate_in_sample", "content_residual_in_sample", "content_residual_lodo",
                "logged_hedge_signal_in_sample"]:
        values = [signals[s][key] for s in signals]
        summary[key] = {
            "seed_mean": float(np.mean(values)),
            "ci95": ci95(values),
            "per_seed": values,
            "n_negative": int(np.sum(np.asarray(values) < 0)),
            "n_seeds": len(values),
        }

    generator = np.random.default_rng(0)
    lodo_values = np.asarray([signals[s]["content_residual_lodo"] for s in signals])
    draws = np.array([np.mean(generator.choice(lodo_values, size=len(lodo_values), replace=True))
                      for _ in range(N_BOOTSTRAP)])
    low, high = np.percentile(draws, [2.5, 97.5])
    bootstrap = {
        "statistic": "seed mean of the LODO Spearman for the content residual",
        "mean": float(lodo_values.mean()),
        "ci95_bootstrap": [float(low), float(high)],
        "p_mean_negative": float(np.mean(draws < 0)),
        "n_resamples": N_BOOTSTRAP,
    }

    with open(run_dir / "metric.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "per_seed_inputs": per_seed, "per_seed_spearman": signals,
                   "summary": summary, "bootstrap": bootstrap}, handle, indent=2, ensure_ascii=False)

    print("\n=== Spearman(signal, realized Dice); more negative = orders domains by difficulty ===", flush=True)

    for key, stats in summary.items():
        per_seed_text = ", ".join(f"{v:+.2f}" for v in stats["per_seed"])
        print(f"  {key:32s} {stats['seed_mean']:+.3f} +- {stats['ci95']:.3f}   "
              f"{stats['n_negative']}/{stats['n_seeds']} negative   [{per_seed_text}]", flush=True)

    print(f"\n  LODO bootstrap 95% CI [{low:+.3f}, {high:+.3f}]  P(mean<0)={bootstrap['p_mean_negative']:.2f}",
          flush=True)
    print(f"[fairness-signal] artifacts under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
