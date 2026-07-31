import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.domain_registry import TARGET_DOMAINS

RUN_ROOT = Path("result/multi_target_adapt_7targets")
STARVED = "IOSTAR"

PAIRS = {
    0: {"nostarve": "nativeF2_tau10_regret_residual_noalign_2026-07-24",
        "starve": "C4_regret_residual_starve_IOSTAR_tau10_noalign_s0_2026-07-24"},
    1: {"nostarve": "S1_regret_residual_tau10_noalign_s1_2026-07-24",
        "starve": "C4_regret_residual_starve_IOSTAR_tau10_noalign_s1_2026-07-24"},
}


def load_run(run, seed, is_starve):
    with open(RUN_ROOT / run / "metric.json", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = payload["config"]
    guards = {"sampling_temperature": 1.0, "use_sep": False, "seed": seed,
              "hedge_beta": 0.01, "fairness": "regret_residual"}
    guards["starve_domain"] = STARVED if is_starve else None

    for key, expected in guards.items():

        if config.get(key) != expected:
            raise ValueError(f"{run}: {key}={config.get(key)} != expected {expected}")

    if is_starve and abs(float(config.get("starve_factor")) - 0.33) > 1e-9:
        raise ValueError(f"{run}: starve_factor {config.get('starve_factor')} != 0.33")

    return config, payload["trajectory"]


def main():
    targets = list(TARGET_DOMAINS)
    per_seed = {}

    for seed, pair in PAIRS.items():
        _, traj_no = load_run(pair["nostarve"], seed, is_starve=False)
        _, traj_st = load_run(pair["starve"], seed, is_starve=True)
        lam_no = traj_no[-1]["lambda"]
        lam_st = traj_st[-1]["lambda"]
        dice_no = traj_no[-1]["dice"]
        dice_st = traj_st[-1]["dice"]
        per_seed[seed] = {
            "lambda_nostarve": lam_no,
            "lambda_starve": lam_st,
            "delta_lambda": {t: lam_st[t] - lam_no[t] for t in targets},
            "dice_iostar_nostarve": dice_no[STARVED],
            "dice_iostar_starve": dice_st[STARVED],
            "traj_nostarve": [{"iter": r["iter"], "lam_iostar": r["lambda"][STARVED]} for r in traj_no],
            "traj_starve": [{"iter": r["iter"], "lam_iostar": r["lambda"][STARVED]} for r in traj_st],
        }

    seeds = sorted(per_seed)
    delta_iostar = [per_seed[s]["delta_lambda"][STARVED] for s in seeds]
    delta_mean = {t: float(np.mean([per_seed[s]["delta_lambda"][t] for s in seeds])) for t in targets}

    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path("result/mechanism_fairness_starve_7targets") / f"underservice_starve_{date}"
    (run_dir / "visualization").mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "C-4 under-service correction: matched starve-vs-no-starve lambda shift (assembly)",
        "starved_domain": STARVED,
        "starve_factor": 0.33,
        "pairs": {str(s): PAIRS[s] for s in seeds},
        "base": "tau=1, regret, per-domain unit, use_sep=false, hedge_beta=0.01 (declared)",
        "claim": "Delta lambda_IOSTAR(starve - no-starve) > 0; pure lambda-allocation, NO Dice-rescue claim (§3.12)",
        "targets": targets,
        "code_commit_hash": "None",
        "launch_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    result = {
        "delta_lambda_iostar_per_seed": {str(s): delta_iostar[i] for i, s in enumerate(seeds)},
        "delta_lambda_iostar_mean": float(np.mean(delta_iostar)),
        "delta_lambda_mean_all_domains": delta_mean,
        "iostar_dice_diagnostic": {str(s): {"nostarve": per_seed[s]["dice_iostar_nostarve"],
                                            "starve": per_seed[s]["dice_iostar_starve"]} for s in seeds},
        "per_seed": per_seed,
    }

    with open(run_dir / "underservice_starve.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "result": result}, handle, indent=2, ensure_ascii=False)

    lines = ["# C-4 under-service correction (matched starve vs no-starve, regret game)", "",
             f"Starved domain = **{STARVED}** (sampling x0.33). Matched pairs at 2 seeds, identical base",
             "(tau=1, regret, use_sep=false, hedge_beta=0.01). Claim = the lambda the game allocates,",
             "measured as Delta lambda = lambda(starve) - lambda(no-starve). **No Dice-rescue claim.**", "",
             f"## Delta lambda_{STARVED}(starve - no-starve)", "",
             "| seed | lambda no-starve | lambda starve | Delta lambda |",
             "|---|---|---|---|"]

    for i, s in enumerate(seeds):
        ln = per_seed[s]["lambda_nostarve"][STARVED]
        ls = per_seed[s]["lambda_starve"][STARVED]
        lines.append(f"| {s} | {ln:.3f} | {ls:.3f} | {ls - ln:+.3f} |")

    lines += [f"| **mean** | | | **{float(np.mean(delta_iostar)):+.3f}** |", "",
              f"Starving {STARVED} raises its regret-game weight by a mean **{float(np.mean(delta_iostar)):+.3f}** "
              "(both seeds same sign)" if all(d > 0 for d in delta_iostar) else
              f"Mean Delta lambda_{STARVED} = {float(np.mean(delta_iostar)):+.3f} (seeds disagree in sign)",
              "-- i.e. the game re-allocates loop weight toward the manufactured under-served domain.", "",
              "## Full mean Delta lambda across domains (seed-averaged)", "",
              "| domain | mean Delta lambda |", "|---|---|"]

    for t in targets:
        lines.append(f"| {t}{' (starved)' if t == STARVED else ''} | {delta_mean[t]:+.3f} |")

    lines += ["", "## IOSTAR Dice diagnostic (single-seed, NOT a claim)", "",
              "| seed | Dice no-starve | Dice starve |", "|---|---|---|"]

    for s in seeds:
        lines.append(f"| {s} | {per_seed[s]['dice_iostar_nostarve']:.4f} | {per_seed[s]['dice_iostar_starve']:.4f} |")

    lines.append("")

    with open(run_dir / "underservice_starve.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    figure, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 5), squeeze=False)

    for ax, s in zip(axes[0], seeds):
        tn = per_seed[s]["traj_nostarve"]
        ts = per_seed[s]["traj_starve"]
        ax.plot([r["iter"] for r in tn], [r["lam_iostar"] for r in tn], marker=".", label="no-starve")
        ax.plot([r["iter"] for r in ts], [r["lam_iostar"] for r in ts], marker=".", label="starve x0.33")
        ax.axhline(1.0 / len(targets), color="black", linestyle="--", linewidth=1, alpha=0.6, label="uniform 1/D")
        ax.set_xlabel("iteration")
        ax.set_ylabel(f"lambda_{STARVED}")
        ax.set_title(f"seed {s}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    figure.suptitle(f"C-4: regret-game lambda_{STARVED} under starvation")
    figure.tight_layout()
    figure.savefig(run_dir / "visualization" / "lambda_iostar_trajectory.png", dpi=150)
    plt.close(figure)

    print("\n".join(lines))
    print(f"\n[C-4] artifacts under {run_dir}")


if __name__ == "__main__":
    main()
